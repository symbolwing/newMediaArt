import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests


MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "emotion-ferplus-8.onnx")


# FER+ emotion classes (8)
EMOTIONS_EN = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt",
]

EMOTIONS_ZH = [
    "中性",
    "高兴",
    "惊讶",
    "伤心",
    "愤怒",
    "厌恶",
    "恐惧",
    "轻蔑",
]


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    probs: np.ndarray  # smoothed probs, shape (8,)
    last_seen_frame: int


def ensure_model(model_path: str = DEFAULT_MODEL_PATH, model_url: str = MODEL_URL) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    print(f"[下载] 正在下载表情识别模型到: {model_path}")
    try:
        with requests.get(model_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            chunk = 8192
            with open(model_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk):
                    if not data:
                        continue
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        done = int(50 * downloaded / total)
                        sys.stdout.write("\r[" + "#" * done + "." * (50 - done) + "]")
                        sys.stdout.flush()
        print("\n[完成] 模型下载完成")
    except Exception as e:
        print(f"[错误] 模型下载失败: {e}")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception:
                pass
        raise
    return model_path


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    # Normalize to [0, 1]
    tensor = face_resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=(0, 1))  # (1,1,64,64)
    return tensor


def infer_emotion(session: ort.InferenceSession, input_name: str, face_gray: np.ndarray) -> np.ndarray:
    input_blob = preprocess_face(face_gray)
    outputs = session.run(None, {input_name: input_blob})
    logits = outputs[0]
    if logits.ndim > 2:
        logits = logits.reshape(-1)
    else:
        logits = logits[0]
    probs = softmax(logits)
    return probs


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union


def draw_label(img: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x, y - th - baseline - 6), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="实时摄像头人脸表情识别 (FER+ ONNX + OpenCV)")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--process-every", type=int, default=1, help="每 N 帧执行一次情绪推理，默认 1（每帧）")
    parser.add_argument("--alpha", type=float, default=0.6, help="EMA平滑系数，0-1，越大越跟随当前帧")
    parser.add_argument("--threshold", type=float, default=0.35, help="显示标签的最小置信度阈值")
    parser.add_argument("--width", type=int, default=0, help="期望摄像头帧宽度；0 表示不强制")
    parser.add_argument("--height", type=int, default=0, help="期望摄像头帧高度；0 表示不强制")
    parser.add_argument("--min-face-size", type=int, default=64, help="最小人脸尺寸（像素）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="ONNX模型路径（默认自动下载）")
    args = parser.parse_args()

    model_path = ensure_model(args.model)

    providers = [
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]
    try:
        session = ort.InferenceSession(model_path, providers=[p[0] for p in providers if p[0] in ort.get_available_providers()])
    except Exception:
        # fallback CPU
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Face detector (Haar cascade)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[错误] 无法加载人脸分类器: ", cascade_path)
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera_index)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头，请检查索引或权限。")
        sys.exit(1)

    print("按 'q' 退出窗口")

    frame_idx = 0
    next_track_id = 1
    tracks: List[Track] = []
    max_idle_frames = 60
    iou_match_threshold = 0.3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 读取帧失败，结束。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(args.min_face_size, args.min_face_size)
            )

            # match detections to existing tracks by IoU
            assigned = [False] * len(faces)
            for track in tracks:
                best_iou = 0.0
                best_j = -1
                for j, (x, y, w, h) in enumerate(faces):
                    if assigned[j]:
                        continue
                    score = iou(track.bbox, (x, y, w, h))
                    if score > best_iou:
                        best_iou = score
                        best_j = j
                if best_j >= 0 and best_iou >= iou_match_threshold:
                    track.bbox = tuple(int(v) for v in faces[best_j])
                    track.last_seen_frame = frame_idx
                    assigned[best_j] = True
                # else keep track as is; may be removed later if idle

            # create new tracks for unassigned detections
            for j, (x, y, w, h) in enumerate(faces):
                if not assigned[j]:
                    new_track = Track(track_id=next_track_id, bbox=(int(x), int(y), int(w), int(h)), probs=np.zeros(8, dtype=np.float32), last_seen_frame=frame_idx)
                    tracks.append(new_track)
                    next_track_id += 1

            # remove idle tracks
            tracks = [t for t in tracks if (frame_idx - t.last_seen_frame) <= max_idle_frames]

            # run inference optionally every N frames
            should_infer = (frame_idx % max(1, args.process_every) == 0)
            for track in tracks:
                x, y, w, h = track.bbox
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame.shape[1], x + w)
                y1 = min(frame.shape[0], y + h)
                face_gray = gray[y0:y1, x0:x1]
                if face_gray.size == 0:
                    continue
                if should_infer:
                    probs = infer_emotion(session, input_name, face_gray)
                    if track.probs.sum() == 0:
                        track.probs = probs
                    else:
                        track.probs = args.alpha * probs + (1.0 - args.alpha) * track.probs

                # draw bbox and label
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if track.probs.sum() > 0:
                    top_idx = int(np.argmax(track.probs))
                    top_prob = float(track.probs[top_idx])
                    label = EMOTIONS_ZH[top_idx] if top_prob >= args.threshold else "不确定"
                    draw_label(frame, f"ID {track.track_id}: {label} {top_prob*100:.1f}%", (x0, y0), (0, 255, 255))
                else:
                    draw_label(frame, f"ID {track.track_id}: 识别中…", (x0, y0), (0, 255, 255))

            cv2.imshow("表情识别", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

