import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import requests


MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "emotion-ferplus-8.onnx")
WEB_DIR = os.path.join(BASE_DIR, "web")


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


def ensure_model(model_path: str = DEFAULT_MODEL_PATH, model_url: str = MODEL_URL) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    with requests.get(model_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return model_path


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    tensor = face_resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=(0, 1))
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


def create_app() -> FastAPI:
    ensure_model(DEFAULT_MODEL_PATH)

    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(DEFAULT_MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"无法加载人脸分类器: {cascade_path}")

    app = FastAPI(title="表情识别服务")

    # 静态网页
    if os.path.isdir(WEB_DIR):
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/analyze")
    async def analyze(
        image: UploadFile = File(...),
        min_face_size: int = Form(64),
        threshold: float = Form(0.0),
    ):
        data = await image.read()
        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"faces": []}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_face_size, min_face_size))
        results = []
        for (x, y, w, h) in faces:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(frame.shape[1], int(x + w))
            y1 = min(frame.shape[0], int(y + h))
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            probs = infer_emotion(session, input_name, roi)
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            label = EMOTIONS_ZH[top_idx]
            if top_prob < threshold:
                label = "不确定"
            results.append({
                "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "emotion_index": top_idx,
                "emotion": label,
                "prob": top_prob,
            })

        return {"faces": results, "image_size": [int(frame.shape[1]), int(frame.shape[0])]}

    return app


app = create_app()

