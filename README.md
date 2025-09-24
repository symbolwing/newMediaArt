# 实时摄像头人脸表情识别（OpenCV + ONNXRuntime + FER+）

本工具通过电脑摄像头实时检测人脸，并使用 ONNX 格式的 FER+ 模型推理 8 类表情：`中性 / 高兴 / 惊讶 / 伤心 / 愤怒 / 厌恶 / 恐惧 / 轻蔑`。内置轻量的人脸检测（OpenCV HaarCascade），整体依赖小、易于部署。

## 特性
- 多人脸检测与跟踪关联（IoU 简易匹配）
- 表情预测概率的指数移动平均平滑（可调 alpha）
- 可配置帧间隔推理，降低 CPU 占用
- CLI 参数可调：摄像头索引、分辨率、最小人脸尺寸、阈值等
- 首次运行自动下载 ONNX 模型（FER+）

## 安装
建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 运行
```bash
python src/main.py --camera-index 0 --process-every 2 --alpha 0.6 --threshold 0.35 --width 1280 --height 720
```

- 按 `q` 退出
- 首次运行会在 `models/` 目录自动下载 `emotion-ferplus-8.onnx`

## 常用参数
- `--camera-index`: 摄像头索引，默认 0
- `--process-every`: 每 N 帧进行一次情绪推理，默认 1（每帧）
- `--alpha`: 概率 EMA 平滑系数，范围 0-1，越大越跟随当前帧
- `--threshold`: 置信度阈值，低于该值显示“不确定”
- `--width / --height`: 期望的摄像头分辨率（0 表示不强制）
- `--min-face-size`: 人脸最小尺寸像素，默认 64
- `--model`: 指定自定义 ONNX 模型路径（默认自动下载）

## 说明
- 该实现使用 OpenCV Haar 级联进行人脸检测，速度快且无需额外权重，但在复杂光照/角度下可能略逊于更先进的检测器（如 MediaPipe、DNN）。如需更高精度可以替换检测器。
- FER+ 模型输入为 64x64 灰度图，本实现做了 0-1 归一化；不同预处理会影响概率分布，但整体分类效果稳定。
- 平滑通过 EMA 降低抖动；如需要响应更快可减小 `--alpha` 或将 `--process-every` 调整为 1。

## 许可证
MIT