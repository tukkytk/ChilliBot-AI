import os
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===== ตั้งค่าโมเดล =====
MODEL_PATH = "ChiliDisease7_finetune.keras"

# >>> แก้ให้เป็น URL ของ GitHub Release ของคุณจริง ๆ <<<
GITHUB_MODEL_URL = "https://github.com/tukkytk/ChilliBot-AI/releases/download/chilli/ChiliDisease7_finetune.keras"


model = None
MODEL_READY = False

# ปรับตามคลาสจริงของโมเดลคุณ
CLASS_NAMES = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
]


def download_model():
    """ดาวน์โหลดโมเดลจาก GitHub Release ถ้ายังไม่มีไฟล์"""
    if os.path.exists(MODEL_PATH):
        print(f"[ML] Model file '{MODEL_PATH}' already exists. Skip download.")
        return

    if not GITHUB_MODEL_URL:
        print("[ML] GITHUB_MODEL_URL is not set. Running in NO-ML mode.")
        return

    print("[ML] Downloading model from GitHub Release...")
    try:
        with requests.get(GITHUB_MODEL_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = 0
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
        print(f"[ML] Model downloaded successfully. Size: {total / (1024*1024):.2f} MB")
    except Exception as e:
        print("[ML ERROR] Failed to download model:", e)


def load_ml_model():
    """โหลดโมเดลจากไฟล์"""
    global model, MODEL_READY

    download_model()

    if not os.path.exists(MODEL_PATH):
        print(f"[ML] Model file '{MODEL_PATH}' not found. Running in NO-ML mode.")
        MODEL_READY = False
        model = None
        return

    try:
        print(f"[ML] Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        MODEL_READY = True
        print("[ML] Model Loaded Successfully.")
    except Exception as e:
        print("[ML ERROR] Failed to load model:", e)
        MODEL_READY = False
        model = None


def _preprocess_image(image_path: str, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_path: str):
    if not MODEL_READY or model is None:
        return "ยังไม่ได้โหลดโมเดล (โหมดทดสอบ)", 0.0

    try:
        x = _preprocess_image(image_path)
        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        conf = float(preds[idx] * 100.0)
        return label, conf
    except Exception as e:
        print("[ML ERROR] Prediction failed:", e)
        return "ไม่สามารถวิเคราะห์ภาพได้", 0.0


# โหลดโมเดลตั้งแต่ตอน import
load_ml_model()
