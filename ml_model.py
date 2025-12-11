import os
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "ChiliDisease7_finetune.keras"  # ตามชื่อไฟล์จริงของคุณ
DRIVE_FILE_ID = "1Of5dcV9FvWUR0iIs-VyrboTKOzZic537"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download"

model = None
MODEL_READY = False

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
    """ดาวน์โหลดโมเดลจาก Google Drive (รองรับไฟล์ใหญ่ / confirm token)"""
    if os.path.exists(MODEL_PATH):
        print(f"[ML] Model file '{MODEL_PATH}' already exists. Skip download.")
        return

    if not DRIVE_FILE_ID:
        print("[ML] DRIVE_FILE_ID is not set. Running in NO-ML mode.")
        return

    print("[ML] Downloading model from Google Drive...")

    session = requests.Session()
    params = {"id": DRIVE_FILE_ID, "export": "download"}

    # เรียกครั้งแรก
    response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)
    token = None

    # หา confirm token ถ้า Google ให้ยืนยัน
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        print("[ML] Found confirm token, retrying download...")
        params["confirm"] = token
        response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

    try:
        response.raise_for_status()
    except Exception as e:
        print("[ML ERROR] Failed to download model:", e)
        return

    total = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    print(f"[ML] Model downloaded successfully. Size: {total / (1024*1024):.2f} MB")


def load_ml_model():
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


# เรียกโหลดโมเดลตอน import
load_ml_model()
