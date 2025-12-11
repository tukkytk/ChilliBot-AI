import os
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # ใช้ tf.keras เพราะมีใน tensorflow

# ===== ตั้งค่าโมเดล =====
MODEL_PATH = "ChiliDisease7_finetune.keras"

# >>>>> แก้ตรงนี้ให้เป็น Google Drive FILE ID ของคุณ <<<<<
DRIVE_FILE_ID = "1Of5dcV9FvWUR0iIs-VyrboTKOzZic537"
DRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

model = None
MODEL_READY = False

# กำหนดชื่อคลาสตามโมเดลของคุณ
# ถ้ายังไม่แน่ใจ ใช้เป็น placeholder ไปก่อนก็ได้
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
    """ดาวน์โหลดโมเดลจาก Google Drive ถ้ายังไม่มีไฟล์ในเครื่อง"""
    if os.path.exists(MODEL_PATH):
        print(f"[ML] Model file '{MODEL_PATH}' already exists. Skip download.")
        return

    if not DRIVE_FILE_ID or "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" in DRIVE_FILE_ID:
        print("[ML] DRIVE_FILE_ID is not set. Running in NO-ML mode.")
        return

    print("[ML] Downloading model from Google Drive...")
    try:
        with requests.get(DRIVE_DOWNLOAD_URL, stream=True, timeout=600) as r:
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

    # ถ้าไม่มีไฟล์ ให้ดาวน์โหลดมาก่อน
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
    """เตรียมรูปให้เข้ากับโมเดล VGG16 (หรือโมเดลที่คุณใช้)"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_path: str):
    """
    คืนค่า (label, confidence)

    ถ้าไม่มีโมเดล → คืน ("ยังไม่ได้โหลดโมเดล (โหมดทดสอบ)", 0.0)
    """
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
