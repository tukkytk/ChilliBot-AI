import os
import requests
from keras.models import load_model

MODEL_PATH = "VGG16_model.keras"
DRIVE_FILE_ID = "YOUR_MODEL_FILE_ID"   # เปลี่ยน!
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

model = None

def download_model():
    """ดาวน์โหลดโมเดลจาก Google Drive ถ้ายังไม่มีไฟล์"""
    if os.path.exists(MODEL_PATH):
        print("[ML] Model file exists. Skipping download.")
        return

    print("[ML] Downloading model from Google Drive...")
    response = requests.get(DRIVE_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("[ML] Model downloaded successfully.")

def load_ml_model():
    global model
    try:
        download_model()
        model = load_model(MODEL_PATH)
        print("[ML] Model Loaded Successfully.")
    except Exception as e:
        print("[ML ERROR] Cannot load model:", e)
        model = None

load_ml_model()

def predict_image(image_path):
    if model is None:
        return "ยังไม่ได้โหลดโมเดล (โหมดทดสอบ)", 0.00
    # ใส่โค้ด predict เดิมของคุณตรงนี้
