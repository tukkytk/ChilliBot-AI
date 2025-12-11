import os
import numpy as np
from PIL import Image

# พยายาม import load_model ถ้ามี TensorFlow
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None
    print("[ML] TensorFlow/Keras not available:", e)

MODEL_PATH = "VGG16_model.keras"
model = None
MODEL_READY = False

# ===== โหลดโมเดลแบบปลอดภัย =====
if load_model is not None and os.path.exists(MODEL_PATH):
    try:
        print("[ML] Loading model from", MODEL_PATH)
        model = load_model(MODEL_PATH)
        MODEL_READY = True
        print("[ML] Model Loaded Successfully.")
    except Exception as e:
        print("[ML] Failed to load model:", e)
else:
    print(f"[ML] Model file '{MODEL_PATH}' not found. Running in NO-ML mode.")

# TODO: แก้ชื่อ class ให้ตรงกับโมเดลจริงของคุณ
CLASS_NAMES = ["Class A", "Class B", "Class C"]


def _preprocess_image(image_path: str, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_path: str):
    """
    คืนค่า (label, confidence)
    ถ้าไม่มีโมเดล → คืนข้อความโหมดทดสอบ + 0.0
    """
    if not MODEL_READY or model is None:
        return "ยังไม่ได้โหลดโมเดล (โหมดทดสอบ)", 0.0

    x = _preprocess_image(image_path)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(preds[idx] * 100.0)
    return label, conf
