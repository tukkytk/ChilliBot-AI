import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageDraw

MODEL_PATH = "VGG16_model.keras"

CLASS_LABELS = ["พริกดี", "พริกเสีย", "โรคใบไหม้"]

print("[ML] Loading model from", MODEL_PATH)
model = load_model(MODEL_PATH)
print("[ML] Model Loaded Successfully.")


def predict_image(image_path):
    print("[ML] Predicting image:", image_path)

    img = image.load_img(image_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)
    print("[ML] Softmax prediction:", preds)

    idx = np.argmax(preds)
    confidence = preds[0][idx]

    label = CLASS_LABELS[idx]
    print(f"[ML] Final Prediction → {label} (index {idx})")

    return label, float(confidence)

def draw_bounding_box(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]
    draw.rectangle(box, outline="red", width=6)
    return img
