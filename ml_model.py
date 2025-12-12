import os
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===== ตั้งค่าโมเดล =====
MODEL_PATH = "ChiliDisease7_finetune.keras"

# URL ของไฟล์โมเดลบน GitHub Release
GITHUB_MODEL_URL = "https://github.com/tukkytk/ChilliBot-AI/releases/download/chilli/ChiliDisease7_finetune.keras"

model = None
MODEL_READY = False

# ===== ข้อมูลโรค: ตั้งตาม class ที่โมเดลของคุณเทรนไว้ =====
# class_id 0–6 ให้ปรับชื่อโรค/รายละเอียด/คำแนะนำ/ลิงก์/รูปตามจริงได้เลย
DISEASE_INFO = {
    0: {
        "name": "โรคใบจุดพริกจากเชื้อรา",
        "description": (
            "ใบพริกมีจุดสีน้ำตาลเข้ม หรือน้ำตาลไหม้เป็นวง ๆ กระจายทั่วใบ "
            "อาจมีวงแหวนซ้อนกัน ใบเหลืองและร่วง ทำให้ต้นอ่อนแอและผลผลิตลดลง"
        ),
        "advice": (
            "• ตัดใบและส่วนที่เป็นโรคออกจากแปลง แล้วนำไปฝังหรือเผาทำลายนอกแปลง\n"
            "• หลีกเลี่ยงการรดน้ำพรมโดนใบ ลดความชื้นสะสมในแปลงปลูก\n"
            "• ใช้สารป้องกันกำจัดเชื้อราในกลุ่มที่กรมวิชาการเกษตรขึ้นทะเบียน "
            "อ่านฉลากและปฏิบัติตามคำแนะนำอย่างเคร่งครัด"
        ),
        "image_url": "",  # ใส่ลิงก์รูปตัวอย่างโรคใบจุด ถ้ามี
        "info_url": "https://www.doa.go.th"
    },
    1: {
        "name": "โรคใบไหม้ / กุ้งแห้งพริก",
        "description": (
            "ขอบใบไหม้สีน้ำตาลเข้ม ลุกลามจากปลายใบเข้าไปด้านใน "
            "ผลพริกแห้งเหี่ยวคล้ายกุ้งแห้ง ต้นทรุดโทรมเร็วโดยเฉพาะในช่วงฝนชุก"
        ),
        "advice": (
            "• เก็บผลพริกและใบที่เป็นโรคออกจากต้นทันที เพื่อลดแหล่งสะสมเชื้อ\n"
            "• จัดแถวปลูกให้โปร่ง แดดและลมพอผ่านได้ ลดความชื้นแฉะในแปลง\n"
            "• พ่นสารป้องกันเชื้อราให้ถูกต้องตามคำแนะนำของเจ้าหน้าที่เกษตรพื้นที่"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
    2: {
        "name": "โรคแอนแทรคโนส (ผลเน่าพริก)",
        "description": (
            "ผลพริกมีแผลบุ๋มสีน้ำตาลหรือดำ กึ่งกลางแผลมีผงสปอร์สีส้ม "
            "แผลขยายตัวจนผลเน่าติดกันเป็นกลุ่ม ทำให้เสียหายมากในช่วงฝนตกชุก"
        ),
        "advice": (
            "• คัดผลที่เป็นโรคออกจากต้นและแปลงปลูก อย่าทิ้งไว้บนพื้นดิน\n"
            "• เก็บเกี่ยวผลที่แก่พอเหมาะให้เร็วขึ้น เพื่อลดโอกาสถูกเชื้อเข้าทำลาย\n"
            "• ใช้สารป้องกันกำจัดเชื้อราในกลุ่มที่เหมาะสม หมุนเวียนตัวยาเพื่อลดการดื้อยา"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
    3: {
        "name": "โรคราแป้งพริก",
        "description": (
            "ผิวใบมีผงสีขาวคล้ายแป้งปกคลุมทั้งด้านบนหรือด้านล่างของใบ "
            "ใบเหลือง บิดงอ การสังเคราะห์แสงลดลง ทำให้ต้นแคระและให้ผลผลิตต่ำ"
        ),
        "advice": (
            "• เพิ่มการถ่ายเทอากาศในแปลง ไม่ปลูกพริกชิดเกินไป\n"
            "• หลีกเลี่ยงการให้น้ำในช่วงเย็นจนใบไม่แห้งก่อนมืด\n"
            "• ใช้สารป้องกันกำจัดราแป้งที่ขึ้นทะเบียนอย่างถูกต้องตามฉลาก"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
    4: {
        "name": "โรครากเน่า–โคนเน่าพริก",
        "description": (
            "โคนต้นมีสีน้ำตาลหรือดำ เน่า หรือแห้งยุ่ย ใบเหลืองเหี่ยวทั้งต้น "
            "เมื่อถอนดูจะเห็นรากเน่าเสียหาย มักเกิดในดินชื้นแฉะระบายน้ำไม่ดี"
        ),
        "advice": (
            "• ถอนต้นที่เป็นโรคทั้งรากออกจากแปลง แล้วปรับปรุงดินให้ระบายน้ำดี\n"
            "• ยกร่องปลูกให้สูง ดินโปร่ง ร่วนซุย ไม่อุ้มน้ำเกินไป\n"
            "• ใช้เชื้อราปฏิปักษ์ไตรโคเดอร์มา หรือสารเคมีที่ขึ้นทะเบียนสำหรับโรครากเน่า–โคนเน่า"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
    5: {
        "name": "โรคใบด่างพริกจากเชื้อไวรัส",
        "description": (
            "ใบมีลายด่างเขียว–เหลือง บางใบแคบและบิดงอ ต้นเตี้ย แคระแกร็น "
            "ผลพริกบิดเบี้ยว คุณภาพลดลง เพลี้ยอ่อนมักเป็นพาหะสำคัญนำโรค"
        ),
        "advice": (
            "• กำจัดเพลี้ยอ่อนและแมลงพาหะด้วยกับดักกาวเหนียวสีเหลือง หรือสารกำจัดแมลงอย่างเหมาะสม\n"
            "• ถอนต้นที่เป็นโรครุนแรงออกจากแปลง เพื่อลดการแพร่ระบาด\n"
            "• ใช้เมล็ดพันธุ์และต้นกล้าที่ปลอดโรค และปลูกหมุนเวียนกับพืชอื่นลดการสะสมของเชื้อไวรัส"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
    6: {
        "name": "โรคใบหงิกเหลืองพริก (Leaf Curl Virus)",
        "description": (
            "ใบหงิกม้วน ขอบใบงุ้มลง เหลืองทั้งใบ แตกยอดแคระแกร็นทั้งต้น "
            "มักพบแมลงหวี่ขาวเป็นพาหะนำเชื้อไวรัสเข้าทำลาย"
        ),
        "advice": (
            "• ควบคุมแมลงหวี่ขาวด้วยมุ้งกันแมลง กับดักกาวเหนียวสีเหลือง หรือสารกำจัดแมลงอย่างถูกต้อง\n"
            "• ตัดยอดและใบที่แสดงอาการรุนแรงออก แล้วนำไปทำลายนอกแปลง\n"
            "• หลีกเลี่ยงการปลูกซ้ำในพื้นที่เดิมที่เคยระบาดรุนแรง และปลูกหมุนเวียนกับพืชอื่น"
        ),
        "image_url": "",
        "info_url": "https://www.doa.go.th"
    },
}


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
    """
    คืนค่า dict ข้อมูลโรค เช่น:
    {
        "ok": True,
        "class_id": 6,
        "confidence": 96.94,
        "disease_name": "...",
        "description": "...",
        "advice": "...",
        "image_url": "...",
        "info_url": "..."
    }
    """
    if not MODEL_READY or model is None:
        return {
            "ok": False,
            "class_id": None,
            "confidence": 0.0,
            "disease_name": "ยังไม่ได้โหลดโมเดล",
            "description": "ระบบกำลังทำงานในโหมดทดสอบ (NO-ML)",
            "advice": "กรุณาติดต่อผู้ดูแลระบบเพื่อตรวจสอบการโหลดโมเดล",
            "image_url": "",
            "info_url": ""
        }

    try:
        x = _preprocess_image(image_path)
        preds = model.predict(x)[0]
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100.0)

        info = DISEASE_INFO.get(
            class_id,
            {
                "name": "ไม่ทราบชนิดโรค",
                "description": "ไม่พบข้อมูลโรคที่ตรงกับผลวิเคราะห์ในระบบ",
                "advice": "ลองถ่ายภาพให้ชัดเจนขึ้น หรือปรึกษาเจ้าหน้าที่เกษตรในพื้นที่",
                "image_url": "",
                "info_url": "https://www.doa.go.th"
            }
        )

        return {
            "ok": True,
            "class_id": class_id,
            "confidence": confidence,
            "disease_name": info["name"],
            "description": info["description"],
            "advice": info["advice"],
            "image_url": info["image_url"],
            "info_url": info["info_url"]
        }

    except Exception as e:
        print("[ML ERROR] Prediction failed:", e)
        return {
            "ok": False,
            "class_id": None,
            "confidence": 0.0,
            "disease_name": "ไม่สามารถวิเคราะห์ภาพได้",
            "description": "อาจเกิดจากไฟล์ภาพเสีย หรือรูปไม่ชัดเจน",
            "advice": "ลองถ่ายภาพใหม่ให้เห็นใบ/ผลพริกชัด ๆ แล้วส่งอีกครั้ง",
            "image_url": "",
            "info_url": ""
        }


# โหลดโมเดลตั้งแต่ตอน import
load_ml_model()
