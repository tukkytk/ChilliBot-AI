from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

# LINE BOT v3
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)

# ML model
from ml_model import predict_image

# ================================
# 1) LINE TOKEN (แก้แล้ว / ไม่มีตัวอักษรเกิน)
# ================================
LINE_CHANNEL_ACCESS_TOKEN = "+gTRyq3Vi4Wep9TNuDkGSDTcxvQUlLBcbr4AKV8CtYisOXvn53RLmDmsnnnyfaGAentBoAXwByF8Pq89IVJap/cW1ioqcTr2Nlj6qagTdcPgCC7gwFC3NMHi2dsIld9bOSsPmTo8ztGqBVt3B8171gdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "bace529b0f4156e5f3a7bd5ff68ea8cd"

# ================================
# 2) FASTAPI
# ================================
app = FastAPI()

# ทำโฟลเดอร์ static ถ้ายังไม่มี
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ================================
# 3) LINE API INITIALIZE
# ================================
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
client = ApiClient(configuration)

messaging_api = MessagingApi(client)
blob_api = MessagingApiBlob(client)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================================
# 4) ROUTES
# ================================
@app.get("/")
def index():
    return {"status": "ChilliBot AI is running with VGG16!"}

@app.post("/webhook")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = (await request.body()).decode("utf-8")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return JSONResponse(status_code=400, content={"error": "Invalid signature"})

    return "OK"

# ================================
# 5) TEXT MESSAGE HANDLER
# ================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    reply_text = f"คุณพิมพ์ว่า: {event.message.text}"

    with ApiClient(configuration) as client:
        MessagingApi(client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

# ================================
# 6) IMAGE MESSAGE HANDLER + ML Model
# ================================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    try:
        # โหลดไฟล์ภาพจาก LINE
        response = blob_api.get_message_content(event.message.id)
        
        # FIX สำคัญมาก!! ต้องใช้ .body
        file_bytes = response.body
        
        input_path = "static/input.jpg"
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # วิเคราะห์ด้วยโมเดล AI
        label, confidence = predict_image(input_path)

        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text=f"ผลลัพธ์: {label}\nความมั่นใจ: {confidence:.2f}%"
                )]
            )
        )

    except Exception as e:
        print("❌ ERROR handle_image:", e)

