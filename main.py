import os
import tempfile

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)


from ml_model import predict_image  # ฟังก์ชันวิเคราะห์รูป

app = FastAPI()

# ====== ดึงค่า TOKEN/SECRET จาก Environment (Render) ======
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    print("[WARN] LINE token/secret is not set in environment variables.")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


@app.get("/")
def root():
    return {"status": "ok", "message": "ChilliBot AI is running"}


# ====== Webhook หลัก ======
@app.post("/webhook")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        # ถ้า signature ไม่ตรง หรือ handle ล้ม → จะมาที่นี่
        print("[ERROR] Webhook handler error:", e)
        raise HTTPException(status_code=400, detail="Webhook handler error")

    # สำคัญ: ต้องคืน 200 เสมอ
    return PlainTextResponse("OK", status_code=200)


# ====== กรณีข้อความ (Text) ======
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = event.message.text
    reply_text = f"คุณพิมพ์ว่า: {user_text}"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            reply_token=event.reply_token,
            messages=[{"type": "text", "text": reply_text}],
        )


# ====== กรณีรูปภาพ (Image) ======
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    message_id = event.message.id
    print(f"[IMG] Receive image message id={message_id}")

    # 1) โหลดรูปจาก LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        content = line_bot_api.get_message_content(message_id)

        # NOTE: ใน SDK v3 object นี้มักจะมีเมธอด iter_content()
        # คล้าย requests.Response
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            for chunk in content.iter_content():
                tmp.write(chunk)
            tmp_path = tmp.name

    # 2) ส่งให้โมเดลวิเคราะห์
    try:
        label, conf = predict_image(tmp_path)
        result_text = f"ผลวิเคราะห์: {label} (ความมั่นใจ {conf:.2f}%)"
    except Exception as e:
        print("[ERROR] predict_image:", e)
        result_text = "เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ"

    # 3) ตอบกลับไปที่ LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            reply_token=event.reply_token,
            messages=[{"type": "text", "text": result_text}],
        )
