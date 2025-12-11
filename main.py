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

from ml_model import predict_image  # ใช้แล้วหรือไม่ใช้ก็ได้ (ตอนนี้ NO-ML mode)

# ================== FastAPI App ==================
app = FastAPI()

# ================== LINE Config ==================
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    print("[WARN] LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET is empty!")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


@app.get("/")
def root():
    return {"status": "ok", "message": "ChilliBot AI is running on Render"}


# ================== Webhook ==================
@app.post("/webhook")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        print("[ERROR] Webhook Error:", e)
        # ถ้า error ภายใน handler ให้ส่ง 400 กลับ LINE
        raise HTTPException(status_code=400, detail="Webhook handler error")

    return PlainTextResponse("OK", status_code=200)


# ================== Text Message Handler ==================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = event.message.text
    reply_text = f"คุณพิมพ์ว่า: {user_text}"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


# ================== Image Message Handler ==================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    message_id = event.message.id
    print(f"[IMG] Received image message id={message_id}")

    # 1) ดึงภาพจาก LINE ด้วย MessagingApi v3
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        content = api.get_message_content(message_id)

        # content.body = bytes stream
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            for chunk in content.body:
                tmp.write(chunk)
            tmp_path = tmp.name

    # 2) วิเคราะห์ภาพ
    try:
        label, conf = predict_image(tmp_path)
        result_text = f"ผลวิเคราะห์: {label} (ความมั่นใจ {conf:.2f}%)"
    except Exception as e:
        print("[ERROR] predict_image:", e)
        result_text = "ไม่สามารถวิเคราะห์รูปภาพได้ในตอนนี้"

    # 3) ตอบกลับ
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=result_text)]
            )
        )

