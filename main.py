import os
import tempfile

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage,  # เผื่อใช้ส่งรูปประกอบ
)

from ml_model import predict_image


app = FastAPI()

CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    print("[WARN] LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET is empty!")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


@app.get("/")
def root():
    return {"status": "ok", "message": "ChilliBot AI is running on Render"}


# (ช่วยให้ทดสอบเองได้) เปิดได้ในเบราว์เซอร์
@app.get("/webhook")
def webhook_get():
    return PlainTextResponse("OK", status_code=200)


@app.post("/webhook")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    # ถ้าลายเซ็นไม่ถูกต้อง ควร 400
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        # อย่าปล่อย 400 เพราะ LINE จะมองว่า webhook ล้มเหลว (Verify อาจไม่ผ่าน)
        print("[ERROR] Webhook handler error:", e)
        return PlainTextResponse("OK", status_code=200)

    return PlainTextResponse("OK", status_code=200)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = event.message.text
    reply_text = f"คุณพิมพ์ว่า: {user_text}"

    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    message_id = event.message.id
    print(f"[IMG] Received image message id={message_id}")

    tmp_path = None
    try:
        with ApiClient(configuration) as api_client:
            blob_api = MessagingApiBlob(api_client)
            content = blob_api.get_message_content(message_id)

            # ✅ FIX
            image_bytes = content if isinstance(content, (bytes, bytearray)) else content.data

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = predict_image(tmp_path)

        # รองรับ predict_image ทั้งแบบ dict และ tuple
        if isinstance(result, dict):
            if result.get("ok"):
                msg1 = f"🔍 ผลวิเคราะห์: {result.get('disease_name')}\nความมั่นใจ: {result.get('confidence',0):.2f}%"
                msg2 = f"✅ คำแนะนำ:\n{result.get('advice','')}\n\nอ่านเพิ่มเติม: {result.get('info_url','')}"
                messages = [TextMessage(text=msg1), TextMessage(text=msg2)]
            else:
                messages = [TextMessage(text=result.get("disease_name","วิเคราะห์ไม่สำเร็จ"))]
        else:
            label, conf = result
            messages = [TextMessage(text=f"ผลวิเคราะห์: {label} (ความมั่นใจ {conf:.2f}%)")]

        with ApiClient(configuration) as api_client:
            msg_api = MessagingApi(api_client)
            msg_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=messages,
                )
            )

    except Exception as e:
        print("[ERROR] Image handler:", e)
        with ApiClient(configuration) as api_client:
            msg_api = MessagingApi(api_client)
            msg_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="ขออภัย ระบบวิเคราะห์รูปภาพขัดข้องชั่วคราว ลองใหม่อีกครั้งค่ะ")],
                )
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

            try:
                os.remove(tmp_path)
            except:
                pass
