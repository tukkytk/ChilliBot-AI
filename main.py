from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import MessagingApi, ApiClient, Configuration
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import uvicorn
import os

app = FastAPI()

CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

handler = WebhookHandler(CHANNEL_SECRET)

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

@app.post("/webhook")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        print("Webhook Error:", e)
        raise HTTPException(status_code=400, detail="Webhook handler error")

    return "OK"   # IMPORTANT: must return OK


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    print("Received event:", event)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            event.reply_token,
            [
                {
                    "type": "text",
                    "text": f"You said: {event.message.text}"
                }
            ]
        )


@app.get("/")
def root():
    return {"msg": "ChilliBot AI is running."}
