from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

app = Flask(__name__)

# Replace with your actual access token and channel secret
ACCESS_TOKEN = '+gTRyq3Vi4Wep9TNuDkGSDTcxvQUlLBcbr4AKV8CtYisOXvn53RLmDmsnnnyfaGAentBoAXwByF8Pq89IVJap/cW1ioqcTr2Nlj6qagTdcPgCC7gwFC3NMHi2dsIld9bOSsPmTo8ztGqBVt3B8171gdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = 'bace529b0f4156e5f3a7bd5ff68ea8cd'

configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


# ตัวอย่างโมเดล (คุณสามารถใช้ Machine Learning Model หรือฟังก์ชันอื่นได้)
def get_response_from_model(input_text):
    # ตัวอย่างฟังก์ชันโมเดล
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm doing great! Thanks for asking.",
        "bye": "Goodbye! Have a great day!",
    }
    # ตรวจสอบข้อความที่ส่งมา และตอบกลับ
    return responses.get(input_text.lower(), "Sorry, I don't understand that.")


@app.route("/webhook", methods=['POST'])
def callback():
    # Get X-Line-Signature header value
    signature = request.headers.get('X-Line-Signature', '')

    # Get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # Handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text  # ข้อความที่ผู้ใช้ส่งมา
    response_message = get_response_from_model(user_message)  # ใช้โมเดลดึงข้อความตอบกลับ

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        try:
            # ส่งข้อความตอบกลับ
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=response_message)]
                )
            )
        except Exception as e:
            app.logger.error(f"Error replying message: {e}")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
