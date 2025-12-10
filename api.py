from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def predict():
    data = request.json
    user_message = data.get("input", "")
    response = f"คุณกล่าวว่า: {user_message}"
    return jsonify({"output": response})

if __name__ == "__main__":
    app.run(port=8000)


