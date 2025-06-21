from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)

# ✅ Allow CORS for localhost:*
CORS(app, origins=["http://localhost", "http://localhost:3000"], supports_credentials=True)

# 🟢 Load a small CPU-friendly model (change if you want)
model = whisper.load_model("tiny")  # options: tiny, base, small, medium, large

@app.route("/stt", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "no audio provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        result = model.transcribe(tmp.name)

    os.remove(tmp.name)

    return jsonify({"text": result["text"]})

# ✅ Handle OPTIONS requests properly for CORS preflight
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3662, debug=True)
