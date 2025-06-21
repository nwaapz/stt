# server.py
# (Creating this file on the server)
# ----------------------------------
# Option A: Using cat and heredoc:
# 1. SSH into your server and navigate to your project folder:
#    cd ~/whisper-stt
# 2. Run:
#    cat > server.py << 'EOF'
#    (paste the rest of this file contents here)
#    EOF
#
# Option B: Using a text editor (nano):
# 1. cd ~/whisper-stt
# 2. nano server.py
# 3. Paste contents, then save with Ctrl+O, Enter, and exit with Ctrl+X
#
# Now ensure the file exists and has the correct content:
#    ls -l server.py
#
# Then make sure your virtualenv is activated and proceed to run the server.

# server.py
# Self-hosted Whisper STT server using Flask on port 3662
# -------------------------------------------------------
# 1. Place this file in your project directory (e.g., ~/whisper-stt).
# 2. Ensure your venv is activated with whisper and flask installed.
# 3. Run: export FLASK_APP=server && flask run --host=0.0.0.0 --port=3662

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)
# Load a small CPU-friendly model:
model = whisper.load_model("tiny")  # options: tiny, base, small, medium, large

@app.route("/stt", methods=["POST"])
def transcribe():
    # Check for audio part
    if "audio" not in request.files:
        return jsonify({"error": "no audio provided"}), 400

    # Save incoming file to a temp path
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        # Run transcription
        result = model.transcribe(tmp.name)
    # Clean up temp file
    os.remove(tmp.name)

    # Return only the text
    return jsonify({"text": result["text"]})

if __name__ == "__main__":
    # Debug mode for testing on port 3662
    app.run(host="0.0.0.0", port=3662, debug=True)



