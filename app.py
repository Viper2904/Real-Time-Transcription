from flask import Flask, request, render_template, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("tiny")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return render_template("index.html", transcription="❌ No file uploaded.")

        audio_file = request.files["audio"]
        ext = os.path.splitext(audio_file.filename)[1]

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                audio_file.save(tmp.name)
                result = model.transcribe(tmp.name, fp16=False)
                os.remove(tmp.name)
                return render_template("index.html", transcription=result["text"])
        except Exception as e:
            return render_template("index.html", transcription=f"❌ Error: {str(e)}")

    return render_template("index.html")
