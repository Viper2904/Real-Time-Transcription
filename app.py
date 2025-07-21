from flask import Flask, request, render_template
import whisper
import tempfile
import os

from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

model = whisper.load_model("large")

language_models = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
}

loaded_models = {}
for lang, model_name in language_models.items():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translator_model = MarianMTModel.from_pretrained(model_name)
    loaded_models[lang] = (tokenizer, translator_model)


def translate_text(text, tokenizer, model):
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]


@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    translations = {}

    if request.method == "POST":
        if "audio" not in request.files:
            return "Please upload an audio file", 400

        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_file.save(tmp.name)
            result = model.transcribe(tmp.name, fp16=False)
            os.remove(tmp.name)
            transcription = result["text"]

            for lang, (tokenizer, model) in loaded_models.items():
                try:
                    translated_text = translate_text(transcription, tokenizer, model)
                    translations[lang] = translated_text
                except Exception as e:
                    translations[lang] = f"[Error: {e}]"

    return render_template("index.html", transcription=transcription, translations=translations)
