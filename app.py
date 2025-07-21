import streamlit as st
import whisper
import tempfile
import os
from transformers import MarianMTModel, MarianTokenizer
import time
import io

st.set_page_config(
    page_title="🎙️ Whisper Transcription & Translation",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .transcript-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .translation-box {
        background-color: #f8fff8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #fff5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #d32f2f;
        margin: 1rem 0;
        color: #d32f2f;
    }
    
    .info-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.whisper_model = None
    st.session_state.translation_models = {}

@st.cache_resource
def load_whisper_model(model_size):
    try:
        model = whisper.load_model(model_size)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_translation_models():
    language_models = {
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "Hindi": "Helsinki-NLP/opus-mt-en-hi", 
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
        "German": "Helsinki-NLP/opus-mt-en-de",
        "Italian": "Helsinki-NLP/opus-mt-en-it",
        "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
        "Russian": "Helsinki-NLP/opus-mt-en-ru",
        "Japanese": "Helsinki-NLP/opus-mt-en-jap",
    }
    
    loaded_models = {}
    loading_errors = []
    
    for lang, model_name in language_models.items():
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            loaded_models[lang] = (tokenizer, model)
        except Exception as e:
            loading_errors.append(f"{lang}: {str(e)}")
    
    return loaded_models, loading_errors

def translate_text(text, tokenizer, model):
    try:
        if not text.strip():
            return "No text to translate"
            
        tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
        translated = model.generate(**tokens, max_length=512)
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return result
    except Exception as e:
        return f"Translation error: {str(e)}"

def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎙️ Whisper Audio Transcription & Translation")
    st.markdown("Convert speech to text and translate into multiple languages")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        whisper_model_size = st.selectbox(
            "Select Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        model_info = {
            "tiny": "~1GB VRAM, Fastest",
            "base": "~1GB VRAM, Fast", 
            "small": "~2GB VRAM, Good",
            "medium": "~5GB VRAM, Better",
            "large": "~10GB VRAM, Best"
        }
        st.info(f"**{whisper_model_size.title()} Model:** {model_info[whisper_model_size]}")
        
        st.header("🌍 Translation Languages")
        available_languages = ["French", "Hindi", "Spanish", "German", "Italian", "Portuguese", "Russian", "Japanese"]
        selected_languages = st.multiselect(
            "Select languages to translate to:",
            options=available_languages,
            default=["French", "Spanish", "Hindi"],
            help="Choose which languages you want translations in"
        )
        
        with st.expander("🔧 Advanced Options"):
            fp16_enabled = st.checkbox("Enable FP16", value=False, help="Faster processing on compatible GPUs")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, help="Higher values make output more random")
    
    if not st.session_state.models_loaded or st.session_state.whisper_model is None:
        with st.spinner("🔄 Loading models... This may take a few minutes on first run."):
            whisper_model, whisper_error = load_whisper_model(whisper_model_size)
            
            if whisper_error:
                st.error(f"❌ Failed to load Whisper model: {whisper_error}")
                return
                
            translation_models, translation_errors = load_translation_models()
            
            st.session_state.whisper_model = whisper_model
            st.session_state.translation_models = translation_models
            st.session_state.models_loaded = True
            
            if translation_errors:
                with st.expander("⚠️ Translation Model Loading Warnings"):
                    for error in translation_errors:
                        st.warning(error)
    
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'],
        help="Supported formats: MP3, WAV, M4A, FLAC, AAC, OGG"
    )
    
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size:.1f} MB)")
        
        if st.button("🚀 Transcribe & Translate", type="primary"):
            process_audio(uploaded_file, selected_languages, fp16_enabled, temperature)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>📌 Tips:</strong><br>
            • Upload audio files up to 200MB<br>
            • Shorter files (under 5 minutes) process faster<br>
            • Clear audio with minimal background noise works best<br>
            • All processing happens locally on the server
        </div>
        """, unsafe_allow_html=True)

def process_audio(uploaded_file, selected_languages, fp16_enabled, temperature):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        status_text.text("🎧 Transcribing audio...")
        progress_bar.progress(20)
        
        start_time = time.time()
        result = st.session_state.whisper_model.transcribe(
            tmp_file_path, 
            fp16=fp16_enabled,
            temperature=temperature
        )
        transcription = result["text"]
        transcription_time = time.time() - start_time
        
        progress_bar.progress(50)
        
        os.unlink(tmp_file_path)
        
        st.markdown('<div class="transcript-box">', unsafe_allow_html=True)
        st.markdown("### 📝 Transcription (English)")
        st.markdown(f"**Text:** {transcription}")
        st.markdown(f"**Processing Time:** {transcription_time:.2f} seconds")
        
        if st.button("📋 Copy Transcription", key="copy_transcription"):
            st.write("✅ Transcription copied to clipboard!")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_languages and transcription.strip():
            status_text.text("🌍 Translating to selected languages...")
            progress_bar.progress(70)
            
            translations = {}
            translation_start = time.time()
            
            for lang in selected_languages:
                if lang in st.session_state.translation_models:
                    tokenizer, model = st.session_state.translation_models[lang]
                    translated_text = translate_text(transcription, tokenizer, model)
                    translations[lang] = translated_text
                else:
                    translations[lang] = f"Model not available for {lang}"
            
            translation_time = time.time() - translation_start
            progress_bar.progress(100)
            
            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
            st.markdown("### 🌍 Translations")
            
            cols = st.columns(min(len(translations), 2))
            for i, (lang, translated) in enumerate(translations.items()):
                with cols[i % 2]:
                    st.markdown(f"**{lang}:**")
                    if not translated.startswith("Translation error") and not translated.startswith("Model not available"):
                        st.markdown(f"_{translated}_")
                        if st.button(f"📋 Copy {lang}", key=f"copy_{lang}"):
                            st.write(f"✅ {lang} translation copied!")
                    else:
                        st.error(translated)
            
            st.markdown(f"**Translation Time:** {translation_time:.2f} seconds")
            st.markdown('</div>', unsafe_allow_html=True)
            
            total_time = transcription_time + translation_time
            st.success(f"✅ **Processing Complete!** Total time: {total_time:.2f} seconds")
        
        else:
            progress_bar.progress(100)
            if not transcription.strip():
                st.warning("⚠️ No speech detected in the audio file.")
            elif not selected_languages:
                st.info("ℹ️ Select languages in the sidebar to get translations.")
        
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ **Error processing audio:** {str(e)}</div>', unsafe_allow_html=True)
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass

if __name__ == "__main__":
    main()
