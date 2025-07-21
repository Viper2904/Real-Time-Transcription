import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import itertools
import os
import time
import signal
import sys

model = whisper.load_model("tiny")
q = queue.Queue()
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n Stopping transcription...")
    running = False
    sys.exit(0)

def callback(indata, frames, time, status):
    """Audio callback function"""
    if status:
        print("Audio status:", status)
    if running:
        q.put(indata.copy())

def transcribe_stream():
    """Main transcription loop"""
    print("🎤 Listening... (transcribing every 5 seconds)")
    print("📝 Press Ctrl+C to stop")
    
    buffer = np.empty((0,), dtype=np.float32)
    file_counter = itertools.cycle([1, 2])
    
    while running:
        try:
            data = q.get(timeout=1.0)
            data = data.flatten()
            buffer = np.concatenate([buffer, data])
            
            if len(buffer) >= 16000 * 5:
                audio_segment = buffer[:16000 * 5]
                buffer = buffer[16000 * 5:] 
                filename = f"temp{next(file_counter)}.wav"
                
                try:
                    import scipy.io.wavfile as wav
                    wav.write(filename, 16000, audio_segment.astype(np.float32))
                    
                    result = model.transcribe(filename, fp16=False)
                    transcribed_text = result["text"].strip()
                    
                    if transcribed_text:
                        print(f"🗣️  You said: {transcribed_text}")
                    else:
                        print("🔇 [No speech detected]")
                    
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
                        
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
                        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Stream processing error: {e}")
            break

def main():
    """Main function"""
    global running
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🔄 Initializing audio stream...")
    
    try:
        devices = sd.query_devices()
        default_device = sd.default.device[0] 
        print(f"🎧 Using audio device: {devices[default_device]['name']}")
        
        with sd.InputStream(
            callback=callback, 
            channels=1, 
            samplerate=16000,
            blocksize=1024,  
            dtype=np.float32
        ) as stream:
            
            print("✅ Audio stream started successfully")
            transcription_thread = threading.Thread(target=transcribe_stream, daemon=True)
            transcription_thread.start()
            try:
                while running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                running = False
                
    except Exception as e:
        print(f"❌ Failed to initialize audio stream: {e}")
        print("🔧 Try checking your microphone permissions or audio device settings")
        return False
    
    print("👋 Transcription stopped")
    return True

if __name__ == "__main__":
    main()
