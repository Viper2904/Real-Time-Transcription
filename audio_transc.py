import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import itertools

model = whisper.load_model("tiny") 
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    q.put(indata.copy())

def transcribe_stream():
    print(" Listening... (every 5 seconds)")
    buffer = np.empty((0,), dtype=np.float32)
    file_counter = itertools.cycle([1, 2])

    while True:
        data = q.get()
        data = data.flatten()
        buffer = np.concatenate([buffer, data])

        if len(buffer) >= 16000 * 5:
            audio_segment = buffer[:16000 * 5]
            buffer = buffer[16000 * 5:]

            filename = f"temp{next(file_counter)}.wav"
            whisper.audio.save_audio(audio_segment, filename, 16000)
            result = model.transcribe(filename, fp16=False)
            print(" You said:", result["text"])

stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
stream.start()

threading.Thread(target=transcribe_stream, daemon=True).start()

while True:
    pass
