import wave
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

print("Loading model...")
voice = PiperVoice.load("models/en_GB-jenny_dioco-medium.onnx")
print("Model loaded")

path = "test_output.wav"
with wave.open(path, "wb") as wf:
    voice.synthesize("Hello, I am Jenny, your voice assistant.", wf)

print("WAV file written")

# Read and play
with wave.open(path, "rb") as wf:
    sample_rate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

print(f"Playing audio — sample_rate={sample_rate}, samples={len(audio)}")
sd.play(audio, sample_rate)
sd.wait()
print("Done")