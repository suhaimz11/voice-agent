"""
tts/tts_engine.py
Offline text-to-speech using pyttsx3
"""

import pyttsx3
from utils.logger import log


class TTSEngine:

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0
    ):

        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index

        log("TTSEngine initialized ✓")

    # ----------------------------------------------------------
    def speak(self, text: str):

        if not text:
            return

        try:
            # Create fresh engine every time
            engine = pyttsx3.init()

            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            voices = engine.getProperty("voices")

            if voices and self.voice_index < len(voices):
                engine.setProperty(
                    "voice",
                    voices[self.voice_index].id
                )

            print(f"🔊 Speaking: {text}")

            engine.say(text)

            engine.runAndWait()

            # Fully stop engine
            engine.stop()

        except Exception as e:

            log(
                f"TTS speak error: {e}",
                level="warning"
            )

            print(f"[TTS] {text}")

    # ----------------------------------------------------------
    def list_voices(self):

        engine = pyttsx3.init()

        voices = engine.getProperty("voices")

        for i, v in enumerate(voices):
            print(f"[{i}] {v.name} — {v.id}")

        engine.stop()