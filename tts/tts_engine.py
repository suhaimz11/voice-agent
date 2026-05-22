"""
Offline text-to-speech engine.

Uses pyttsx3 for local speech synthesis
without requiring any external API.
"""

import pyttsx3

from utils.logger import log


class TTSEngine:

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0,
    ):

        # Speech settings
        self.rate = rate

        self.volume = volume

        self.voice_index = voice_index

        log("TTSEngine initialized")

    # ---------------------------------------------------------
    def speak(self, text: str):
        """
        Convert text into speech output.
        """

        if not text:
            return

        try:

            # Create a fresh engine instance
            # to avoid Windows audio lock issues
            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                self.rate
            )

            engine.setProperty(
                "volume",
                self.volume
            )

            voices = engine.getProperty("voices")

            # Select configured system voice
            if voices and self.voice_index < len(voices):

                engine.setProperty(
                    "voice",
                    voices[self.voice_index].id
                )

            print(f"🔊 Speaking: {text}")

            engine.say(text)

            engine.runAndWait()

            engine.stop()

        except Exception as e:

            log(
                f"TTS error: {e}",
                level="warning"
            )

            # Fallback if audio output fails
            print(f"[TTS] {text}")

    # ---------------------------------------------------------
    def list_voices(self):
        """
        Print available system voices.
        """

        engine = pyttsx3.init()

        voices = engine.getProperty("voices")

        for index, voice in enumerate(voices):

            print(
                f"[{index}] "
                f"{voice.name} — {voice.id}"
            )

        engine.stop()