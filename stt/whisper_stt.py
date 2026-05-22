"""
Whisper speech-to-text wrapper.

Loads a local Whisper model and
transcribes recorded audio into text.
"""

import os

import whisper

from utils.logger import log


class WhisperSTT:

    def __init__(self, model_size: str = "base"):
        """
        Load Whisper model during startup.

        Available models:
        tiny   -> fastest
        base   -> balanced
        small  -> better accuracy
        medium -> slower but more accurate
        """

        log(
            f"Loading Whisper model '{model_size}'..."
        )

        self.model = whisper.load_model(model_size)

        log(
            f"Whisper '{model_size}' loaded ✓"
        )

    # ---------------------------------------------------------
    def transcribe(
        self,
        audio_path: str,
        language: str = "en"
    ) -> str:
        """
        Convert audio file into text.
        """

        try:

            result = self.model.transcribe(
                audio_path,

                # Set to None for auto-detection
                language=language,

                # Enable if CUDA GPU is available
                fp16=False,

                verbose=False,
            )

            text = result["text"].strip()

            # Cleanup temp audio file
            try:
                os.remove(audio_path)

            except Exception:
                pass

            return text

        except Exception as e:

            log(
                f"Whisper transcription error: {e}",
                level="error"
            )

            return ""