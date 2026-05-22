"""
stt/whisper_stt.py
Wraps OpenAI Whisper for local speech-to-text transcription.
Model sizes:  tiny(~40MB) | base(~140MB) | small(~460MB) | medium(~1.5GB)
"""

import os
import whisper
from utils.logger import log


class WhisperSTT:
    def __init__(self, model_size: str = "base"):
        """
        Load Whisper model. First run downloads the weights automatically.
        Recommended: 'base' for speed, 'small' for better accuracy.
        """
        log(f"Loading Whisper model '{model_size}'... (first run downloads weights)")
        self.model = whisper.load_model(model_size)
        log(f"Whisper '{model_size}' loaded ✓")

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe WAV file → string.
        language: ISO code, e.g. 'en', 'de', 'fr'. None = auto-detect.
        """
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,          # set True if you have a CUDA GPU
                verbose=False,
            )
            text = result["text"].strip()

            # Clean up temp file
            try:
                os.remove(audio_path)
            except Exception:
                pass

            return text

        except Exception as e:
            log(f"Whisper transcription error: {e}", level="error")
            return ""
