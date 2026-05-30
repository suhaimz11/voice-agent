"""
Whisper speech-to-text wrapper.

Loads a local faster-whisper model and
transcribes recorded audio into text.

Uses CTranslate2 backend — no PyTorch required.
"""

import os
import time

from faster_whisper import WhisperModel

from utils.logger import log


class WhisperSTT:

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Load faster-whisper model during startup.

        Available models:
            tiny.en    -> fastest, English-only (~150 MB)
            base.en    -> balanced, English-only (~290 MB)
            small.en   -> better accuracy (~490 MB)
            medium.en  -> slower, more accurate (~1.5 GB)

        compute_type options:
            int8       -> recommended for CPU and ARM (Raspberry Pi)
            float16    -> recommended if CUDA GPU is available
            float32    -> fallback for older hardware
        """

        log(
            f"Loading faster-whisper model '{model_size}' "
            f"[device={device}, compute_type={compute_type}]..."
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        log(
            f"faster-whisper '{model_size}' loaded ✓"
        )

    # ---------------------------------------------------------
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
    ) -> str:
        """
        Convert audio file into text.
        """

        try:

            started_at = time.time()

            segments, info = self.model.transcribe(
                audio_path,

                # Set to None for auto language detection
                language=language,

                # Suppress blank/silence outputs
                vad_filter=True,
            )

            # Segments are a lazy generator — consume fully
            text = " ".join(
                segment.text for segment in segments
            ).strip()

            log(
                (
                    "Transcription complete "
                    f"in {time.time() - started_at:.1f}s: '{text}'"
                )
            )

            # Cleanup temp audio file
            try:
                os.remove(audio_path)

            except Exception:
                pass

            return text

        except Exception as e:

            log(
                f"Whisper transcription error: {e}",
                level="error",
            )

            return ""