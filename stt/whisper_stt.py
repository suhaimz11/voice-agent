"""
Whisper speech-to-text wrapper.

Loads a local faster-whisper model and
transcribes recorded audio into text.

Uses CTranslate2 backend — no PyTorch required.
"""

import os

from faster_whisper import WhisperModel

from utils.logger import (
    elapsed_seconds,
    format_duration,
    log,
    log_timing,
    monotonic_seconds,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


class WhisperSTT:

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int | None = None,
        best_of: int | None = None,
        vad_filter: bool | None = None,
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

        self.beam_size = int(
            os.environ.get(
                "VOICE_AGENT_STT_BEAM_SIZE",
                str(beam_size if beam_size is not None else 1),
            )
        )

        self.best_of = int(
            os.environ.get(
                "VOICE_AGENT_STT_BEST_OF",
                str(best_of if best_of is not None else 1),
            )
        )

        self.vad_filter = _env_bool(
            "VOICE_AGENT_STT_VAD_FILTER",
            vad_filter if vad_filter is not None else False,
        )

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

        log(
            (
                "STT decode settings "
                f"(beam_size={self.beam_size}, best_of={self.best_of}, "
                f"vad_filter={self.vad_filter})"
            )
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

            started_at = monotonic_seconds()

            transcribe_call_started_at = monotonic_seconds()

            segments, info = self.model.transcribe(
                audio_path,

                # Set to None for auto language detection
                language=language,

                # Recorder already trims silence; disabling this avoids a
                # second VAD pass unless explicitly enabled.
                vad_filter=self.vad_filter,
                beam_size=self.beam_size,
                best_of=self.best_of,
                condition_on_previous_text=False,
            )

            # Segments are a lazy generator — consume fully
            transcribe_call_duration = elapsed_seconds(
                transcribe_call_started_at
            )

            segment_decode_started_at = monotonic_seconds()

            segment_list = list(segments)

            segment_decode_duration = elapsed_seconds(
                segment_decode_started_at
            )

            text = " ".join(
                segment.text for segment in segment_list
            ).strip()

            # Cleanup temp audio file
            cleanup_started_at = monotonic_seconds()

            cleanup_status = "removed"

            try:
                os.remove(audio_path)

            except Exception:
                cleanup_status = "failed"

            cleanup_duration = elapsed_seconds(cleanup_started_at)

            details = [
                f"audio_path={audio_path}",
                f"segments={len(segment_list)}",
                f"chars={len(text)}",
                f"transcribe_call={format_duration(transcribe_call_duration)}",
                f"segment_decode={format_duration(segment_decode_duration)}",
                f"cleanup={cleanup_status}/{format_duration(cleanup_duration)}",
            ]

            detected_language = getattr(info, "language", None)

            if detected_language:
                details.append(f"language={detected_language}")

            language_probability = getattr(info, "language_probability", None)

            if isinstance(language_probability, (int, float)):
                details.append(f"language_probability={language_probability:.2f}")

            audio_duration = getattr(info, "duration", None)

            if isinstance(audio_duration, (int, float)):
                details.append(f"audio={format_duration(audio_duration)}")

            log_timing(
                "Transcription",
                started_at,
                details=", ".join(details),
            )

            log(f"Transcription text: '{text}'")

            return text

        except Exception as e:

            log_timing(
                "Transcription failed",
                started_at,
                level="error",
                details=str(e),
            )

            log(
                f"Whisper transcription error: {e}",
                level="error",
            )

            return ""
