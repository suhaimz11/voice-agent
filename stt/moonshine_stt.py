"""Moonshine Voice speech-to-text adapter for recorded WAV utterances."""

from __future__ import annotations

import os

from utils.logger import elapsed_seconds, format_duration, log, log_timing, monotonic_seconds


class MoonshineSTT:
    """Transcribe complete WAV utterances with a local Moonshine model."""

    def __init__(
        self,
        language: str = "en",
        model_arch: int | None = None,
        model_path: str | None = None,
    ):
        try:
            from moonshine_voice import Transcriber, get_model_for_language, load_wav_file
        except ImportError as exc:
            raise RuntimeError(
                "Moonshine Voice is not installed. Run: pip install moonshine-voice"
            ) from exc

        self.language = language
        self._load_wav_file = load_wav_file
        started_at = monotonic_seconds()

        if model_path:
            if model_arch is None:
                raise ValueError(
                    "VOICE_AGENT_MOONSHINE_MODEL_ARCH is required when "
                    "VOICE_AGENT_MOONSHINE_MODEL_PATH is set."
                )
            resolved_path, resolved_arch = model_path, model_arch
        else:
            resolved_path, resolved_arch = get_model_for_language(language, model_arch)

        self.model_path = resolved_path
        self.model_arch = resolved_arch
        self.model = Transcriber(model_path=resolved_path, model_arch=resolved_arch)

        log_timing(
            "Moonshine model load",
            started_at,
            details=(
                f"language={language}, model_arch={resolved_arch}, "
                f"model_path={resolved_path}"
            ),
        )

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Convert a recorded WAV file into plain text."""
        started_at = monotonic_seconds()

        try:
            audio_data, sample_rate = self._load_wav_file(audio_path)
            inference_started_at = monotonic_seconds()
            transcript = self.model.transcribe_without_streaming(
                audio_data, sample_rate=sample_rate, flags=0
            )
            inference_duration = elapsed_seconds(inference_started_at)

            lines = list(transcript.lines)
            text = " ".join(
                line.text.strip()
                for line in lines
                if getattr(line, "text", "").strip()
            ).strip()

            log_timing(
                "Transcription",
                started_at,
                details=(
                    "backend=moonshine, "
                    f"language={language or self.language}, sample_rate={sample_rate}, "
                    f"lines={len(lines)}, chars={len(text)}, "
                    f"inference={format_duration(inference_duration)}"
                ),
            )
            log(f"Transcription text: '{text}'")
            return text

        except Exception as exc:
            log_timing(
                "Transcription failed",
                started_at,
                level="error",
                details=f"backend=moonshine, error={exc}",
            )
            log(f"Moonshine transcription error: {exc}", level="error")
            return ""

        finally:
            try:
                os.remove(audio_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                log(f"Could not remove temporary audio '{audio_path}': {exc}", level="warning")
