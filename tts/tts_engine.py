"""Moonshine Voice text-to-speech adapter."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import sounddevice as sd

from utils.logger import elapsed_seconds, log, log_timing, monotonic_seconds


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TTSEngine:
    """Speak responses using Moonshine Voice only."""

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0,
        barge_in_enabled: bool | None = None,
        barge_in_threshold: float = 1200,
        barge_in_grace_seconds: float = 0.4,
        barge_in_duration: float = 0.15,
        input_device: int | None = None,
        output_device: int | None = None,
    ):
        del voice_index  # Kept in the signature for compatibility.

        try:
            from moonshine_voice import TextToSpeech
        except ImportError as exc:
            raise RuntimeError(
                "Moonshine Voice is not installed. Run: pip install moonshine-voice"
            ) from exc

        self.rate = rate
        self.volume = volume
        self.language = os.environ.get("VOICE_AGENT_TTS_LANGUAGE", "en_us")
        self.voice = os.environ.get(
            "VOICE_AGENT_TTS_VOICE",
            "kokoro_af_heart",
        ) or None
        self.input_device = input_device
        self.output_device = output_device
        self.barge_in_enabled = (
            _env_bool("VOICE_AGENT_BARGE_IN", False)
            if barge_in_enabled is None
            else barge_in_enabled
        )
        self.barge_in_threshold = barge_in_threshold
        self.barge_in_grace_seconds = barge_in_grace_seconds
        self.barge_in_duration = barge_in_duration
        self.barge_in_sample_rate = 16000
        self.barge_in_chunk_size = 1024

        started_at = monotonic_seconds()
        self._tts = TextToSpeech(
            self.language,
            voice=self.voice,
            output_device=self.output_device,
            volume=self.volume,
            download=True,
        )
        log_timing(
            "Moonshine TTS load",
            started_at,
            details=(
                f"language={self.language}, voice={self.voice or 'default'}, "
                f"output_device={self.output_device}"
            ),
        )
        log(
            "TTSEngine initialized "
            f"(backend=moonshine, input_device={self.input_device}, "
            f"output_device={self.output_device}, "
            f"barge_in_enabled={self.barge_in_enabled})"
        )

    @property
    def speed(self) -> float:
        """Map the existing 120-230 speech-rate range to Moonshine speed."""
        return round(self.rate / 175.0, 2)

    def adjust_rate(
        self,
        delta: int,
        min_rate: int = 120,
        max_rate: int = 230,
    ) -> int:
        self.rate = max(min_rate, min(max_rate, self.rate + delta))
        log(f"Moonshine TTS rate set to {self.rate} (speed={self.speed})")
        return self.rate

    def speak(self, text: str, interruptible: bool = False) -> bool:
        """Speak text and return False only when genuine barge-in stops it."""
        if not text:
            return True

        try:
            if interruptible and self.barge_in_enabled:
                return self._speak_interruptible(text)
            return self._speak_blocking(text)
        except Exception as exc:
            log(f"Moonshine TTS error: {exc}", level="error", exc_info=True)
            log(f"TTS fallback text: {text}")
            return True

    def list_voices(self):
        from moonshine_voice import list_tts_voices

        voices = list_tts_voices(self.language)
        log(f"Moonshine TTS voices for {self.language}: {voices}")
        return voices

    def _speak_blocking(self, text: str) -> bool:
        started_at = monotonic_seconds()
        log(f"TTS started: {text}")
        self._tts.say(text, speed=self.speed)
        self._tts.wait()
        log_timing(
            "TTS total",
            started_at,
            details=(
                f"backend=moonshine, mode=blocking, text_chars={len(text)}, "
                f"speed={self.speed}"
            ),
        )
        return True

    def _speak_interruptible(self, text: str) -> bool:
        started_at = monotonic_seconds()
        chunk_seconds = self.barge_in_chunk_size / self.barge_in_sample_rate
        chunks_needed = max(1, math.ceil(self.barge_in_duration / chunk_seconds))
        voice_chunks = 0
        max_rms = 0.0
        interrupted = False

        log(f"TTS started (interruptible): {text}")
        self._tts.say(text, speed=self.speed)

        try:
            with sd.RawInputStream(
                samplerate=self.barge_in_sample_rate,
                blocksize=self.barge_in_chunk_size,
                channels=1,
                dtype="int16",
                device=self.input_device,
            ) as mic:
                while self._tts.is_talking():
                    if elapsed_seconds(started_at) < self.barge_in_grace_seconds:
                        time.sleep(0.02)
                        continue

                    data, _ = mic.read(self.barge_in_chunk_size)
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    max_rms = max(max_rms, rms)
                    voice_chunks = (
                        voice_chunks + 1
                        if rms >= self.barge_in_threshold
                        else max(0, voice_chunks - 1)
                    )

                    if voice_chunks >= chunks_needed:
                        self._tts.stop()
                        interrupted = True
                        log(
                            "Moonshine TTS interrupted by barge-in "
                            f"(rms={rms:.0f}, threshold={self.barge_in_threshold:.0f})"
                        )
                        break
        finally:
            if not interrupted:
                self._tts.wait()

        log_timing(
            "TTS total",
            started_at,
            details=(
                "backend=moonshine, mode=interruptible, "
                f"result={'interrupted' if interrupted else 'completed'}, "
                f"text_chars={len(text)}, max_barge_in_rms={max_rms:.0f}"
            ),
        )
        return not interrupted
