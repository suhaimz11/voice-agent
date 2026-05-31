"""
Offline text-to-speech engine.

Primary  : Piper TTS with Jenny voice (neural, natural female voice)
Fallback : pyttsx3 (system voice, used if Piper is not installed)

Uses sounddevice for all audio playback and barge-in monitoring.
No PyAudio required.

Piper model setup:
    1. pip install piper-tts
    2. Download model files into models/:
       https://github.com/rhasspy/piper/releases
       Recommended: en_US-jenny_dioco-medium.onnx
                    en_US-jenny_dioco-medium.onnx.json
    3. Set PIPER_MODEL_PATH env var if using a different path/voice.
"""

import io
import math
import os
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from utils.logger import log


# --- Optional Piper import ---
try:
    from piper.voice import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False


# Default model path — override with PIPER_MODEL_PATH env var
PIPER_MODEL_PATH = os.environ.get(
    "PIPER_MODEL_PATH",
    "models/en_GB-jenny_dioco-medium.onnx",
)


class TTSEngine:

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0,
        barge_in_threshold: float = 1200,
        barge_in_grace_seconds: float = 0.4,
        barge_in_duration: float = 0.15,
        input_device: int | None = None,
        output_device: int | None = None,
    ):

        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index

        self.barge_in_threshold = barge_in_threshold
        self.barge_in_grace_seconds = barge_in_grace_seconds
        self.barge_in_duration = barge_in_duration
        self.barge_in_sample_rate = 16000
        self.barge_in_chunk_size = 1024
        self.input_device = input_device
        self.output_device = output_device

        # Piper speed control — mapped from pyttsx3-style rate
        self._length_scale = self._rate_to_length_scale(rate)

        # Decide which backend to use
        self._use_piper = (
            _PIPER_AVAILABLE
            and os.path.exists(PIPER_MODEL_PATH)
        )

        if self._use_piper:

            log(f"Loading Piper TTS model: {PIPER_MODEL_PATH}")

            self._piper_voice = PiperVoice.load(PIPER_MODEL_PATH)

            log("Piper TTS loaded ✓ (Jenny — en_US-jenny_dioco-medium)")

        else:

            if _PIPER_AVAILABLE and not os.path.exists(PIPER_MODEL_PATH):
                log(
                    f"Piper model not found at '{PIPER_MODEL_PATH}'. "
                    "Falling back to pyttsx3. "
                    "Download: https://github.com/rhasspy/piper/releases",
                    level="warning",
                )

            elif not _PIPER_AVAILABLE:
                log(
                    "piper-tts not installed — using pyttsx3 fallback.",
                    level="warning",
                )

            import pyttsx3 as _pyttsx3
            self._pyttsx3 = _pyttsx3

        log(
            (
                "TTSEngine initialized "
                f"(input_device={self.input_device}, "
                f"output_device={self.output_device})"
            )
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _rate_to_length_scale(rate: int) -> float:
        """
        Convert pyttsx3-style rate (120–230) to Piper length_scale.

        length_scale 1.0 = normal speed
        length_scale > 1 = slower
        length_scale < 1 = faster

        rate 175 (default) -> 1.00
        rate 120 (slow)    -> 1.46
        rate 230 (fast)    -> 0.76
        """

        return round(175 / max(rate, 1), 2)

    def adjust_rate(
        self,
        delta: int,
        min_rate: int = 120,
        max_rate: int = 230,
    ) -> int:
        """
        Adjust speech rate and return the new value.
        """

        self.rate = max(
            min_rate,
            min(max_rate, self.rate + delta)
        )

        self._length_scale = self._rate_to_length_scale(self.rate)

        log(
            f"TTS rate set to {self.rate} "
            f"(length_scale={self._length_scale})"
        )

        return self.rate

    # ---------------------------------------------------------
    # Public
    # ---------------------------------------------------------

    def speak(
        self,
        text: str,
        interruptible: bool = False,
    ) -> bool:
        """
        Convert text into speech output.

        Returns True when speech finishes normally,
        False when interrupted by barge-in.
        """

        if not text:
            return True

        try:

            if interruptible:
                return self._speak_interruptible(text)

            return self._speak_blocking(text)

        except Exception as e:

            log(
                f"TTS error: {e}",
                level="warning",
                exc_info=True,
            )

            log(f"TTS fallback text: {text}")

            return True

    def list_voices(self):
        """
        Print available voices.
        Piper: shows current model.
        pyttsx3: lists all system voices.
        """

        if self._use_piper:
            log(
                "Piper TTS active — "
                f"model: {PIPER_MODEL_PATH}"
            )
            return

        engine = self._pyttsx3.init()

        voices = engine.getProperty("voices")

        for index, voice in enumerate(voices):
            log(f"[{index}] {voice.name} — {voice.id}")

        engine.stop()

    # ---------------------------------------------------------
    # Piper synthesis helper
    # ---------------------------------------------------------

    def _synthesize(self, text: str):
        """
        Synthesize text with Piper entirely in memory.
        Returns (audio_array, sample_rate) — no temp files.
        """

        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            self._piper_voice.synthesize_wav(text, wf)

        buf.seek(0)

        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        audio = (
            np.frombuffer(raw, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        return audio, sample_rate

    # ---------------------------------------------------------
    # Blocking speech
    # ---------------------------------------------------------

    def _speak_blocking(self, text: str) -> bool:

        started_at = time.time()

        log(f"TTS started: {text}")

        if self._use_piper:

            audio, sample_rate = self._synthesize(text)

            sd.play(
                audio,
                sample_rate,
                device=self.output_device,
            )
            sd.wait()

        else:

            engine = self._configure_pyttsx3()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        log(f"TTS finished in {time.time() - started_at:.1f}s")

        return True

    # ---------------------------------------------------------
    # Interruptible speech
    # ---------------------------------------------------------

    def _speak_interruptible(self, text: str) -> bool:

        started_at = time.time()

        log(f"TTS started (interruptible): {text}")

        if self._use_piper:
            return self._piper_interruptible(text, started_at)

        return self._pyttsx3_interruptible(text, started_at)

    # ---------------------------------------------------------

    def _piper_interruptible(
        self,
        text: str,
        started_at: float,
    ) -> bool:

        chunk_seconds = (
            self.barge_in_chunk_size / self.barge_in_sample_rate
        )

        chunks_needed = max(
            1,
            math.ceil(self.barge_in_duration / chunk_seconds)
        )

        voice_chunks = 0
        interrupted = False
        max_rms = 0.0

        audio, sample_rate = self._synthesize(text)

        try:

            sd.play(
                audio,
                sample_rate,
                device=self.output_device,
            )

            with sd.RawInputStream(
                samplerate=self.barge_in_sample_rate,
                blocksize=self.barge_in_chunk_size,
                channels=1,
                dtype="int16",
                device=self.input_device,
            ) as mic:

                while True:

                    # Check if playback has finished
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:
                        break

                    if (
                        time.time() - started_at
                        >= self.barge_in_grace_seconds
                    ):

                        data, _ = mic.read(self.barge_in_chunk_size)

                        arr = np.frombuffer(
                            data,
                            dtype=np.int16,
                        ).astype(np.float32)

                        rms = float(np.sqrt(np.mean(arr ** 2)))

                        max_rms = max(max_rms, rms)

                        if rms >= self.barge_in_threshold:
                            voice_chunks += 1
                        else:
                            voice_chunks = max(0, voice_chunks - 1)

                        if voice_chunks >= chunks_needed:

                            sd.stop()

                            interrupted = True

                            log(
                                (
                                    "TTS interrupted by barge-in "
                                    f"(rms={rms:.0f}, "
                                    f"threshold={self.barge_in_threshold:.0f})"
                                )
                            )

                            break

                    else:
                        time.sleep(0.02)

        finally:

            sd.stop()

        if interrupted:
            return False

        log(
            (
                f"TTS finished in {time.time() - started_at:.1f}s "
                f"(max_barge_in_rms={max_rms:.0f})"
            )
        )

        return True

    # ---------------------------------------------------------

    def _pyttsx3_interruptible(
        self,
        text: str,
        started_at: float,
    ) -> bool:

        chunk_seconds = (
            self.barge_in_chunk_size / self.barge_in_sample_rate
        )

        chunks_needed = max(
            1,
            math.ceil(self.barge_in_duration / chunk_seconds)
        )

        voice_chunks = 0
        interrupted = False
        max_rms = 0.0

        speech_done = threading.Event()
        speech_ready = threading.Event()
        speech_error = []
        engine_holder = {"engine": None}

        def run_speech():
            try:
                engine = self._configure_pyttsx3()
                engine_holder["engine"] = engine
                speech_ready.set()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                speech_error.append(e)
            finally:
                speech_ready.set()
                speech_done.set()

        speech_thread = threading.Thread(
            target=run_speech,
            daemon=True,
        )

        speech_thread.start()
        speech_ready.wait(timeout=2.0)

        try:

            with sd.RawInputStream(
                samplerate=self.barge_in_sample_rate,
                blocksize=self.barge_in_chunk_size,
                channels=1,
                dtype="int16",
                device=self.input_device,
            ) as mic:

                while not speech_done.is_set():

                    if (
                        time.time() - started_at
                        >= self.barge_in_grace_seconds
                    ):

                        data, _ = mic.read(self.barge_in_chunk_size)

                        arr = np.frombuffer(
                            data,
                            dtype=np.int16,
                        ).astype(np.float32)

                        rms = float(np.sqrt(np.mean(arr ** 2)))

                        max_rms = max(max_rms, rms)

                        if rms >= self.barge_in_threshold:
                            voice_chunks += 1
                        else:
                            voice_chunks = max(0, voice_chunks - 1)

                        if voice_chunks >= chunks_needed:

                            interrupted = True

                            engine = engine_holder["engine"]

                            if engine is not None:
                                engine.stop()

                            speech_done.wait(timeout=1.0)

                            log(
                                (
                                    "TTS interrupted by barge-in "
                                    f"(rms={rms:.0f}, "
                                    f"threshold={self.barge_in_threshold:.0f})"
                                )
                            )

                            break

                    else:
                        time.sleep(0.02)

        finally:

            engine = engine_holder["engine"]

            if engine is not None:
                engine.stop()

            speech_thread.join(timeout=1.0)

        if speech_error:
            raise speech_error[0]

        if interrupted:
            return False

        log(
            (
                f"TTS finished in {time.time() - started_at:.1f}s "
                f"(max_barge_in_rms={max_rms:.0f}, "
                f"threshold={self.barge_in_threshold:.0f})"
            )
        )

        return True

    # ---------------------------------------------------------
    # pyttsx3 engine config (fallback only)
    # ---------------------------------------------------------

    def _configure_pyttsx3(self):

        engine = self._pyttsx3.init()

        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        voices = engine.getProperty("voices")

        if voices and self.voice_index < len(voices):
            engine.setProperty(
                "voice",
                voices[self.voice_index].id,
            )

        return engine
