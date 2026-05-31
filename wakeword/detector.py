"""
Wake word detection using openWakeWord.
"""

import queue
import time as time_module
from pathlib import Path

import numpy as np
import sounddevice as sd

from openwakeword.model import Model
from utils.logger import log


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MODEL_DIR = PROJECT_ROOT / "models"

BUILT_IN_WAKE_WORDS = {
    "alexa",
    "hey_mycroft",
    "hey_rhasspy",
    "timer",
    "weather",
}


class WakeWordDetector:

    def __init__(
        self,
        wake_word: str = "alexa",
        model_path: str | None = None,
        fallback_wake_word: str = "alexa",
        threshold: float = 0.5,
        patience: int = 1,
        ignore_initial_seconds: float = 0.2,
    ):

        requested_wake_word = self._normalize_wake_word(wake_word)

        resolved_model, active_wake_word = self._resolve_model(
            requested_wake_word=requested_wake_word,
            model_path=model_path,
            fallback_wake_word=fallback_wake_word,
        )

        self.wake_word = active_wake_word
        self.requested_wake_word = requested_wake_word

        self.model = Model(
            wakeword_models=[resolved_model],
            inference_framework="onnx"
        )

        self.sample_rate = 16000

        self.chunk_size = 1280

        self.audio_queue = queue.Queue(maxsize=25)

        self.threshold = threshold

        self.patience = patience

        self.ignore_initial_seconds = ignore_initial_seconds

        log(
            (
                f"Wake word armed: requested='{self.requested_wake_word}', "
                f"active='{self.wake_word}'"
            )
        )

    # -----------------------------------------------------
    @staticmethod
    def _normalize_wake_word(wake_word: str) -> str:
        return wake_word.strip().lower().replace(" ", "_")

    # -----------------------------------------------------
    @classmethod
    def _resolve_model(
        cls,
        requested_wake_word: str,
        model_path: str | None,
        fallback_wake_word: str,
    ) -> tuple[str, str]:

        if model_path:
            path = Path(model_path)

            if path.exists():
                return str(path), path.stem

            log(
                (
                    f"Wake model not found at '{model_path}'. "
                    "Checking built-in and default model paths."
                ),
                level="warning",
            )

        candidate = CUSTOM_MODEL_DIR / f"{requested_wake_word}.onnx"

        if candidate.exists():
            return str(candidate), candidate.stem

        if requested_wake_word in BUILT_IN_WAKE_WORDS:
            return requested_wake_word, requested_wake_word

        fallback = cls._normalize_wake_word(fallback_wake_word)

        log(
            (
                f"Wake word '{requested_wake_word}' needs a custom model at "
                f"'{CUSTOM_MODEL_DIR / (requested_wake_word + '.onnx')}'. "
                f"Using '{fallback}' for now."
            ),
            level="warning",
        )

        return fallback, fallback

    # -----------------------------------------------------
    def _audio_callback(
        self,
        indata,
        frames,
        time,
        status
    ):

        if status:
            log(f"Wake audio stream status: {status}", level="warning")

        try:
            self.audio_queue.put_nowait(
                indata.copy()
            )
        except queue.Full:
            # Keep the detector on live audio instead of old buffered frames.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass

            self.audio_queue.put_nowait(
                indata.copy()
            )

    # -----------------------------------------------------
    def reset(self):
        self.model.reset()
        self._clear_audio_queue()

    # -----------------------------------------------------
    def _clear_audio_queue(self):
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    # -----------------------------------------------------
    def listen(self):

        log("Wake listener armed")

        self.reset()

        detected = False

        started_at = time_module.time()

        stream = sd.InputStream(

            samplerate=self.sample_rate,

            channels=1,

            dtype="int16",

            blocksize=self.chunk_size,

            callback=self._audio_callback

        )

        stream.start()

        try:

            while not detected:

                audio = self.audio_queue.get()

                if (
                    time_module.time() - started_at
                    < self.ignore_initial_seconds
                ):
                    continue

                audio = np.frombuffer(
                    audio,
                    dtype=np.int16
                )

                prediction = self.model.predict(
                    audio,
                    patience={self.wake_word: self.patience},
                    threshold={self.wake_word: self.threshold},
                )

                score = prediction.get(
                    self.wake_word,
                    0.0
                )

                if score >= self.threshold:

                    log(
                        (
                            f"Wake word detected: {self.wake_word} "
                            f"(score={score:.3f})"
                        )
                    )

                    self.reset()

                    detected = True

        finally:

            # Stop microphone stream completely
            stream.stop()

            stream.close()

        return True
