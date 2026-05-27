"""
Wake word detection using openWakeWord.
"""

import queue
import time as time_module

import numpy as np
import sounddevice as sd

from openwakeword.model import Model
from utils.logger import log


class WakeWordDetector:

    def __init__(
        self,
        wake_word: str = "hey jarvis",
        threshold: float = 0.5,
        patience: int = 1,
        ignore_initial_seconds: float = 0.2,
    ):

        self.wake_word = self._normalize_wake_word(wake_word)

        self.model = Model(
            wakeword_models=[self.wake_word],
            inference_framework="onnx"
        )

        self.sample_rate = 16000

        self.chunk_size = 1280

        self.audio_queue = queue.Queue(maxsize=25)

        self.threshold = threshold

        self.patience = patience

        self.ignore_initial_seconds = ignore_initial_seconds

    # -----------------------------------------------------
    @staticmethod
    def _normalize_wake_word(wake_word: str) -> str:
        return wake_word.strip().lower().replace(" ", "_")

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
