"""
Wake word detection using openWakeWord.
"""

import queue

import numpy as np
import sounddevice as sd

from openwakeword.model import Model


class WakeWordDetector:

    def __init__(
        self,
        wake_word: str = "hey jarvis"
    ):

        self.model = Model(
            inference_framework="onnx"
        )

        self.sample_rate = 16000

        self.chunk_size = 1280

        self.audio_queue = queue.Queue()

        self.wake_word = wake_word.lower()

    # -----------------------------------------------------
    def _audio_callback(
        self,
        indata,
        frames,
        time,
        status
    ):

        if status:
            print(status)

        self.audio_queue.put(
            indata.copy()
        )

    # -----------------------------------------------------
    def listen(self):

        print("👂 Waiting for wake word...")

        detected = False

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

                audio = np.frombuffer(
                    audio,
                    dtype=np.int16
                )

                prediction = self.model.predict(audio)

                for name, score in prediction.items():

                    if score > 0.5:

                        print(
                            f"🟢 Wake word detected: {name}"
                        )

                        # Clear old audio frames
                        with self.audio_queue.mutex:
                            self.audio_queue.queue.clear()

                        detected = True

                        break

        finally:

            # Stop microphone stream completely
            stream.stop()

            stream.close()

        return True