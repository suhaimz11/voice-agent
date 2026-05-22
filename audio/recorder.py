"""
audio/recorder.py
Records from microphone using PyAudio.
Stops automatically when silence is detected.
"""

import pyaudio
import wave
import tempfile
import numpy as np

from utils.logger import log


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        silence_threshold: float = 500,
        silence_duration: float = 1.5,
        max_record_seconds: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_record_seconds = max_record_seconds
        self.format = pyaudio.paInt16

        log("AudioRecorder initialized")

    # ----------------------------------------------------------
    def record_until_silence(self) -> str | None:

        # CREATE PyAudio HERE
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames = []
        silent_chunks = 0
        speaking = False

        chunks_per_second = self.sample_rate / self.chunk_size
        max_chunks = int(self.max_record_seconds * chunks_per_second)

        silence_chunks_needed = int(
            self.silence_duration * chunks_per_second
        )

        print("🎙️ Speak now...")

        for _ in range(max_chunks):

            data = stream.read(
                self.chunk_size,
                exception_on_overflow=False
            )

            frames.append(data)

            # RMS volume
            arr = np.frombuffer(
                data,
                dtype=np.int16
            ).astype(np.float32)

            rms = float(np.sqrt(np.mean(arr ** 2)))

            if rms > self.silence_threshold:
                speaking = True
                silent_chunks = 0

            else:
                if speaking:
                    silent_chunks += 1

                    if silent_chunks >= silence_chunks_needed:
                        break

        # VERY IMPORTANT
        stream.stop_stream()
        stream.close()

        # TERMINATE PyAudio COMPLETELY
        pa.terminate()

        if not speaking or len(frames) < 5:
            return None

        return self._save_wav(frames)

    # ----------------------------------------------------------
    def _save_wav(self, frames: list) -> str:

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        path = tmp.name
        tmp.close()

        # CREATE TEMP PyAudio ONLY FOR SAMPLE WIDTH
        pa = pyaudio.PyAudio()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)

            wf.setsampwidth(
                pa.get_sample_size(self.format)
            )

            wf.setframerate(self.sample_rate)

            wf.writeframes(b"".join(frames))

        pa.terminate()

        return path