"""
Handles microphone recording for the voice agent.

Records audio until silence is detected,
then saves the result as a temporary WAV file.
"""

import time
import tempfile
import wave

import numpy as np
import pyaudio

from utils.logger import log


class AudioRecorder:

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        silence_threshold: float = 500,
        silence_duration: float = 1.5,
        no_speech_timeout: float = 1.0,
        max_record_seconds: float = 30.0,
    ):

        # Whisper performs best at 16kHz mono audio
        self.sample_rate = sample_rate

        self.chunk_size = chunk_size

        self.channels = channels

        # Minimum RMS level considered as speech
        self.silence_threshold = silence_threshold

        # Stop recording after this duration of silence
        self.silence_duration = silence_duration

        # Return None if speech has not started within this duration
        self.no_speech_timeout = no_speech_timeout

        # Safety cap to avoid endless recording
        self.max_record_seconds = max_record_seconds

        self.format = pyaudio.paInt16

        log("AudioRecorder initialized")

    # ---------------------------------------------------------
    def record_until_silence(self) -> str | None:
        """
        Start recording from the default microphone
        and stop once silence is detected.
        """

        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames = []

        speaking = False

        silent_chunks = 0

        chunks_per_second = (
            self.sample_rate / self.chunk_size
        )

        max_chunks = int(
            self.max_record_seconds * chunks_per_second
        )

        silence_chunks_needed = int(
            self.silence_duration * chunks_per_second
        )

        no_speech_chunks_needed = int(
            self.no_speech_timeout * chunks_per_second
        )

        started_at = time.time()

        log("Recording started")

        for chunk_index in range(max_chunks):

            data = stream.read(
                self.chunk_size,
                exception_on_overflow=False,
            )

            frames.append(data)

            # Convert raw audio bytes into numpy array
            arr = np.frombuffer(
                data,
                dtype=np.int16
            ).astype(np.float32)

            # Root mean square volume
            rms = float(
                np.sqrt(np.mean(arr ** 2))
            )

            # Voice activity detected
            if rms > self.silence_threshold:

                speaking = True

                silent_chunks = 0

            else:

                if (
                    not speaking
                    and chunk_index >= no_speech_chunks_needed
                ):
                    log(
                        (
                            "No speech detected "
                            f"after {time.time() - started_at:.1f}s"
                        ),
                        level="debug"
                    )
                    break

                # Start counting silence only
                # after speech has begun
                if speaking:

                    silent_chunks += 1

                    if silent_chunks >= silence_chunks_needed:
                        break

        # Release microphone
        stream.stop_stream()

        stream.close()

        pa.terminate()

        # Ignore empty recordings
        if not speaking or len(frames) < 5:
            return None

        log(
            f"Recording stopped after {time.time() - started_at:.1f}s"
        )

        return self._save_wav(frames)

    # ---------------------------------------------------------
    def _save_wav(self, frames: list) -> str:
        """
        Save recorded audio to a temporary WAV file.
        """

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        path = tmp.name

        tmp.close()

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
