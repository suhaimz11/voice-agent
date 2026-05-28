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
from openwakeword.vad import VAD

from utils.logger import log


class AudioRecorder:

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 960,
        channels: int = 1,
        vad_threshold: float = 0.35,
        silence_duration: float = 2.5,
        no_speech_timeout: float = 4.0,
        max_record_seconds: float = 45.0,
    ):

        # Whisper performs best at 16kHz mono audio
        self.sample_rate = sample_rate

        self.chunk_size = chunk_size

        self.channels = channels

        # Minimum Silero VAD score considered as speech
        self.vad_threshold = vad_threshold

        # Stop recording after this duration of silence
        self.silence_duration = silence_duration

        # Return None if speech has not started within this duration
        self.no_speech_timeout = no_speech_timeout

        # Safety cap to avoid endless recording
        self.max_record_seconds = max_record_seconds

        self.format = pyaudio.paInt16

        self.vad = VAD()

        log(
            (
                "AudioRecorder initialized "
                f"(start_timeout={self.no_speech_timeout}s, "
                f"end_silence={self.silence_duration}s, "
                f"vad_threshold={self.vad_threshold})"
            )
        )

    # ---------------------------------------------------------
    def record_until_silence(
        self,
        no_speech_timeout=None
    ) -> str | None:
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

        max_vad_score = 0.0

        max_rms = 0.0

        chunks_per_second = (
            self.sample_rate / self.chunk_size
        )

        max_chunks = int(
            self.max_record_seconds * chunks_per_second
        )

        silence_chunks_needed = int(
            self.silence_duration * chunks_per_second
        )

        effective_no_speech_timeout = (
            self.no_speech_timeout
            if no_speech_timeout is None
            else no_speech_timeout
        )

        no_speech_chunks_needed = int(
            effective_no_speech_timeout * chunks_per_second
        )

        started_at = time.time()

        self.vad.reset_states()

        log(
            (
                "Recording started "
                f"(start_timeout={effective_no_speech_timeout:.1f}s)"
            )
        )

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

            # Root mean square volume is kept for debugging only.
            rms = float(
                np.sqrt(np.mean(arr ** 2))
            )

            vad_score = float(
                self.vad.predict(
                    arr.astype(np.int16),
                    frame_size=480
                )
            )

            max_rms = max(
                max_rms,
                rms
            )

            max_vad_score = max(
                max_vad_score,
                vad_score
            )

            # Voice activity detected
            if vad_score >= self.vad_threshold:

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
            log(
                (
                    "Recording discarded "
                    f"(max_vad={max_vad_score:.2f}, "
                    f"max_rms={max_rms:.0f})"
                ),
                level="debug"
            )
            return None

        log(
            (
                f"Recording stopped after {time.time() - started_at:.1f}s "
                f"(max_vad={max_vad_score:.2f}, max_rms={max_rms:.0f})"
            )
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
