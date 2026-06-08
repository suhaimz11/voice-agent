"""
Handles microphone recording for the voice agent.

Records audio until silence is detected,
then saves the result as a temporary WAV file.

Uses sounddevice for audio I/O (replaces PyAudio).
VAD runs via openWakeWord which uses ONNX Runtime
internally — no PyTorch required.
"""

import tempfile
import wave

import numpy as np
import sounddevice as sd
from openwakeword.vad import VAD

from utils.logger import (
    elapsed_seconds,
    format_duration,
    log,
    log_timing,
    monotonic_seconds,
)


# int16 = 2 bytes per sample — fixed, no PyAudio needed to look this up
SAMPLE_WIDTH = 2


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
        input_device: int | None = None,
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

        self.input_device = input_device

        # openWakeWord VAD uses Silero via ONNX Runtime internally
        self.vad = VAD()

        log(
            (
                "AudioRecorder initialized "
                f"(start_timeout={self.no_speech_timeout}s, "
                f"end_silence={self.silence_duration}s, "
                f"vad_threshold={self.vad_threshold}, "
                f"input_device={self.input_device})"
            )
        )

    # ---------------------------------------------------------
    def record_until_silence(
        self,
        no_speech_timeout=None,
    ) -> str | None:
        """
        Start recording from the default microphone
        and stop once silence is detected.
        """

        frames = []

        speaking = False

        first_speech_at = None

        silent_chunks = 0

        max_vad_score = 0.0

        max_rms = 0.0

        vad_calls = 0

        vad_elapsed = 0.0

        stop_reason = "max_record_seconds"

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

        started_at = monotonic_seconds()

        self.vad.reset_states()

        log(
            (
                "Recording started "
                f"(start_timeout={effective_no_speech_timeout:.1f}s)"
            )
        )

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype="int16",
            device=self.input_device,
        ) as stream:

            for chunk_index in range(max_chunks):

                data, _ = stream.read(self.chunk_size)

                frames.append(bytes(data))

                # Convert raw bytes into numpy array for VAD
                arr = np.frombuffer(
                    data,
                    dtype=np.int16,
                ).astype(np.float32)

                # RMS kept for debug logging only
                rms = float(
                    np.sqrt(np.mean(arr ** 2))
                )

                vad_started_at = monotonic_seconds()

                vad_score = float(
                    self.vad.predict(
                        arr.astype(np.int16),
                        frame_size=480,
                    )
                )

                vad_elapsed += elapsed_seconds(vad_started_at)

                vad_calls += 1

                max_rms = max(max_rms, rms)

                max_vad_score = max(max_vad_score, vad_score)

                # Voice activity detected
                if vad_score >= self.vad_threshold:

                    if not speaking:

                        first_speech_at = elapsed_seconds(started_at)

                        log(
                            (
                                "Speech detected "
                                f"after {format_duration(first_speech_at)} "
                                f"(vad={vad_score:.2f}, rms={rms:.0f})"
                            )
                        )

                    speaking = True

                    silent_chunks = 0

                else:

                    if (
                        not speaking
                        and chunk_index >= no_speech_chunks_needed
                    ):
                        stop_reason = "no_speech_timeout"

                        log(
                            (
                                "No speech detected "
                                f"after {format_duration(elapsed_seconds(started_at))}"
                            ),
                            level="debug",
                        )
                        break

                    # Count silence only after speech has begun
                    if speaking:

                        silent_chunks += 1

                        if silent_chunks >= silence_chunks_needed:
                            stop_reason = "silence_detected"
                            break

        captured_seconds = len(frames) / chunks_per_second

        avg_vad_duration = (
            vad_elapsed / vad_calls
            if vad_calls
            else 0.0
        )

        # Ignore empty or noise-only recordings
        if not speaking or len(frames) < 5:

            log(
                (
                    "Recording discarded "
                    f"(max_vad={max_vad_score:.2f}, "
                    f"max_rms={max_rms:.0f})"
                ),
                level="debug",
            )

            log_timing(
                "Recording",
                started_at,
                details=(
                    f"result=discarded, reason={stop_reason}, "
                    f"captured={format_duration(captured_seconds)}, "
                    f"chunks={len(frames)}, "
                    f"max_vad={max_vad_score:.2f}, "
                    f"max_rms={max_rms:.0f}, "
                    f"avg_vad={format_duration(avg_vad_duration)}"
                ),
            )

            return None

        save_started_at = monotonic_seconds()

        audio_path = self._save_wav(frames)

        save_duration = elapsed_seconds(save_started_at)

        log_timing(
            "Recording",
            started_at,
            details=(
                f"result=saved, reason={stop_reason}, "
                f"captured={format_duration(captured_seconds)}, "
                f"speech_start={format_duration(first_speech_at or 0.0)}, "
                f"chunks={len(frames)}, "
                f"max_vad={max_vad_score:.2f}, "
                f"max_rms={max_rms:.0f}, "
                f"avg_vad={format_duration(avg_vad_duration)}, "
                f"save={format_duration(save_duration)}, "
                f"path={audio_path}"
            ),
        )

        return audio_path

    # ---------------------------------------------------------
    def _save_wav(self, frames: list) -> str:
        """
        Save recorded audio to a temporary WAV file.
        """

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        )

        path = tmp.name

        tmp.close()

        with wave.open(path, "wb") as wf:

            wf.setnchannels(self.channels)

            # int16 is always 2 bytes — no PyAudio needed
            wf.setsampwidth(SAMPLE_WIDTH)

            wf.setframerate(self.sample_rate)

            wf.writeframes(b"".join(frames))

        return path
