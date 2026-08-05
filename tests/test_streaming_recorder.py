import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault(
    "sounddevice",
    types.SimpleNamespace(RawInputStream=None),
)
sys.modules.setdefault("openwakeword", types.SimpleNamespace())
sys.modules.setdefault(
    "openwakeword.vad",
    types.SimpleNamespace(VAD=object),
)

from audio.recorder import AudioRecorder


class _FakeInputStream:
    def __init__(self, chunks, **kwargs):
        self.chunks = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, chunk_size):
        return next(self.chunks), False


class _FakeVad:
    def __init__(self, scores):
        self.scores = iter(scores)

    def reset_states(self):
        pass

    def predict(self, samples, frame_size):
        return next(self.scores)


class StreamingRecorderTests(unittest.TestCase):
    def _recorder(self, scores):
        recorder = AudioRecorder.__new__(AudioRecorder)
        recorder.sample_rate = 100
        recorder.chunk_size = 10
        recorder.channels = 1
        recorder.vad_threshold = 0.5
        recorder.silence_duration = 0.2
        recorder.no_speech_timeout = 0.5
        recorder.max_record_seconds = 1.0
        recorder.minimum_speech_seconds = 0.2
        recorder.pre_speech_seconds = 0.2
        recorder.input_device = None
        recorder.vad = _FakeVad(scores)
        return recorder

    def test_streams_pre_roll_speech_and_endpoint_audio(self):
        scores = [0.0, 0.0, 0.8, 0.9, 0.0, 0.0]
        chunks = [
            np.full(10, index * 100, dtype=np.int16).tobytes()
            for index in range(len(scores))
        ]
        received = []
        recorder = self._recorder(scores)

        with patch(
            "audio.recorder.sd.RawInputStream",
            side_effect=lambda **kwargs: _FakeInputStream(chunks, **kwargs),
        ):
            result = recorder.stream_until_silence(
                lambda audio, rate: received.append((audio, rate))
            )

        self.assertTrue(result.speech_detected)
        self.assertEqual(result.stop_reason, "silence_detected")
        self.assertEqual(len(received), 5)
        self.assertTrue(all(rate == 100 for _, rate in received))

    def test_returns_no_speech_without_sending_audio(self):
        scores = [0.0] * 5
        chunks = [np.zeros(10, dtype=np.int16).tobytes() for _ in scores]
        received = []
        recorder = self._recorder(scores)

        with patch(
            "audio.recorder.sd.RawInputStream",
            side_effect=lambda **kwargs: _FakeInputStream(chunks, **kwargs),
        ):
            result = recorder.stream_until_silence(
                lambda audio, rate: received.append((audio, rate))
            )

        self.assertFalse(result.speech_detected)
        self.assertEqual(result.stop_reason, "no_speech_timeout")
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
