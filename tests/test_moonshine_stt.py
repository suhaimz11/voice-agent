import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from stt.moonshine_stt import MoonshineSTT


class _Line:
    def __init__(self, text):
        self.text = text


class _Transcript:
    lines = [_Line(" hello "), _Line("world")]


class _Transcriber:
    def __init__(self, model_path, model_arch):
        self.model_path = model_path
        self.model_arch = model_arch

    def transcribe_without_streaming(self, audio_data, sample_rate, flags):
        return _Transcript()


class MoonshineSTTTests(unittest.TestCase):
    def test_transcribes_wav_and_removes_temporary_file(self):
        fake_module = types.SimpleNamespace(
            Transcriber=_Transcriber,
            get_model_for_language=lambda language, arch: ("cached/model", arch or 1),
            load_wav_file=lambda path: ([0.0, 0.1], 16000),
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            audio_path = Path(wav_file.name)

        with patch.dict(sys.modules, {"moonshine_voice": fake_module}):
            stt = MoonshineSTT(language="en")
            text = stt.transcribe(str(audio_path))

        self.assertEqual(text, "hello world")
        self.assertFalse(audio_path.exists())


if __name__ == "__main__":
    unittest.main()
