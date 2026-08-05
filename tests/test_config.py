import os
import unittest
from unittest.mock import patch

from config import load_runtime_profile


class RuntimeProfileTests(unittest.TestCase):
    def test_pi4_profile_uses_tiny_streaming_and_fast_endpoint(self):
        with patch.dict(os.environ, {"VOICE_AGENT_PROFILE": "pi4"}):
            profile = load_runtime_profile()

        self.assertEqual(profile.moonshine_model_arch, 2)
        self.assertEqual(profile.ollama_model, "qwen3:0.6b")
        self.assertEqual(profile.end_silence_seconds, 0.8)
        self.assertEqual(profile.maximum_utterance_seconds, 15.0)

    def test_unknown_profile_is_rejected(self):
        with patch.dict(os.environ, {"VOICE_AGENT_PROFILE": "unknown"}):
            with self.assertRaises(ValueError):
                load_runtime_profile()


if __name__ == "__main__":
    unittest.main()
