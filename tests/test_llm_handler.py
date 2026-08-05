import os
import sys
import types
import unittest
from unittest.mock import patch

from agent.llm_handler import OllamaBackend


class OllamaBackendTests(unittest.TestCase):
    def test_pi_defaults_use_qwen_with_small_context(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"message": {"content": "Short answer."}}

        fake_ollama = types.SimpleNamespace(chat=fake_chat)
        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {
                "OLLAMA_MODEL",
                "OLLAMA_NUM_CTX",
                "OLLAMA_NUM_PREDICT",
            }
        }

        with patch.dict(os.environ, clean_environment, clear=True), patch.dict(
            sys.modules, {"ollama": fake_ollama}
        ):
            backend = OllamaBackend()
            reply = backend.generate([{"role": "user", "content": "Hello"}])

        self.assertEqual(reply, "Short answer.")
        self.assertEqual(backend.model, "qwen3:1.7b")
        self.assertEqual(calls[0]["options"]["num_ctx"], 1024)
        self.assertEqual(calls[0]["options"]["num_predict"], 60)
        self.assertFalse(calls[0]["think"])

    def test_pi_profile_uses_small_qwen_and_persistent_keep_alive(self):
        fake_ollama = types.SimpleNamespace(
            chat=lambda **kwargs: {"message": {"content": "Ready."}}
        )
        environment = {
            "VOICE_AGENT_PROFILE": "pi4",
        }

        with patch.dict(os.environ, environment, clear=True), patch.dict(
            sys.modules, {"ollama": fake_ollama}
        ):
            backend = OllamaBackend()

        self.assertEqual(backend.model, "qwen3:0.6b")
        self.assertEqual(backend.keep_alive, "-1")


if __name__ == "__main__":
    unittest.main()
