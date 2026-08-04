import sys
import types
import unittest
from unittest.mock import Mock

sys.modules.setdefault("sounddevice", types.SimpleNamespace())

from tts.tts_engine import TTSEngine


class TTSEngineFlowTests(unittest.TestCase):
    def _engine(self, barge_in_enabled):
        engine = TTSEngine.__new__(TTSEngine)
        engine.barge_in_enabled = barge_in_enabled
        engine._speak_blocking = Mock(return_value=True)
        engine._speak_interruptible = Mock(return_value=False)
        return engine

    def test_interruptible_request_is_blocking_when_barge_in_is_disabled(self):
        engine = self._engine(barge_in_enabled=False)

        completed = engine.speak("Complete this response.", interruptible=True)

        self.assertTrue(completed)
        engine._speak_blocking.assert_called_once_with("Complete this response.")
        engine._speak_interruptible.assert_not_called()

    def test_barge_in_can_be_enabled_explicitly(self):
        engine = self._engine(barge_in_enabled=True)

        completed = engine.speak("Interruptible response.", interruptible=True)

        self.assertFalse(completed)
        engine._speak_interruptible.assert_called_once_with("Interruptible response.")
        engine._speak_blocking.assert_not_called()


if __name__ == "__main__":
    unittest.main()
