import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.memory_store import MemoryStore
from agent.processor import AgentProcessor
from agent.tool_registry import ToolRegistry


class ProcessorToolTests(unittest.TestCase):

    def _processor(self):
        self.executed = None
        registry = ToolRegistry()
        registry.register(
            "set_timer",
            "Set a timer.",
            {"seconds": "integer"},
            self._record_tool_call,
        )

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        return AgentProcessor(
            memory_store=MemoryStore(Path(temp_dir.name) / "memory.json"),
            tool_registry=registry,
        )

    def _record_tool_call(self, arguments):
        self.executed = arguments
        return "Timer set for 5 seconds."

    def test_executes_llm_tool_call_json(self):
        processor = self._processor()
        llm_reply = (
            '{"tool_call": {"name": "set_timer", '
            '"arguments": {"seconds": 5, "label": "tea"}}}'
        )

        with patch("agent.processor.ask_llm", return_value=llm_reply):
            response = processor.process("set a tea timer for five seconds")

        self.assertEqual(response, "Timer set for 5 seconds.")
        self.assertEqual(self.executed, {"seconds": 5, "label": "tea"})

    def test_plain_llm_response_still_passes_through(self):
        processor = self._processor()

        with patch("agent.processor.ask_llm", return_value="Sure thing."):
            response = processor.process("tell me something nice")

        self.assertEqual(response, "Sure thing.")
        self.assertIsNone(self.executed)

    def test_wellbeing_question_is_answered_locally(self):
        processor = self._processor()

        with patch("agent.processor.ask_llm") as ask_llm:
            response = processor.process("How are you?")

        self.assertEqual(response, "I'm doing well, thanks. How can I help?")
        ask_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
