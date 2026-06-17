"""
Main intent router for the voice agent.

Responsible for:
- intent classification
- request routing
- short-term memory
- response generation
"""

import datetime
import json
import re

from agent.math_handler import handle_math
from agent.llm_handler import ask_llm, reset_conversation
from agent.memory_store import MemoryStore
from agent.tool_registry import ToolRegistry, create_default_tool_registry
from utils.logger import log, log_timing, monotonic_seconds


# ---------------------------------------------------------------------------
# Intent patterns
# More specific / anchored patterns come first so they take priority.
# All anchored command patterns drop the ^ anchor and rely on re.match()
# (which implies start-of-string) + a \s*$ tail so trailing whitespace /
# punctuation is handled by the pre-classification strip.
# ---------------------------------------------------------------------------

MATH_TRIGGERS = re.compile(
    r"\b("
    r"calculate|compute|evaluate|"
    r"how much is|plus|minus|times|divided|"
    r"multiply|add|subtract|square root|"
    r"factorial|percent|power|squared|cubed"
    r")\b",
    re.I,
)

GREETING_TRIGGERS = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|yo)\b",
    re.I,
)

TIME_TRIGGERS = re.compile(
    r"\b("
    r"what time|current time|"
    r"what('s| is) the time|"
    r"what('s| is) today|"
    r"what('s| is) the date|"
    r"what day"
    r")\b",
    re.I,
)

# --- Anchored command triggers (use re.match, no ^ needed) ---

SLEEP_TRIGGERS = re.compile(
    r"(go to sleep|sleep|stop|stop listening|pause listening)\s*$",
    re.I,
)

EXIT_TRIGGERS = re.compile(
    r"(quit|exit|goodbye|shut down|shutdown|turn off)\s*$",
    re.I,
)

RESET_CONVERSATION_TRIGGERS = re.compile(
    r"(reset conversation|clear conversation|reset chat|clear chat)\s*$",
    re.I,
)

REPEAT_TRIGGERS = re.compile(
    r"(repeat that|say that again|repeat|one more time)\s*$",
    re.I,
)

SPEAK_SLOWER_TRIGGERS = re.compile(
    r"(speak slower|talk slower|slow down)\s*$",
    re.I,
)

SPEAK_FASTER_TRIGGERS = re.compile(
    r"(speak faster|talk faster|speed up)\s*$",
    re.I,
)

HELP_TRIGGERS = re.compile(
    r"\b(help|what can you do|commands|features|capabilities)\b",
    re.I,
)

MEMORY_QUERY_TRIGGERS = re.compile(
    r"\b(what do you remember|show memory|what do you know about me)\b",
    re.I,
)

NAME_PATTERNS = [
    re.compile(
        r"\b(?:my name is|call me)\s+(?P<value>[\w\s'-]{1,60})\s*$",
        re.I,
    ),
]

FAVORITE_PATTERNS = [
    re.compile(
        r"\bmy favorite\s+(?P<key>[\w\s'-]{1,40})\s+is\s+"
        r"(?P<value>.+?)\s*$",
        re.I,
    ),
]

PREFERENCE_PATTERNS = [
    re.compile(
        r"\bi prefer\s+(?P<value>.+?)\s*$",
        re.I,
    ),
]

FACT_PATTERNS = [
    re.compile(
        r"\bremember(?: that)?\s+(?P<value>.+?)\s*$",
        re.I,
    ),
]

TOOL_RESPONSE_SCHEMA = {
    "tool_call": {
        "name": "set_timer|get_time|get_weather",
        "arguments": {
            "seconds": "integer, required for set_timer",
            "label": "string, optional for set_timer",
            "location": "string, optional for get_weather",
        },
    }
}


def _strip_punctuation(text: str) -> str:
    """
    Remove trailing/leading punctuation that transcribers (e.g. Whisper)
    commonly append to short utterances, e.g. "stop." → "stop".

    Only non-word, non-space characters are removed so that math expressions
    like "2 + 3" are preserved.
    """
    return re.sub(r"[^\w\s]", "", text).strip()


class AgentProcessor:

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
    ):

        # Lightweight conversation memory
        self.memory = {
            "last_result": None,
            "last_input": None,
        }

        self.memory_store = memory_store or MemoryStore()
        self.tool_registry = tool_registry or create_default_tool_registry()

        log("AgentProcessor initialized")

    # ---------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Process user input and return a response string.

        Pipeline
        --------
        1. strip leading/trailing whitespace
        2. strip punctuation for intent classification
           (keeps the original text for downstream handlers that may need it,
            e.g. math expressions)
        3. classify intent on the cleaned text
        4. route to the appropriate handler
        """

        started_at = monotonic_seconds()

        text = text.strip()

        # Clean copy used only for classification so that "stop." → "stop"
        # and "Stop!" → "stop" both match SLEEP_TRIGGERS correctly.
        text_for_classification = _strip_punctuation(text)

        intent = self._classify(text_for_classification)

        log(
            f"Intent: {intent} | Raw: '{text}' | Cleaned: '{text_for_classification}'",
            level="debug",
        )

        def finish(response: str) -> str:
            log_timing(
                "Agent processing",
                started_at,
                details=(
                    f"intent={intent}, "
                    f"used_llm={intent == 'UNKNOWN'}, "
                    f"input_chars={len(text)}, "
                    f"response_chars={len(response)}"
                ),
            )

            return response

        # --- Conversation control commands ---

        if intent == "SLEEP":
            return finish("__SLEEP__")

        if intent == "EXIT":
            return finish("__EXIT__")

        if intent == "RESET_CONVERSATION":
            self.reset_conversation()
            return finish("__RESET_CONVERSATION__")

        if intent == "REPEAT":
            return finish("__REPEAT__")

        if intent == "SPEAK_SLOWER":
            return finish("__SPEAK_SLOWER__")

        if intent == "SPEAK_FASTER":
            return finish("__SPEAK_FASTER__")

        # --- Content intents (use original text for better accuracy) ---

        if intent == "GREETING":
            return finish(self._handle_greeting())

        if intent == "TIME":
            return finish(self._handle_time())

        if intent == "HELP":
            return finish(self._handle_help())

        memory_response = self._handle_memory(text)

        if memory_response:
            return finish(memory_response)

        if intent == "MATH":
            return finish(self._handle_math(text))

        # Fallback to LLM
        return finish(self._handle_llm_or_tool(text))

    # ---------------------------------------------------------
    def _classify(self, text: str) -> str:
        """
        Rule-based intent classification.

        Receives punctuation-stripped text so that transcriber artefacts
        (trailing periods, commas, etc.) don't break anchored patterns.

        Priority order (most specific / least ambiguous first):
          SLEEP > EXIT > RESET_CONVERSATION > REPEAT >
          SPEAK_SLOWER > SPEAK_FASTER > HELP > GREETING > TIME > MATH
        """

        # re.match() checks from the start of the string (replaces the old ^
        # anchor).  The patterns end with \s*$ so only exact short commands
        # trigger these intents — a longer sentence containing "stop" won't
        # accidentally put the agent to sleep.

        if SLEEP_TRIGGERS.match(text):
            return "SLEEP"

        if EXIT_TRIGGERS.match(text):
            return "EXIT"

        if RESET_CONVERSATION_TRIGGERS.match(text):
            return "RESET_CONVERSATION"

        if REPEAT_TRIGGERS.match(text):
            return "REPEAT"

        if SPEAK_SLOWER_TRIGGERS.match(text):
            return "SPEAK_SLOWER"

        if SPEAK_FASTER_TRIGGERS.match(text):
            return "SPEAK_FASTER"

        # HELP before GREETING so "help me" isn't swallowed by the greeting
        # check (neither would match, but keeps the intent order explicit).
        if HELP_TRIGGERS.search(text):
            return "HELP"

        if GREETING_TRIGGERS.match(text):
            return "GREETING"

        if TIME_TRIGGERS.search(text):
            return "TIME"

        if MATH_TRIGGERS.search(text):
            return "MATH"

        # Fallback: bare digits → treat as math
        if re.search(r"\d", text):
            return "MATH"

        return "UNKNOWN"

    # ---------------------------------------------------------
    def _handle_greeting(self) -> str:

        hour = datetime.datetime.now().hour

        if hour < 12:
            return "Good morning. How can I help?"

        if hour < 17:
            return "Good afternoon. What can I do for you?"

        return "Good evening. How can I help?"

    # ---------------------------------------------------------
    def _handle_time(self) -> str:

        now = datetime.datetime.now()

        time_str = now.strftime("%I:%M %p").lstrip("0")

        date_str = now.strftime("%A, %B %d").replace(" 0", " ")

        return f"It's {time_str} on {date_str}."

    # ---------------------------------------------------------
    def _handle_help(self) -> str:

        return (
            "I can solve math problems, "
            "tell you the time and date, "
            "set timers, fetch local weather, "
            "remember your profile, preferences, and facts, "
            "and handle commands like go to sleep, "
            "reset conversation, repeat that, "
            "speak slower, and speak faster."
        )

    # ---------------------------------------------------------
    def _handle_llm_or_tool(self, text: str) -> str:
        reply = ask_llm(
            text,
            memory_context=self.memory_store.to_prompt_context(),
            system_context=self._tool_prompt_context(),
        )

        tool_call = self._parse_tool_call(reply)

        if not tool_call:
            return reply

        name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}

        if not isinstance(name, str) or not isinstance(arguments, dict):
            log(f"Ignoring invalid tool call: {tool_call}", level="warning")
            return "I could not understand that tool request."

        log(f"Executing tool call: {name} args={arguments}", level="debug")

        return self.tool_registry.execute(name, arguments)

    # ---------------------------------------------------------
    def _tool_prompt_context(self) -> str:
        return (
            "You can call local tools when they are needed for real actions.\n"
            "If a tool is needed, respond with JSON only, no prose or markdown.\n"
            "Use this exact shape:\n"
            f"{json.dumps(TOOL_RESPONSE_SCHEMA)}\n"
            "Available tools:\n"
            f"{json.dumps(self.tool_registry.schema())}\n"
            "For ordinary conversation, respond with plain text."
        )

    # ---------------------------------------------------------
    @staticmethod
    def _parse_tool_call(reply: str) -> dict | None:
        raw = reply.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
            raw = re.sub(r"```$", "", raw).strip()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        tool_call = payload.get("tool_call")

        if isinstance(tool_call, dict):
            return tool_call

        return None

    # ---------------------------------------------------------
    def reset_conversation(self):

        reset_conversation()

        self.memory = {
            "last_result": None,
            "last_input": None,
        }

        log("Conversation memory reset")

    # ---------------------------------------------------------
    def _handle_memory(self, text: str) -> str | None:
        """
        Save or summarize durable local memory from natural voice commands.
        """

        memory_text = text.strip().rstrip(".,!?;:")

        if MEMORY_QUERY_TRIGGERS.search(memory_text):
            return self._summarize_memory()

        for pattern in NAME_PATTERNS:
            match = pattern.search(memory_text)
            if match:
                name = self._clean_memory_value(match.group("value"))
                self.memory_store.set_profile("name", name)
                return f"I will remember your name is {name}."

        for pattern in FAVORITE_PATTERNS:
            match = pattern.search(memory_text)
            if match:
                key = f"favorite {match.group('key')}"
                value = self._clean_memory_value(match.group("value"))
                self.memory_store.set_preference(key, value)
                return f"I will remember your {key} is {value}."

        for pattern in PREFERENCE_PATTERNS:
            match = pattern.search(memory_text)
            if match:
                preference = self._clean_memory_value(match.group("value"))
                self.memory_store.set_preference("general", preference)
                return f"I will remember that you prefer {preference}."

        for pattern in FACT_PATTERNS:
            match = pattern.search(memory_text)
            if match:
                fact = self._clean_memory_value(match.group("value"))
                added = self.memory_store.add_fact(fact)

                if added:
                    return "I will remember that."

                return "I already have that in memory."

        return None

    # ---------------------------------------------------------
    def _summarize_memory(self) -> str:
        profile = self.memory_store.get_profile()
        preferences = self.memory_store.get_preferences()
        facts = self.memory_store.get_facts()

        if not profile and not preferences and not facts:
            return "I do not have any saved memory yet."

        parts = []

        name = profile.get("name")
        if name:
            parts.append(f"your name is {name}")

        if preferences:
            preference_items = [
                f"{key.replace('_', ' ')} is {value}"
                for key, value in sorted(preferences.items())
            ]
            parts.append("your preferences: " + "; ".join(preference_items))

        recent_facts = [
            fact.get("text", "")
            for fact in facts[-3:]
            if fact.get("text")
        ]
        if recent_facts:
            parts.append("recent facts: " + "; ".join(recent_facts))

        return "I remember " + ". ".join(parts) + "."

    # ---------------------------------------------------------
    @staticmethod
    def _clean_memory_value(value: str) -> str:
        return value.strip().strip(".,!?;:")

    # ---------------------------------------------------------
    def _handle_math(self, text: str) -> str:
        """
        Handle math queries with short-term memory.

        Supports contextual references, e.g. "multiply that by 2"
        where "that" refers to the last computed result.
        """

        if "that" in text.lower():

            last_result = self.memory.get("last_result")

            if last_result is not None:
                text = text.lower().replace("that", str(last_result))

        result = handle_math(text)

        if result is not None:

            self.memory["last_result"] = result
            self.memory["last_input"] = text

            return f"The answer is {result}."

        return (
            "I couldn't compute that. "
            "Try something like: what is 25 plus 17?"
        )

    # ---------------------------------------------------------
    def _handle_unknown(self) -> str:

        return (
            "I didn't understand that. "
            "Try saying help to see available commands."
        )
