"""
Main intent router for the voice agent.

Responsible for:
- intent classification
- request routing
- short-term memory
- response generation
"""

import datetime
import re

from agent.math_handler import handle_math
from agent.llm_handler import ask_llm, reset_conversation
from utils.logger import log


# Intent patterns
# More specific checks should come first

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

SLEEP_TRIGGERS = re.compile(
    r"^(go to sleep|sleep|stop|stop listening|pause listening)\s*$",
    re.I,
)

EXIT_TRIGGERS = re.compile(
    r"^(quit|exit|goodbye|shut down|shutdown|turn off)\s*$",
    re.I,
)

RESET_CONVERSATION_TRIGGERS = re.compile(
    r"^(reset conversation|clear conversation|reset chat|clear chat)\s*$",
    re.I,
)

REPEAT_TRIGGERS = re.compile(
    r"^(repeat that|say that again|repeat|one more time)\s*$",
    re.I,
)

SPEAK_SLOWER_TRIGGERS = re.compile(
    r"^(speak slower|talk slower|slow down)\s*$",
    re.I,
)

SPEAK_FASTER_TRIGGERS = re.compile(
    r"^(speak faster|talk faster|speed up)\s*$",
    re.I,
)

HELP_TRIGGERS = re.compile(
    r"\b(help|what can you do|commands|features|capabilities)\b",
    re.I,
)


class AgentProcessor:

    def __init__(self):

        # Lightweight conversation memory
        self.memory = {
            "last_result": None,
            "last_input": None,
        }

        log("AgentProcessor initialized")

    # ---------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Process user input
        and return a response string.
        """

        text = text.strip()

        intent = self._classify(text)

        log(
            f"Intent: {intent} | Input: '{text}'",
            level="debug"
        )

        # Conversation control commands
        if intent == "SLEEP":
            return "__SLEEP__"

        if intent == "EXIT":
            return "__EXIT__"

        if intent == "RESET_CONVERSATION":
            self.reset_conversation()
            return "__RESET_CONVERSATION__"

        if intent == "REPEAT":
            return "__REPEAT__"

        if intent == "SPEAK_SLOWER":
            return "__SPEAK_SLOWER__"

        if intent == "SPEAK_FASTER":
            return "__SPEAK_FASTER__"

        # Greetings
        if intent == "GREETING":
            return self._handle_greeting()

        # Time / date queries
        if intent == "TIME":
            return self._handle_time()

        # Help command
        if intent == "HELP":
            return self._handle_help()

        # Math handling
        if intent == "MATH":
            return self._handle_math(text)

        return ask_llm(text)

    # ---------------------------------------------------------
    def _classify(self, text: str) -> str:
        """
        Basic rule-based intent classification.
        """

        if SLEEP_TRIGGERS.search(text):
            return "SLEEP"

        if EXIT_TRIGGERS.search(text):
            return "EXIT"

        if RESET_CONVERSATION_TRIGGERS.search(text):
            return "RESET_CONVERSATION"

        if REPEAT_TRIGGERS.search(text):
            return "REPEAT"

        if SPEAK_SLOWER_TRIGGERS.search(text):
            return "SPEAK_SLOWER"

        if SPEAK_FASTER_TRIGGERS.search(text):
            return "SPEAK_FASTER"

        if HELP_TRIGGERS.search(text):
            return "HELP"

        if GREETING_TRIGGERS.search(text):
            return "GREETING"

        if TIME_TRIGGERS.search(text):
            return "TIME"

        if MATH_TRIGGERS.search(text):
            return "MATH"

        # Fallback:
        # treat anything containing digits as math
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

        date_str = now.strftime(
            "%A, %B %d"
        ).replace(" 0", " ")

        return f"It's {time_str} on {date_str}."

    # ---------------------------------------------------------
    def _handle_help(self) -> str:

        return (
            "I can solve math problems, "
            "tell you the time and date, "
            "and handle commands like go to sleep, "
            "reset conversation, repeat that, "
            "speak slower, and speak faster."
        )

    # ---------------------------------------------------------
    def reset_conversation(self):

        reset_conversation()

        self.memory = {
            "last_result": None,
            "last_input": None,
        }

        log("Conversation memory reset")

    # ---------------------------------------------------------
    def _handle_math(self, text: str) -> str:
        """
        Handle math queries with short-term memory.
        """

        # Support contextual references
        # Example:
        # "multiply that by 2"

        if "that" in text.lower():

            last_result = self.memory.get(
                "last_result"
            )

            if last_result is not None:

                text = text.lower().replace(
                    "that",
                    str(last_result)
                )

        result = handle_math(text)

        if result is not None:

            # Save latest interaction
            self.memory["last_result"] = result

            self.memory["last_input"] = text

            return f"The answer is {result}."

        return (
            "I couldn't compute that. "
            "Try something like: "
            "what is 25 plus 17?"
        )

    # ---------------------------------------------------------
    def _handle_unknown(self) -> str:

        return (
            "I didn't understand that. "
            "Try saying help to see available commands."
        )
