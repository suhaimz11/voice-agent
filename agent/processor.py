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
from utils.logger import log


# Intent patterns
# More specific checks should come first

MATH_TRIGGERS = re.compile(
    r"\b("
    r"what is|what's|calculate|compute|evaluate|"
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

STOP_TRIGGERS = re.compile(
    r"^(stop|quit|exit|goodbye|shut down|shutdown|turn off)\s*$",
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

        # Exit command
        if intent == "STOP":
            return "__EXIT__"

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

        return self._handle_unknown()

    # ---------------------------------------------------------
    def _classify(self, text: str) -> str:
        """
        Basic rule-based intent classification.
        """

        if STOP_TRIGGERS.search(text):
            return "STOP"

        if GREETING_TRIGGERS.search(text):
            return "GREETING"

        if TIME_TRIGGERS.search(text):
            return "TIME"

        if HELP_TRIGGERS.search(text):
            return "HELP"

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
            "and handle basic voice commands."
        )

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