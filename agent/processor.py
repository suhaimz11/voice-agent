"""
agent/processor.py
Central router: classifies intent → dispatches to the right handler.

Current intents:
  MATH       → math_handler
  TIME       → time/date queries
  GREETING   → hello/hi/hey
  STOP       → exit commands
  UNKNOWN    → fallback

Designed for easy extension:
  Add crypto/DeFi intents here later:
  SWAP       → crypto_handler.swap(from_token, to_token, amount)
  SEND       → crypto_handler.send(token, amount, address)
  BALANCE    → crypto_handler.balance(token)
  PRICE      → crypto_handler.price(token)
"""

import re
import datetime
from agent.math_handler import handle_math
from utils.logger import log


# ---------------------------------------------------------------------------
# Intent patterns  (order matters — more specific first)
# ---------------------------------------------------------------------------
MATH_TRIGGERS = re.compile(
    r"\b(what is|what's|calculate|compute|evaluate|how much is|"
    r"plus|minus|times|divided|multiply|add|subtract|"
    r"square root|factorial|percent|power|squared|cubed)\b",
    re.I,
)

GREETING_TRIGGERS = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|yo)\b",
    re.I,
)

TIME_TRIGGERS = re.compile(
    r"\b(what time|current time|what('s| is) the time|"
    r"what('s| is) today|what('s| is) the date|what day)\b",
    re.I,
)

STOP_TRIGGERS = re.compile(
    r"^(stop|quit|exit|bye|goodbye|shut down|shutdown|turn off)\b",
    re.I,
)

HELP_TRIGGERS = re.compile(
    r"\b(help|what can you do|commands|features|capabilities)\b",
    re.I,
)


# ---------------------------------------------------------------------------
class AgentProcessor:
    def __init__(self):
        log("AgentProcessor initialized")

    # ------------------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Takes transcribed text, returns a response string.
        """
        t = text.strip()

        intent = self._classify(t)
        log(f"Intent: {intent} | Input: '{t}'", level="debug")

        if intent == "STOP":
            return "__EXIT__"   # main.py watches for this

        if intent == "GREETING":
            return self._handle_greeting(t)

        if intent == "TIME":
            return self._handle_time()

        if intent == "HELP":
            return self._handle_help()

        if intent == "MATH":
            return self._handle_math(t)

        # --- Future intents go here ---
        # if intent == "SWAP":
        #     return crypto_handler.swap(...)
        # if intent == "SEND":
        #     return crypto_handler.send(...)

        return self._handle_unknown(t)

    # ------------------------------------------------------------------
    def _classify(self, text: str) -> str:
        if STOP_TRIGGERS.search(text):    return "STOP"
        if GREETING_TRIGGERS.search(text): return "GREETING"
        if TIME_TRIGGERS.search(text):    return "TIME"
        if HELP_TRIGGERS.search(text):    return "HELP"
        if MATH_TRIGGERS.search(text):    return "MATH"

        # Also try math if text looks like a raw expression
        if re.search(r"\d", text):        return "MATH"

        return "UNKNOWN"

    # ------------------------------------------------------------------
    def _handle_greeting(self, text: str) -> str:
        now = datetime.datetime.now().hour
        if now < 12:
            return "Good morning! Ready to calculate or help you out."
        elif now < 17:
            return "Good afternoon! What can I do for you?"
        else:
            return "Good evening! How can I help?"

    def _handle_time(self) -> str:
        now = datetime.datetime.now()
        time_str = now.strftime("%-I:%M %p")
        date_str = now.strftime("%A, %B %-d")
        return f"It's {time_str} on {date_str}."

    def _handle_help(self) -> str:
        return (
            "I can do math calculations like addition, subtraction, "
            "multiplication, division, square roots, and percentages. "
            "I can also tell you the time and date. "
            "More features like crypto commands are coming soon!"
        )

    def _handle_math(self, text: str) -> str:
        result = handle_math(text)
        if result is not None:
            return f"The answer is {result}."
        return "I couldn't compute that. Try saying something like: what is 25 plus 17?"

    def _handle_unknown(self, text: str) -> str:
        return (
            "I didn't quite catch that. "
            "Try asking me to calculate something, or say 'help' for options."
        )
