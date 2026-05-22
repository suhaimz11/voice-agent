"""
Math query handler for the voice agent.

Converts spoken math into valid expressions
and evaluates them safely.
"""

import math
import re

from utils.logger import log


# Basic spoken-word replacements
WORD_MAP = {

    # Operators
    r"\bplus\b": "+",
    r"\bminus\b": "-",
    r"\btimes\b": "*",
    r"\bmultiplied by\b": "*",
    r"\bdivided by\b": "/",
    r"\bover\b": "/",

    # Powers
    r"\bto the power of\b": "**",
    r"\bto the power\b": "**",
    r"\bpow\b": "**",
    r"\bsquared\b": "**2",
    r"\bcubed\b": "**3",

    # Modulo
    r"\bmod\b": "%",
    r"\bmodulo\b": "%",
    r"\bmodulus\b": "%",

    # Number words
    r"\bzero\b": "0",
    r"\bone\b": "1",
    r"\btwo\b": "2",
    r"\bthree\b": "3",
    r"\bfour\b": "4",
    r"\bfive\b": "5",
    r"\bsix\b": "6",
    r"\bseven\b": "7",
    r"\beight\b": "8",
    r"\bnine\b": "9",
    r"\bten\b": "10",

    # Cleanup
    r"\band\b": "",
}


# Common spoken patterns
SQRT_PATTERN = re.compile(
    r"square\s*root\s*of\s*([\d.]+)",
    re.I
)

PERCENT_PATTERN = re.compile(
    r"([\d.]+)\s*percent\s*of\s*([\d.]+)",
    re.I
)

FACTORIAL_PATTERN = re.compile(
    r"factorial\s*(?:of\s*)?([\d]+)",
    re.I
)


def handle_math(text: str) -> str | None:
    """
    Try to evaluate a spoken math query.
    Returns a formatted result or None.
    """

    t = text.lower().strip()

    # Remove common conversational prefixes
    prefixes = [
        "what is",
        "what's",
        "calculate",
        "compute",
        "evaluate",
        "how much is",
        "tell me",
        "give me",
    ]

    for prefix in prefixes:
        t = t.removeprefix(prefix).strip()

    # ----------------------------------------
    # Special cases
    # ----------------------------------------

    # Square root
    match = SQRT_PATTERN.search(t)

    if match:
        value = float(match.group(1))

        result = math.sqrt(value)

        return _format_result(result)

    # Percent calculation
    match = PERCENT_PATTERN.search(t)

    if match:

        percent = float(match.group(1))
        total = float(match.group(2))

        result = (percent / 100) * total

        return _format_result(result)

    # Factorial
    match = FACTORIAL_PATTERN.search(t)

    if match:

        value = int(match.group(1))

        # Prevent absurdly large values
        if value > 20:
            return "That factorial is too large."

        return str(math.factorial(value))

    # ----------------------------------------
    # Convert spoken words to operators
    # ----------------------------------------

    for pattern, replacement in WORD_MAP.items():

        t = re.sub(
            pattern,
            replacement,
            t,
            flags=re.I
        )

    # Keep only valid math characters
    t = re.sub(
        r"[^0-9+\-*/().%\s]",
        "",
        t
    ).strip()

    # Remove remaining spaces
    t = re.sub(r"\s+", "", t)

    if not t:
        return None

    # ----------------------------------------
    # Evaluate expression
    # ----------------------------------------

    try:

        result = _safe_eval(t)

        return _format_result(result)

    except Exception as e:

        log(
            f"Math eval failed for '{t}': {e}",
            level="debug"
        )

        return None


# ---------------------------------------------------------
def _safe_eval(expr: str) -> float:
    """
    Evaluate a restricted math expression safely.
    """

    allowed = re.compile(
        r"^[0-9+\-*/().%\s\*]+$"
    )

    if not allowed.match(expr):
        raise ValueError(f"Unsafe expression: {expr}")

    code = compile(expr, "<string>", "eval")

    return eval(
        code,
        {"__builtins__": {}},
        {}
    )


# ---------------------------------------------------------
def _format_result(value):
    """
    Clean up float formatting for speech output.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)