"""
Math query handler for the voice agent.

Supports:
- natural math phrases
- contextual memory math
- percentages
- factorials
- powers
- safe evaluation
"""

import math
import re

from utils.logger import log


# ---------------------------------------------------------
# Spoken word replacements
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Special phrase patterns
# ---------------------------------------------------------

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

MULTIPLY_PATTERN = re.compile(
    r"(?:multiply|times)\s+([\d.]+)\s+(?:by\s+)?([\d.]+)",
    re.I
)

DIVIDE_PATTERN = re.compile(
    r"(?:divide|divided)\s+([\d.]+)\s+(?:by\s+)?([\d.]+)",
    re.I
)

SUBTRACT_PATTERN = re.compile(
    r"subtract\s+([\d.]+)\s+(?:from|by)\s+([\d.]+)",
    re.I
)

ADD_PATTERN = re.compile(
    r"add\s+([\d.]+)\s+(?:to\s+)?([\d.]+)",
    re.I
)


# ---------------------------------------------------------
def handle_math(text: str) -> str | None:
    """
    Parse and evaluate spoken math.
    """

    t = text.lower().strip()

    # Remove conversational prefixes
    prefixes = [
        "what is",
        "what's",
        "calculate",
        "compute",
        "evaluate",
        "how much is",
        "tell me",
        "give me",
        "can you",
        "please",
    ]

    for prefix in prefixes:
        t = t.removeprefix(prefix).strip()

    # -----------------------------------------------------
    # Special math handlers
    # -----------------------------------------------------

    # Square root
    match = SQRT_PATTERN.search(t)

    if match:

        value = float(match.group(1))

        return _format_result(
            math.sqrt(value)
        )

    # Percentages
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

        if value > 20:
            return "That factorial is too large."

        return str(math.factorial(value))

    # Multiply
    match = MULTIPLY_PATTERN.search(t)

    if match:

        a = float(match.group(1))

        b = float(match.group(2))

        return _format_result(a * b)

    # Divide
    match = DIVIDE_PATTERN.search(t)

    if match:

        a = float(match.group(1))

        b = float(match.group(2))

        if b == 0:
            return "Division by zero is not allowed."

        return _format_result(a / b)

    # Subtract
    match = SUBTRACT_PATTERN.search(t)

    if match:

        a = float(match.group(1))

        b = float(match.group(2))

        # Handle:
        # subtract 2 from 10
        if "from" in t:
            return _format_result(b - a)

        # Handle:
        # subtract 10 by 2
        return _format_result(a - b)

    # Add
    match = ADD_PATTERN.search(t)

    if match:

        a = float(match.group(1))

        b = float(match.group(2))

        return _format_result(a + b)

    # -----------------------------------------------------
    # Word replacement
    # -----------------------------------------------------

    for pattern, replacement in WORD_MAP.items():

        t = re.sub(
            pattern,
            replacement,
            t,
            flags=re.I
        )

    # Keep only safe math characters
    t = re.sub(
        r"[^0-9+\-*/().%\s]",
        "",
        t
    ).strip()

    # Remove spaces
    t = re.sub(
        r"\s+",
        "",
        t
    )

    if not t:
        return None

    # -----------------------------------------------------
    # Safe evaluation
    # -----------------------------------------------------

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
def _safe_eval(expr: str):
    """
    Evaluate restricted expressions safely.
    """

    allowed = re.compile(
        r"^[0-9+\-*/().%\s\*]+$"
    )

    if not allowed.match(expr):

        raise ValueError(
            f"Unsafe expression: {expr}"
        )

    code = compile(
        expr,
        "<string>",
        "eval"
    )

    return eval(
        code,
        {"__builtins__": {}},
        {}
    )


# ---------------------------------------------------------
def _format_result(value):
    """
    Format results for cleaner speech output.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)