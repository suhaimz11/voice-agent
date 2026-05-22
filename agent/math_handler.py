"""
agent/math_handler.py
Handles spoken math expressions safely.
Converts spoken words → expression → evaluates → spoken result.

Examples handled:
  "what is 25 plus 17"            → 42
  "calculate 100 divided by 4"    → 25.0
  "what's the square root of 144" → 12.0
  "8 to the power of 3"           → 512
  "15 percent of 200"             → 30.0
"""

import re
import math
from utils.logger import log


# Words → symbols
WORD_MAP = {
    # Operators
    r"\bplus\b":              "+",
    r"\bminus\b":             "-",
    r"\btimes\b":             "*",
    r"\bmultiplied by\b":     "*",
    r"\bdivided by\b":        "/",
    r"\bover\b":              "/",
    r"\bto the power of\b":   "**",
    r"\bto the power\b":      "**",
    r"\bpow\b":               "**",
    r"\bsquared\b":           "**2",
    r"\bcubed\b":             "**3",
    r"\bmod\b":               "%",
    r"\bmodulo\b":            "%",
    r"\bmodulus\b":           "%",

    # Number words (basic)
    r"\bzero\b":  "0",
    r"\bone\b":   "1",
    r"\btwo\b":   "2",
    r"\bthree\b": "3",
    r"\bfour\b":  "4",
    r"\bfive\b":  "5",
    r"\bsix\b":   "6",
    r"\bseven\b": "7",
    r"\beight\b": "8",
    r"\bnine\b":  "9",
    r"\bten\b":   "10",

    # Misc
    r"\band\b":   "",  # "one hundred and twenty" → strip "and"
}

# Special function patterns (handled before general eval)
SQRT_PATTERN    = re.compile(r"square\s*root\s*of\s*([\d.]+)", re.I)
PERCENT_PATTERN = re.compile(r"([\d.]+)\s*percent\s*of\s*([\d.]+)", re.I)
FACTORIAL_PATTERN = re.compile(r"factorial\s*(?:of\s*)?([\d]+)", re.I)


def handle_math(text: str) -> str | None:
    """
    Try to extract and evaluate a math expression from the text.
    Returns a human-readable answer string, or None if not a math query.
    """
    t = text.lower().strip()

    # Strip common preamble
    for prefix in ["what is", "what's", "calculate", "compute", "evaluate",
                   "how much is", "tell me", "give me"]:
        t = t.removeprefix(prefix).strip()

    # --- Special cases first ---

    # Square root
    m = SQRT_PATTERN.search(t)
    if m:
        n = float(m.group(1))
        result = math.sqrt(n)
        return _format_result(result)

    # Percent of
    m = PERCENT_PATTERN.search(t)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = (pct / 100) * total
        return _format_result(result)

    # Factorial
    m = FACTORIAL_PATTERN.search(t)
    if m:
        n = int(m.group(1))
        if n > 20:
            return "That factorial is astronomically large."
        return str(math.factorial(n))

    # --- Word substitution ---
    for pattern, replacement in WORD_MAP.items():
        t = re.sub(pattern, replacement, t, flags=re.I)

    # Remove leftover words, keep: digits, operators, parens, decimal point, spaces
    t = re.sub(r"[^0-9+\-*/().**%.\s]", "", t).strip()
    t = re.sub(r"\s+", "", t)  # remove spaces

    if not t:
        return None

    # --- Safe eval ---
    try:
        result = _safe_eval(t)
        return _format_result(result)
    except Exception as e:
        log(f"Math eval failed for '{t}': {e}", level="debug")
        return None


# ------------------------------------------------------------------
def _safe_eval(expr: str) -> float:
    """
    Evaluate only math expressions. Raises on anything non-numeric.
    Uses compile() to restrict to expressions, then eval with empty builtins.
    """
    # Extra safety: only allow safe characters
    allowed = re.compile(r"^[0-9+\-*/().%\s\*]+$")
    if not allowed.match(expr):
        raise ValueError(f"Unsafe expression: {expr}")

    code = compile(expr, "<string>", "eval")

    # Whitelist allowed names (none needed for pure math)
    result = eval(code, {"__builtins__": {}}, {})
    return result


def _format_result(value) -> str:
    """Return int if whole number, else float with up to 6 sig digits."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
