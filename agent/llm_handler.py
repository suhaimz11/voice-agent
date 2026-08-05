"""
Local LLM handler.

Supports two backends, selected via LLM_BACKEND env var:

    LLM_BACKEND=ollama      (default) uses Ollama + Qwen3 1.7B
    LLM_BACKEND=llamacpp    uses llama-cpp-python with a local GGUF model

Handles:
- conversational responses
- short-term chat memory
- multi-turn interactions
"""

import os

from config import load_runtime_profile
from utils.logger import (
    elapsed_seconds,
    format_duration,
    log,
    log_timing,
    monotonic_seconds,
)


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------

SYSTEM_PROMPT = """
You are a local AI voice assistant.

You can remember information shared during the current conversation.
You may also receive persistent local memory about the user.

If the user tells you their name,
remember it and use it naturally later.

Keep responses:
- very short
- conversational
- natural for speech
- one sentence unless the user asks for detail
- under 25 words whenever possible

Avoid long paragraphs.
"""

MAX_HISTORY = 8

# -----------------------------------------------------------
# Backend: Ollama
# -----------------------------------------------------------

class OllamaBackend:

    def __init__(self):

        from ollama import chat as ollama_chat

        profile = load_runtime_profile()
        self._chat = ollama_chat
        self.model = os.environ.get("OLLAMA_MODEL", profile.ollama_model)
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "1024"))
        self.num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "60"))
        self.temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.6"))
        self.keep_alive = os.environ.get(
            "OLLAMA_KEEP_ALIVE",
            profile.ollama_keep_alive,
        )

        log(
            (
                f"LLM backend: Ollama (model={self.model}, "
                f"num_ctx={self.num_ctx}, num_predict={self.num_predict}, "
                f"keep_alive={self.keep_alive})"
            )
        )

    def generate(self, messages: list) -> str:

        response = self._chat(
            model=self.model,
            messages=messages,
            think=False,
            keep_alive=self.keep_alive,
            options={
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        )

        return response["message"]["content"]


# -----------------------------------------------------------
# Backend: llama-cpp-python
# -----------------------------------------------------------

class LlamaCppBackend:

    def __init__(self):

        from llama_cpp import Llama

        model_path = os.environ.get(
            "LLAMACPP_MODEL_PATH",
            "models/model.gguf",
        )

        n_threads = int(
            os.environ.get("LLAMACPP_THREADS", "4")
        )

        n_ctx = int(
            os.environ.get("LLAMACPP_CTX", "2048")
        )

        log(
            f"LLM backend: llama-cpp-python "
            f"(model={model_path}, threads={n_threads}, ctx={n_ctx})"
        )

        self.model = model_path

        self.llm = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,

            # Suppress llama.cpp console output
            verbose=False,
        )

    def generate(self, messages: list) -> str:

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )

        return response["choices"][0]["message"]["content"]


# -----------------------------------------------------------
# Load selected backend lazily
# -----------------------------------------------------------

def _load_backend():

    # Select backend via environment variable
    # export LLM_BACKEND=llamacpp   (Raspberry Pi / low RAM)
    # export LLM_BACKEND=ollama     (default)
    llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()

    if llm_backend == "llamacpp":
        return LlamaCppBackend()

    if llm_backend == "ollama":
        return OllamaBackend()

    log(
        f"Unknown LLM_BACKEND '{llm_backend}', falling back to Ollama.",
        level="warning",
    )

    return OllamaBackend()


_backend = None


def _describe_backend(backend) -> str:
    model = getattr(backend, "model", None)

    if model:
        return f"{backend.__class__.__name__}/{model}"

    return backend.__class__.__name__


def _get_backend():
    global _backend

    if _backend is None:
        started_at = monotonic_seconds()

        _backend = _load_backend()

        log_timing(
            "LLM backend load",
            started_at,
            details=f"backend={_describe_backend(_backend)}",
        )

    return _backend

# Stores ongoing conversation history
conversation_history = []


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

def reset_conversation():
    """
    Clear short-term LLM conversation history.
    """

    conversation_history.clear()

    log("Conversation history cleared.")


def warm_llm() -> bool:
    """Load the configured backend and model before the first user request."""
    started_at = monotonic_seconds()
    try:
        backend = _get_backend()
        backend.generate(
            [
                {"role": "system", "content": "Reply with exactly: ready"},
                {"role": "user", "content": "ready"},
            ]
        )
        log_timing(
            "LLM warm-up",
            started_at,
            details=f"backend={_describe_backend(backend)}, result=ready",
        )
        return True
    except Exception as exc:
        log_timing(
            "LLM warm-up failed",
            started_at,
            level="warning",
            details=str(exc),
        )
        return False


def ask_llm(
    prompt: str,
    memory_context: str | None = None,
    system_context: str | None = None,
) -> str:
    """
    Send user prompt to the active LLM backend
    and return the assistant response.
    """

    started_at = monotonic_seconds()

    try:

        # Build message list
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persistent local memory for this user:\n"
                        f"{memory_context}"
                    ),
                }
            )

        if system_context:
            messages.append(
                {
                    "role": "system",
                    "content": system_context,
                }
            )

        # Add previous conversation turns
        messages.extend(conversation_history)

        # Add latest user message
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        backend_ready_started_at = monotonic_seconds()

        backend = _get_backend()

        backend_ready_duration = elapsed_seconds(backend_ready_started_at)

        generation_started_at = monotonic_seconds()

        # Generate response from active backend
        reply = backend.generate(messages).strip()

        generation_duration = elapsed_seconds(generation_started_at)

        log_timing(
            "LLM generation",
            generation_started_at,
            details=(
                f"backend={_describe_backend(backend)}, "
                f"messages={len(messages)}, "
                f"history_turns={len(conversation_history)}, "
                f"prompt_chars={len(prompt)}, "
                f"response_chars={len(reply)}"
            ),
        )

        # Save this turn to memory
        conversation_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        # Prevent memory from growing forever
        if len(conversation_history) > MAX_HISTORY:

            del conversation_history[
                : len(conversation_history) - MAX_HISTORY
            ]

        log_timing(
            "LLM processing",
            started_at,
            details=(
                f"backend_ready={format_duration(backend_ready_duration)}, "
                f"generation={format_duration(generation_duration)}, "
                f"history_turns_after={len(conversation_history)}"
            ),
        )

        return reply

    except Exception as e:

        log_timing(
            "LLM processing failed",
            started_at,
            level="error",
            details=str(e),
        )

        log(f"LLM error: {e}", level="error")

        return f"LLM error: {e}"
