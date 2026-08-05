"""Runtime profiles for desktop development and Raspberry Pi deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    moonshine_model_arch: int | None
    ollama_model: str
    ollama_keep_alive: str
    speech_start_timeout: float
    end_silence_seconds: float
    minimum_speech_seconds: float
    maximum_utterance_seconds: float
    pre_speech_seconds: float
    response_cooldown_seconds: float


PROFILES = {
    "desktop": RuntimeProfile(
        name="desktop",
        moonshine_model_arch=None,
        ollama_model="qwen3:1.7b",
        ollama_keep_alive="10m",
        speech_start_timeout=5.0,
        end_silence_seconds=0.55,
        minimum_speech_seconds=0.3,
        maximum_utterance_seconds=30.0,
        pre_speech_seconds=0.3,
        response_cooldown_seconds=0.15,
    ),
    "pi4": RuntimeProfile(
        name="pi4",
        moonshine_model_arch=2,
        ollama_model="qwen3:0.6b",
        ollama_keep_alive="-1",
        speech_start_timeout=5.0,
        end_silence_seconds=0.55,
        minimum_speech_seconds=0.3,
        maximum_utterance_seconds=15.0,
        pre_speech_seconds=0.3,
        response_cooldown_seconds=0.15,
    ),
}


def load_runtime_profile() -> RuntimeProfile:
    name = os.environ.get("VOICE_AGENT_PROFILE", "desktop").strip().lower()
    if name not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown VOICE_AGENT_PROFILE '{name}'. Choose: {choices}.")
    return PROFILES[name]
