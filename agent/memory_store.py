"""
Persistent local memory store for user-specific assistant context.

Stores profile data, preferences, and conversation facts in a small JSON file.
The store is intentionally simple so it remains easy to inspect and back up.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import log


DEFAULT_MEMORY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "memory.json"
)

MEMORY_PATH_ENV = "VOICE_AGENT_MEMORY_PATH"

DEFAULT_MEMORY = {
    "profile": {},
    "preferences": {},
    "facts": [],
}


class MemoryStore:
    """
    JSON-backed local memory for durable assistant context.

    Data shape:
    {
      "profile": {"name": "Sam"},
      "preferences": {"favorite_color": "green"},
      "facts": [{"text": "User is learning Python.", "created_at": "..."}]
    }
    """

    def __init__(self, path: str | Path | None = None, autosave: bool = True):
        configured_path = path or os.environ.get(MEMORY_PATH_ENV)

        self.path = Path(configured_path) if configured_path else DEFAULT_MEMORY_PATH
        self.autosave = autosave
        self.data = deepcopy(DEFAULT_MEMORY)

        self.load()

    def load(self) -> dict[str, Any]:
        """
        Load memory from disk. Missing files start with an empty memory record.
        """

        if not self.path.exists():
            self.data = deepcopy(DEFAULT_MEMORY)
            return self.data

        try:
            with self.path.open("r", encoding="utf-8") as memory_file:
                loaded = json.load(memory_file)
        except (OSError, json.JSONDecodeError) as exc:
            log(
                f"Could not load memory store at {self.path}: {exc}",
                level="warning",
            )
            self.data = deepcopy(DEFAULT_MEMORY)
            return self.data

        self.data = self._normalize(loaded)

        return self.data

    def save(self) -> None:
        """
        Save memory to disk with an atomic replace to avoid partial writes.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as memory_file:
                json.dump(
                    self.data,
                    memory_file,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                memory_file.write("\n")

            Path(temp_path).replace(self.path)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def get_profile(self) -> dict[str, Any]:
        return dict(self.data["profile"])

    def get_preferences(self) -> dict[str, Any]:
        return dict(self.data["preferences"])

    def get_facts(self) -> list[dict[str, str]]:
        return list(self.data["facts"])

    def set_profile(self, key: str, value: Any) -> None:
        self.data["profile"][self._normalize_key(key)] = value
        self._save_if_needed()

    def set_preference(self, key: str, value: Any) -> None:
        self.data["preferences"][self._normalize_key(key)] = value
        self._save_if_needed()

    def add_fact(self, text: str) -> bool:
        """
        Add a conversation fact. Returns False when the fact is empty or a duplicate.
        """

        fact_text = text.strip()

        if not fact_text:
            return False

        normalized_text = fact_text.casefold()
        existing_facts = {
            fact.get("text", "").casefold()
            for fact in self.data["facts"]
        }

        if normalized_text in existing_facts:
            return False

        self.data["facts"].append(
            {
                "text": fact_text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_if_needed()

        return True

    def clear(self) -> None:
        self.data = deepcopy(DEFAULT_MEMORY)
        self._save_if_needed()

    def to_prompt_context(self, max_facts: int = 12) -> str:
        """
        Render memory as compact context for the LLM system prompt.
        """

        sections = []

        profile = self.data["profile"]
        if profile:
            sections.append(
                "User profile: "
                + self._format_key_values(profile)
            )

        preferences = self.data["preferences"]
        if preferences:
            sections.append(
                "User preferences: "
                + self._format_key_values(preferences)
            )

        facts = self.data["facts"][-max_facts:]
        if facts:
            fact_lines = [
                f"- {fact['text']}"
                for fact in facts
                if fact.get("text")
            ]
            sections.append(
                "Conversation facts:\n"
                + "\n".join(fact_lines)
            )

        return "\n\n".join(sections)

    def _save_if_needed(self) -> None:
        if self.autosave:
            self.save()

    @staticmethod
    def _normalize_key(key: str) -> str:
        return "_".join(key.strip().lower().split())

    @classmethod
    def _normalize(cls, loaded: Any) -> dict[str, Any]:
        data = deepcopy(DEFAULT_MEMORY)

        if not isinstance(loaded, dict):
            return data

        for section in ("profile", "preferences"):
            if isinstance(loaded.get(section), dict):
                data[section] = {
                    cls._normalize_key(str(key)): value
                    for key, value in loaded[section].items()
                    if str(key).strip()
                }

        if isinstance(loaded.get("facts"), list):
            for fact in loaded["facts"]:
                if isinstance(fact, str):
                    text = fact.strip()
                    created_at = None
                elif isinstance(fact, dict):
                    text = str(fact.get("text", "")).strip()
                    created_at = fact.get("created_at")
                else:
                    continue

                if not text:
                    continue

                data["facts"].append(
                    {
                        "text": text,
                        "created_at": (
                            str(created_at)
                            if created_at
                            else datetime.now(timezone.utc).isoformat()
                        ),
                    }
                )

        return data

    @staticmethod
    def _format_key_values(values: dict[str, Any]) -> str:
        return "; ".join(
            f"{key.replace('_', ' ')}: {value}"
            for key, value in sorted(values.items())
        )
