"""
Structured tool registry for local assistant actions.

Tools return short, speakable text so the existing TTS pipeline can keep
working with plain strings.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from utils.logger import log


ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, str]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        handler: ToolHandler,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        tool = self._tools.get(name)

        if tool is None:
            return f"I do not have a tool named {name}."

        try:
            return tool.handler(arguments or {})
        except Exception as exc:
            log(f"Tool '{name}' failed: {exc}", level="error", exc_info=True)
            return f"I could not run {name}."

    def schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]


def _set_timer(arguments: dict[str, Any]) -> str:
    seconds = arguments.get("seconds")
    label = str(arguments.get("label") or "timer").strip()

    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "I need a timer duration in seconds."

    if seconds <= 0:
        return "The timer duration needs to be greater than zero seconds."

    if seconds > 24 * 60 * 60:
        return "I can only set timers up to 24 hours."

    def notify():
        log(f"Timer finished: {label} ({seconds}s)")

    timer = threading.Timer(seconds, notify)
    timer.daemon = True
    timer.start()

    return f"Timer set for {_format_duration(seconds)}."


def _get_time(arguments: dict[str, Any]) -> str:
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%A, %B %d").replace(" 0", " ")

    return f"It's {time_str} on {date_str}."


def _get_weather(arguments: dict[str, Any]) -> str:
    location = str(arguments.get("location") or "local").strip()
    base_url = str(
        arguments.get("api_url")
        or os.environ.get("VOICE_AGENT_WEATHER_API_URL")
        or "http://localhost:5000/weather"
    ).strip()

    query = urllib.parse.urlencode({"location": location})
    separator = "&" if "?" in base_url else "?"
    url = f"{base_url}{separator}{query}"

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        log(f"Weather tool failed for {url}: {exc}", level="warning")
        return "I could not fetch the local weather right now."

    summary = payload.get("summary") or payload.get("condition")
    temperature = payload.get("temperature")
    if temperature is None:
        temperature = payload.get("temp")
    unit = payload.get("unit") or "degrees"

    if summary and temperature is not None:
        return f"It's {temperature} {unit} and {summary} in {location}."

    if summary:
        return f"The weather in {location} is {summary}."

    return "I got the weather response, but it did not include a forecast."


def _format_duration(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if remaining_minutes:
        parts.append(
            f"{remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
        )
    if remaining_seconds or not parts:
        parts.append(
            f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
        )

    return " and ".join(parts)


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "set_timer",
        "Set a local timer. The timer completion is logged.",
        {
            "seconds": "integer duration in seconds",
            "label": "optional short timer label",
        },
        _set_timer,
    )
    registry.register(
        "get_time",
        "Get the current local time and date.",
        {},
        _get_time,
    )
    registry.register(
        "get_weather",
        "Fetch weather from the local HTTP weather API.",
        {
            "location": "city or place name, defaults to local",
            "api_url": "optional local weather API URL",
        },
        _get_weather,
    )
    return registry
