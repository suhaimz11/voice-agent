"""
Audio device discovery and selection helpers.

Configuration:
    VOICE_AGENT_INPUT_DEVICE   -> microphone index or name fragment
    VOICE_AGENT_OUTPUT_DEVICE  -> speaker index or name fragment
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import sounddevice as sd

from utils.logger import log


INPUT_DEVICE_ENV = "VOICE_AGENT_INPUT_DEVICE"
OUTPUT_DEVICE_ENV = "VOICE_AGENT_OUTPUT_DEVICE"

_VIRTUAL_DEVICE_PREFIXES = (
    "microsoft sound mapper",
    "primary sound",
    "primary sound capture",
)


def log_audio_devices() -> None:
    """
    Log all currently available audio devices with input/output capabilities.
    """

    try:
        devices = list(sd.query_devices())
        default_input, default_output = _default_device_indexes()

    except Exception as e:
        log(f"Unable to query audio devices: {e}", level="warning")
        return

    if not devices:
        log("No audio devices reported by sounddevice.", level="warning")
        return

    available = _available_device_indexes(devices)

    log("Available audio devices:")

    for index, device in enumerate(devices):
        if index not in available:
            continue

        name = str(device.get("name", "unknown"))
        input_channels = int(device.get("max_input_channels", 0))
        output_channels = int(device.get("max_output_channels", 0))
        sample_rate = device.get("default_samplerate", "unknown")

        labels = []

        if index == default_input:
            labels.append("default input")

        if index == default_output:
            labels.append("default output")

        label_text = f" ({', '.join(labels)})" if labels else ""

        log(
            (
                f"  [{index}] {name} | "
                f"in={input_channels}, out={output_channels}, "
                f"default_sr={sample_rate}{label_text}"
            )
        )


def resolve_input_device(selector: str | int | None = None) -> int | None:
    """
    Resolve a configured microphone selector into a sounddevice index.
    """

    return _resolve_device(
        selector=selector,
        env_name=INPUT_DEVICE_ENV,
        kind="input",
    )


def resolve_output_device(selector: str | int | None = None) -> int | None:
    """
    Resolve a configured speaker selector into a sounddevice index.
    """

    return _resolve_device(
        selector=selector,
        env_name=OUTPUT_DEVICE_ENV,
        kind="output",
    )


def _resolve_device(
    selector: str | int | None,
    env_name: str,
    kind: str,
) -> int | None:

    selected = selector

    if selected is None:
        selected = os.environ.get(env_name)

    if selected is None or str(selected).strip() == "":
        log(f"No {kind} device configured; using system default.")
        return None

    selected_text = str(selected).strip()

    try:
        devices = list(sd.query_devices())

    except Exception as e:
        log(
            f"Unable to resolve {kind} device '{selected_text}': {e}",
            level="warning",
        )
        return None

    if selected_text.lstrip("-").isdigit():
        return _resolve_device_index(
            index=int(selected_text),
            devices=devices,
            kind=kind,
            selected_text=selected_text,
        )

    return _resolve_device_name(
        selected_text=selected_text,
        devices=devices,
        kind=kind,
    )


def _resolve_device_index(
    index: int,
    devices: list[Any],
    kind: str,
    selected_text: str,
) -> int | None:

    if index < 0 or index >= len(devices):
        log(
            f"Configured {kind} device index '{selected_text}' is out of range.",
            level="warning",
        )
        return None

    if not _is_device_available(index):
        log(
            (
                f"Configured {kind} device [{index}] "
                f"{devices[index].get('name', 'unknown')} is not currently available."
            ),
            level="warning",
        )
        return None

    device = devices[index]

    if not _supports_kind(device, kind):
        log(
            (
                f"Configured {kind} device [{index}] "
                f"{device.get('name', 'unknown')} does not support {kind}."
            ),
            level="warning",
        )
        return None

    log(f"Using configured {kind} device: {_device_label(index, device)}")

    return index


def _resolve_device_name(
    selected_text: str,
    devices: list[Any],
    kind: str,
) -> int | None:

    selected_lower = selected_text.lower()
    available = _available_device_indexes(devices)

    matches = [
        (index, device)
        for index, device in enumerate(devices)
        if (
            index in available
            and _supports_kind(device, kind)
            and selected_lower in str(device.get("name", "")).lower()
        )
    ]

    if not matches:
        log(
            f"No {kind} device matched '{selected_text}'. Using system default.",
            level="warning",
        )
        return None

    exact_matches = [
        (index, device)
        for index, device in matches
        if str(device.get("name", "")).lower() == selected_lower
    ]

    selected_index, selected_device = (
        exact_matches[0] if exact_matches else matches[0]
    )

    if len(matches) > 1 and not exact_matches:
        log(
            (
                f"Multiple {kind} devices matched '{selected_text}'. "
                f"Using first match: {_device_label(selected_index, selected_device)}"
            ),
            level="warning",
        )

    log(
        f"Using configured {kind} device: "
        f"{_device_label(selected_index, selected_device)}"
    )

    return selected_index


def _is_device_available(index: int) -> bool:
    """Return True only if the device can actually be opened right now."""
    try:
        device = sd.query_devices(index)
        input_ch = int(device.get("max_input_channels", 0))
        output_ch = int(device.get("max_output_channels", 0))

        if input_ch == 0 and output_ch == 0:
            return False

        # Fast pre-filter: skip Windows virtual mapper aliases
        name_lower = str(device.get("name", "")).lower()
        if any(name_lower.startswith(prefix) for prefix in _VIRTUAL_DEVICE_PREFIXES):
            return False

        # Actually try to open a stream to confirm hardware is present
        if input_ch > 0:
            with sd.InputStream(device=index, channels=1, samplerate=16000):
                pass
        else:
            with sd.OutputStream(device=index, channels=1, samplerate=44100):
                pass

        return True

    except Exception:
        return False


def _available_device_indexes(devices: list[Any]) -> set[int]:
    """Check all devices in parallel to keep startup fast."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_is_device_available, i): i
            for i in range(len(devices))
        }
        return {futures[f] for f in as_completed(futures) if f.result()}


def _supports_kind(device: Any, kind: str) -> bool:
    channel_key = (
        "max_input_channels"
        if kind == "input"
        else "max_output_channels"
    )

    return int(device.get(channel_key, 0)) > 0


def _device_label(index: int, device: Any) -> str:
    return f"[{index}] {device.get('name', 'unknown')}"


def _default_device_indexes() -> tuple[int | None, int | None]:
    default_device = sd.default.device

    if isinstance(default_device, (list, tuple)):
        return _index_or_none(default_device[0]), _index_or_none(default_device[1])

    index = _index_or_none(default_device)

    return index, index


def _index_or_none(value: Any) -> int | None:
    try:
        index = int(value)

    except (TypeError, ValueError):
        return None

    if index < 0:
        return None

    return index