"""Pure helpers for compact dashboard signal/status instruments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Prefer human-facing body/sense capabilities first. Unknown/new capabilities
# are appended alphabetically so new organs automatically appear without a UI patch.
CAPABILITY_DISPLAY_ORDER = (
    "textual_available",
    "audio_available",
    "vision_available",
    "depth_available",
    "lidar_available",
    "motion_available",
    "power_available",
    "user_assist_available",
    "safety_clear",
    "guardian_clear",
    "hazard_clear",
    "speech_allowed",
    "expression_allowed",
    "awake",
    "not_sleeping",
    "not_charging",
)

CAPABILITY_SHORT_LABELS = {
    "textual_available": "text",
    "audio_available": "audio",
    "vision_available": "vision",
    "depth_available": "depth",
    "lidar_available": "lidar",
    "motion_available": "motion",
    "power_available": "power",
    "user_assist_available": "assist",
    "safety_clear": "safety",
    "guardian_clear": "guardian",
    "hazard_clear": "hazard",
    "speech_allowed": "speech",
    "expression_allowed": "express",
    "awake": "awake",
    "not_sleeping": "not-sleep",
    "not_charging": "not-charge",
}


def _name_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def capability_signal_map(payload: Mapping[str, Any] | None) -> dict[str, bool]:
    """Return ordered capability booleans from a ``capability/state`` payload.

    Actual component state wins over ``alias_available``.  This matters because
    aliases are fallback/readiness answers: for example a lidar requirement may
    be satisfiable through vision even when a physical lidar sensor is absent.
    The dashboard status lamps should describe the actual body first.
    """

    if not isinstance(payload, Mapping):
        return {}

    available = _name_set(payload.get("available_components"))
    unavailable = _name_set(payload.get("unavailable_components"))
    names = available | unavailable

    signals: dict[str, bool] = {}
    if names:
        for name in names:
            signals[name] = name in available
    else:
        aliases = payload.get("alias_available")
        if isinstance(aliases, Mapping):
            for name, value in aliases.items():
                key = str(name).strip()
                if key:
                    signals[key] = bool(value)

    order_index = {name: i for i, name in enumerate(CAPABILITY_DISPLAY_ORDER)}
    return dict(sorted(signals.items(), key=lambda item: (order_index.get(item[0], 10_000), item[0])))


def capability_short_label(name: str) -> str:
    key = str(name or "").strip()
    if key in CAPABILITY_SHORT_LABELS:
        return CAPABILITY_SHORT_LABELS[key]
    for suffix in ("_available", "_allowed", "_clear"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key.replace("_", "-") or "?"


def capability_counts(payload: Mapping[str, Any] | None) -> tuple[int, int]:
    signals = capability_signal_map(payload)
    up = sum(1 for value in signals.values() if value)
    return up, len(signals)
