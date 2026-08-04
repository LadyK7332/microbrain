from __future__ import annotations

import re
from typing import Any, Dict

from microbrain.orchestrator.neuron_base import Event
from microbrain.utils.heartbeat_stream import is_infrastructure_event

_INTERNAL_CHANNELS = {"thought", "internal"}
_CONTROL_UI_TOPICS = {"ui/status", "ui/error", "control/status", "control/error"}
_BLOCK_KINDS = {
    "initiative_reflection",
    "internal_reflection",
    "status_introspection",
    "debug",
    "control_reply",
    "control_status",
    "control_error",
    "command_status",
    "command_error",
}
_JUNK_PATTERNS = [
    re.compile(r"^internal reflection only\b", re.I),
    re.compile(r"llm backend not configured", re.I),
    re.compile(r"reasoning backend returned empty", re.I),
    re.compile(r"my reasoning core isn't wired to a model", re.I),
    re.compile(r"^reinforcement menu is still open\b", re.I),
]

def _text_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return str(payload or "").strip()

def _raw_meta(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("raw_meta"), dict):
        return dict(payload.get("raw_meta") or {})
    return {}

def classify_event_for_memory(event: Event) -> Dict[str, Any]:
    if is_infrastructure_event(event):
        return {
            "text": "",
            "channel": "body",
            "kind": "body_infrastructure",
            "source": str(event.source or "system"),
            "transport_source": str(event.source or "system"),
            "role": "system",
            "is_internal": True,
            "is_system": True,
            "is_control_ui_topic": False,
            "explicit_no_memory": True,
            "junk_reason": "body_infrastructure",
            "allow_longterm": False,
            "allow_trace": False,
            "allow_hrm": False,
            "allow_pattern": False,
        }

    payload = event.payload
    text = _text_from_payload(payload)
    raw_meta = _raw_meta(payload)
    event_meta = dict(event.meta or {})

    channel = ""
    if isinstance(payload, dict):
        channel = str(payload.get("channel", "") or "")
    channel = channel or str(event_meta.get("channel", "") or raw_meta.get("channel", "") or "")

    style = str(payload.get("style", "") or "") if isinstance(payload, dict) else ""
    kind = str(event_meta.get("kind", "") or (payload.get("kind", "") if isinstance(payload, dict) else "") or "")
    source = ""
    if isinstance(payload, dict):
        source = str(payload.get("source", "") or "")
    source = source or str(raw_meta.get("source", "") or event_meta.get("source", "") or event.source or "")
    transport_source = str(raw_meta.get("transport_source", source) or source)
    control = bool(event_meta.get("control", False) or (payload.get("control", False) if isinstance(payload, dict) else False))
    explicit_no_memory = (
        event_meta.get("store_in_memory") is False
        or event_meta.get("cognitive_visible") is False
        or (isinstance(payload, dict) and payload.get("store_in_memory") is False)
        or (isinstance(payload, dict) and payload.get("cognitive_visible") is False)
    )

    role = "user"
    if event.topic == "act/speech":
        role = "system" if style == "system" else "assistant"
    elif source in ("assistant", "system", "internal"):
        role = source

    is_control_ui_topic = event.topic in _CONTROL_UI_TOPICS or event.topic.startswith("ui/")
    is_internal = (
        explicit_no_memory
        or is_control_ui_topic
        or (channel in _INTERNAL_CHANNELS)
        or (source == "internal")
        or (kind in _BLOCK_KINDS)
    )
    is_system = control or role == "system" or is_control_ui_topic
    junk_reason = ""
    if not text:
        junk_reason = "empty_text"
    elif text.lstrip().startswith("/"):
        junk_reason = "control_command_text"
    else:
        for pat in _JUNK_PATTERNS:
            if pat.search(text):
                junk_reason = f"junk_text:{pat.pattern}"
                break

    allow_longterm = bool(text) and not is_internal and not is_system and not junk_reason
    allow_trace = bool(text) and not is_internal and not junk_reason
    allow_hrm = allow_longterm
    allow_pattern = allow_longterm and role == "user" and channel not in _INTERNAL_CHANNELS

    return {
        "text": text,
        "channel": channel,
        "kind": kind,
        "source": source,
        "transport_source": transport_source,
        "role": role,
        "is_internal": is_internal,
        "is_system": is_system,
        "is_control_ui_topic": is_control_ui_topic,
        "explicit_no_memory": explicit_no_memory,
        "junk_reason": junk_reason,
        "allow_longterm": allow_longterm,
        "allow_trace": allow_trace,
        "allow_hrm": allow_hrm,
        "allow_pattern": allow_pattern,
    }
