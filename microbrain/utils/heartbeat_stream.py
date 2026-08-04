from __future__ import annotations

import math
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Canonical MicroBrain body pacemaker.
# Unit: hertz / seconds.  20 TPS gives one nominal body tick every 50 ms.
HEARTBEAT_HZ = 20.0
HEARTBEAT_INTERVAL_S = 1.0 / HEARTBEAT_HZ

# Primary non-cognitive infrastructure stream for body timing / pacemaker use.
PRIMARY_HEARTBEAT_TOPIC = "body/heartbeat"

# Historical subscription/producer alias.  The runtime canonicalizes this to
# ``body/heartbeat``; it is never emitted as a second bus event.
COMPAT_HEARTBEAT_TOPIC = "clock/tick"

# Derived body-service topics.  These remain on the body/infrastructure bus and
# are scheduling opportunities, not semantic evidence.
SERVICE_TOPIC_PREFIX = "body/service/"

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

HEARTBEAT_SCHEMA = "body.heartbeat.v2"
SERVICE_TICK_SCHEMA = "body.service_tick.v2"
HEARTBEAT_STREAM_KIND = "body_heartbeat"
HEARTBEAT_STREAM_NAME = "body_clock"
SERVICE_STREAM_KIND = "body_service_tick"
INFRASTRUCTURE_EVENT_CLASS = "infrastructure"


def _clean_target(target: Any) -> str:
    text = str(target or "").strip().lower().replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})[:64]


def service_topic(target: Any) -> str:
    cleaned = _clean_target(target)
    if not cleaned:
        raise ValueError("body service target must be non-empty")
    return f"{SERVICE_TOPIC_PREFIX}{cleaned}"


def service_target(topic: Any) -> str:
    text = str(topic or "")
    if not text.startswith(SERVICE_TOPIC_PREFIX):
        return ""
    return _clean_target(text[len(SERVICE_TOPIC_PREFIX) :])


def canonical_topic(topic: Any) -> str:
    """Return the authoritative routing topic for infrastructure aliases."""
    text = str(topic or "")
    if text == COMPAT_HEARTBEAT_TOPIC:
        return PRIMARY_HEARTBEAT_TOPIC
    return text


def canonical_subscription_topic(topic: Any) -> str:
    """Normalize legacy heartbeat subscriptions without creating a second pulse."""
    return canonical_topic(topic)


def is_heartbeat_topic(topic: Any) -> bool:
    return canonical_topic(topic) == PRIMARY_HEARTBEAT_TOPIC


def is_service_topic(topic: Any) -> bool:
    return canonical_topic(topic).startswith(SERVICE_TOPIC_PREFIX)


def is_infrastructure_topic(topic: Any) -> bool:
    canonical = canonical_topic(topic)
    return canonical == PRIMARY_HEARTBEAT_TOPIC or canonical.startswith(SERVICE_TOPIC_PREFIX)


def is_heartbeat_event(event: Any) -> bool:
    topic = getattr(event, "topic", event)
    if is_heartbeat_topic(topic):
        return True
    meta = getattr(event, "meta", {})
    if isinstance(meta, Mapping):
        if meta.get("infrastructure_stream") == HEARTBEAT_STREAM_NAME:
            return True
        if meta.get("heartbeat_stream") is True:
            return True
    return False


def is_infrastructure_event(event: Any) -> bool:
    topic = getattr(event, "topic", event)
    if is_infrastructure_topic(topic):
        return True
    meta = getattr(event, "meta", {})
    if not isinstance(meta, Mapping):
        return False
    if str(meta.get("event_class", "") or "").strip().lower() == INFRASTRUCTURE_EVENT_CLASS:
        return True
    if meta.get("infrastructure_only") is True:
        return True
    if meta.get("heartbeat_stream") is True:
        return True
    return False


def canonicalize_event_in_place(event: Any) -> Any:
    """Canonicalize a legacy ``clock/tick`` Event without cloning it.

    This is a routing compatibility shim only.  A producer may still submit the
    historical alias, but the runtime immediately converts it to the canonical
    body heartbeat before enqueueing.  No duplicate event is emitted.
    """
    topic = str(getattr(event, "topic", "") or "")
    if topic != COMPAT_HEARTBEAT_TOPIC:
        return event
    try:
        event.topic = PRIMARY_HEARTBEAT_TOPIC
        meta = getattr(event, "meta", None)
        if not isinstance(meta, dict):
            meta = {}
            event.meta = meta
        meta.setdefault("compat_alias_from", COMPAT_HEARTBEAT_TOPIC)
        meta.setdefault("event_class", INFRASTRUCTURE_EVENT_CLASS)
        meta.setdefault("semantic_input", False)
        meta.setdefault("store_in_memory", False)
        meta.setdefault("memory_eligible", False)
        meta.setdefault("reinforcement_eligible", False)
        meta.setdefault("correlation_propagation", False)
        meta.setdefault("ui_hidden", True)
    except Exception:
        pass
    return event


def heartbeat_payload(
    *,
    tick: int,
    epoch_s: float,
    monotonic_s: float,
    delta_s: float,
    drift_s: float = 0.0,
    missed_estimate: int = 0,
) -> dict[str, float | int | str]:
    """Build one canonical body-heartbeat packet.

    Tick count is a scheduling coordinate.  ``monotonic_s`` and ``delta_s`` are
    elapsed-time truth; consumers must never infer elapsed time from tick count.
    Missed pulses are summarized, never replayed as a catch-up storm.
    """
    return {
        "schema": HEARTBEAT_SCHEMA,
        "tick": max(1, int(tick)),
        "epoch_s": float(epoch_s),
        "ts": float(epoch_s),  # compatibility for older timing helpers
        "monotonic_s": float(monotonic_s),
        "delta_s": max(0.0, float(delta_s)),
        "drift_s": float(drift_s),
        "missed_estimate": max(0, int(missed_estimate)),
        "nominal_hz": HEARTBEAT_HZ,
        "nominal_interval_s": HEARTBEAT_INTERVAL_S,
        "stream": HEARTBEAT_STREAM_NAME,
    }


def heartbeat_meta() -> dict[str, Any]:
    return {
        "source": "system",
        "channel": "body",
        "kind": HEARTBEAT_STREAM_KIND,
        "event_class": INFRASTRUCTURE_EVENT_CLASS,
        "semantic_input": False,
        "heartbeat_stream": True,
        "infrastructure_only": True,
        "infrastructure_stream": HEARTBEAT_STREAM_NAME,
        "store_in_memory": False,
        "memory_eligible": False,
        "ui_hidden": True,
        "cognitive_visible": False,
        "reinforcement_eligible": False,
        "correlation_propagation": False,
        "self_output_track": False,
    }


def service_tick_payload(
    heartbeat: Mapping[str, Any],
    *,
    target: str,
    mode: str,
    divisor: int,
) -> dict[str, Any]:
    cleaned_target = _clean_target(target)
    return {
        "schema": SERVICE_TICK_SCHEMA,
        "target": cleaned_target,
        "mode": str(mode),
        "divisor": max(1, int(divisor)),
        "tick": int(heartbeat.get("tick", 0) or 0),
        "epoch_s": float(heartbeat.get("epoch_s", heartbeat.get("ts", 0.0)) or 0.0),
        "ts": float(heartbeat.get("epoch_s", heartbeat.get("ts", 0.0)) or 0.0),
        "monotonic_s": float(heartbeat.get("monotonic_s", 0.0) or 0.0),
        "delta_s": max(0.0, float(heartbeat.get("delta_s", 0.0) or 0.0)),
        "nominal_interval_s": HEARTBEAT_INTERVAL_S,
    }


def service_tick_meta(target: str) -> dict[str, Any]:
    cleaned_target = _clean_target(target)
    return {
        "source": "body_scheduler",
        "channel": "body",
        "kind": SERVICE_STREAM_KIND,
        "event_class": INFRASTRUCTURE_EVENT_CLASS,
        "semantic_input": False,
        "infrastructure_only": True,
        "service_target": cleaned_target,
        "store_in_memory": False,
        "memory_eligible": False,
        "ui_hidden": True,
        "cognitive_visible": False,
        "reinforcement_eligible": False,
        "correlation_propagation": False,
        "self_output_track": False,
    }


def service_tick_is_for(event: Any, target: str) -> bool:
    """Return True when an infrastructure service event is for ``target``.

    The old aggregate ``body/service_tick`` shape is accepted for tests / old
    plugins, but the v2 scheduler emits target-specific topics.
    """
    expected = _clean_target(target)
    topic = str(getattr(event, "topic", "") or "")
    if service_target(topic) == expected:
        return True
    payload = getattr(event, "payload", {})
    if topic == "body/service_tick" and isinstance(payload, Mapping):
        if _clean_target(payload.get("target")) == expected:
            return True
        return expected in {_clean_target(x) for x in payload.get("due_targets", []) or []}
    return False


def event_epoch_s(event: Any, default: float = 0.0) -> float:
    payload = getattr(event, "payload", {})
    if isinstance(payload, Mapping):
        for key in ("epoch_s", "ts"):
            try:
                value = float(payload.get(key, 0.0) or 0.0)
            except Exception:
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                return value
    try:
        value = float(getattr(event, "timestamp", 0.0) or 0.0)
    except Exception:
        value = 0.0
    return value if math.isfinite(value) and value > 0.0 else float(default)


def heartbeat_reason() -> str:
    return PRIMARY_HEARTBEAT_TOPIC
