from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Primary non-cognitive infrastructure stream for body timing / pacemaker use.
PRIMARY_HEARTBEAT_TOPIC = "body/heartbeat"

# Compatibility alias retained while older organs still subscribe to the
# historical topic.  This should eventually disappear after migration.
COMPAT_HEARTBEAT_TOPIC = "clock/tick"

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

HEARTBEAT_TOPICS = {PRIMARY_HEARTBEAT_TOPIC, COMPAT_HEARTBEAT_TOPIC}
HEARTBEAT_SCHEMA = "body.heartbeat.v1"
HEARTBEAT_STREAM_KIND = "body_heartbeat"
HEARTBEAT_STREAM_NAME = "body_clock"


def is_heartbeat_topic(topic: Any) -> bool:
    return str(topic or "") in HEARTBEAT_TOPICS


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
        if meta.get("compat_alias_for") == PRIMARY_HEARTBEAT_TOPIC:
            return True
    return False


def heartbeat_payload(ts: float) -> dict[str, float | str]:
    return {
        "schema": HEARTBEAT_SCHEMA,
        "ts": float(ts),
        "stream": HEARTBEAT_STREAM_NAME,
    }


def heartbeat_meta(*, compat_alias: bool = False) -> dict[str, Any]:
    meta = {
        "source": "system",
        "channel": "body",
        "kind": HEARTBEAT_STREAM_KIND,
        "heartbeat_stream": True,
        "infrastructure_only": True,
        "infrastructure_stream": HEARTBEAT_STREAM_NAME,
        "store_in_memory": False,
        "ui_hidden": True,
        "reinforcement_eligible": False,
        "self_output_track": False,
    }
    if compat_alias:
        meta["compat_alias_for"] = PRIMARY_HEARTBEAT_TOPIC
    return meta


def heartbeat_reason() -> str:
    return PRIMARY_HEARTBEAT_TOPIC
