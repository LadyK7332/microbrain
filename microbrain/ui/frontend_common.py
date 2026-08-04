"""Shared frontend helpers for MicroBrain's human-facing interfaces.

This module intentionally contains no Textual or Qt imports.  UI frontends may
come and go; the bridge contract, transcript behavior, pressure snapshot, and
trace/evidence helpers should remain usable by any face.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from microbrain.utils.memdir import resolve_memdir_cli

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# UI-only age windows used to turn recent reinforcement/trainer events into
# visible pulse values.  Units: seconds.
REINFORCEMENT_PULSE_TTL_S = 18.0
TRAINER_PULSE_TTL_S = 24.0

# Runtime KV names containing one of these tokens may be offered as live
# engineering controls when the value is a simple scalar.  This is deliberately
# conservative: state payloads and structural identifiers remain read-only.
RUNTIME_TUNING_NAME_TOKENS = (
    "threshold",
    "bias",
    "ttl",
    "cooldown",
    "timeout",
    "limit",
    "enabled",
    "fps",
    "keep",
    "rate",
    "gain",
    "weight",
    "floor",
    "scale",
    "window",
    "attempt",
    "batch",
    "max_",
    "min_",
)

# Prefixes that are appropriate for engineering/runtime calibration.  This does
# not grant access to arbitrary neuron attributes; only central runtime KV keys
# are considered.
RUNTIME_TUNING_PREFIXES = (
    "affect:",
    "capability:",
    "drive:",
    "hypothesis:",
    "mem_cell:",
    "memory:",
    "scene:",
    "slearn:",
    "thought:",
    "vision:",
)

# Hard cap on recursive evidence extraction so a malformed payload cannot make
# the diagnostic face spend unbounded time walking data.
EVIDENCE_SCAN_ITEM_LIMIT = 256

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

PRESSURE_SCHEMA = "ui.pressure_band.v1"
DASHBOARD_SNAPSHOT_SCHEMA = "ui.dashboard_snapshot.v1"

# These names are references carried by cognition/sensory events.  The
# dashboard may render/open them, but must never invent a path by guessing from
# unrelated files on disk.
EVIDENCE_REFERENCE_KEYS = {
    "artifact_ref",
    "audio_ref",
    "data_ref",
    "evidence_ref",
    "file",
    "file_ref",
    "frame_ref",
    "image_ref",
    "motor_ref",
    "path",
    "source_ref",
    "touch_ref",
    "trace_ref",
}
MEMORY_REFERENCE_KEYS = {
    "cell_id",
    "memory_cell_id",
    "memory_cell_ids",
    "evidence_cell_ids",
    "selected_memory_cell_ids",
}


@dataclass(slots=True)
class UIMessage:
    """Transport-neutral event packet consumed by human-facing UIs."""

    topic: str
    payload: object
    source: str = ""
    meta: dict | None = None
    correlation_id: str = ""
    timestamp: float = 0.0


def safe_json(value: object) -> object:
    """Return a JSON-safe-ish value without letting UI telemetry throw."""

    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def resolve_ui_memdir(memdir: str | None) -> Path:
    try:
        return Path(memdir) if memdir else resolve_memdir_cli(None)
    except Exception:
        return Path.cwd() / "memory"


def load_display_labels(memdir: str | None) -> tuple[str, str]:
    """Return (assistant_label, user_label) from persistent profiles."""

    root = resolve_ui_memdir(memdir)
    assistant = "MB"
    user = "you"

    try:
        pdna_path = root / "pdna_profile.json"
        if pdna_path.exists():
            data = json.loads(pdna_path.read_text(encoding="utf-8"))
            name = str(data.get("name", "") or "").strip()
            if name:
                assistant = name
    except Exception:
        pass

    try:
        user_path = root / "state" / "user_profile.json"
        if user_path.exists():
            data = json.loads(user_path.read_text(encoding="utf-8"))
            name = str(data.get("user_name", "") or "").strip()
            if name:
                user = name
    except Exception:
        pass

    # Safe in both Textual markup and plain Qt labels.
    assistant = assistant.replace("[", "(").replace("]", ")")
    user = user.replace("[", "(").replace("]", ")")
    return assistant, user


class TranscriptWriter:
    """Best-effort UI transcript writer that can never take down the face."""

    def __init__(self, memdir: str | None, *, prefix: str) -> None:
        self.raw_path: Path | None = None
        self.conversation_path: Path | None = None
        try:
            log_dir = resolve_ui_memdir(memdir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.raw_path = log_dir / f"{prefix}_raw.jsonl"
            self.conversation_path = log_dir / f"{prefix}_conversation.log"
        except Exception:
            pass

    @staticmethod
    def _append(path: Path | None, line: str) -> None:
        if path is None:
            return
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line.rstrip("\n") + "\n")
        except Exception:
            pass

    def append_raw(self, msg: UIMessage) -> None:
        record = {
            "ts": msg.timestamp or time.time(),
            "topic": msg.topic,
            "source": msg.source,
            "correlation_id": msg.correlation_id,
            "payload": safe_json(msg.payload),
            "meta": safe_json(msg.meta or {}),
        }
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        except Exception:
            line = repr(record)
        self._append(self.raw_path, line)

    def append_conversation(self, line: str) -> None:
        self._append(self.conversation_path, line)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _age_decay(now: float, ts: Any, *, ttl_s: float) -> float:
    then = _safe_float(ts, 0.0)
    if then <= 0.0:
        return 0.0
    age = max(0.0, now - then)
    if age >= ttl_s:
        return 0.0
    return max(0.0, 1.0 - (age / max(0.001, ttl_s)))


def pressure_snapshot(orch: Any) -> dict[str, Any]:
    """Build the two-speed body/pressure snapshot used by all frontends."""

    now = time.time()
    kv = getattr(orch, "kv_store", {}) or {}

    power_state = _as_dict(kv.get("power:state"))
    boredom = _as_dict(kv.get("drive:boredom"))
    social = _as_dict(kv.get("drive:social_interaction"))
    social_exp = _as_dict(kv.get("drive:social_experimentation"))
    thought_turn = _as_dict(kv.get("thought:turn:last_state"))
    thought_momentum = _as_dict(kv.get("thought:momentum"))
    capability = _as_dict(kv.get("capability:state"))
    maintenance = _as_dict(kv.get("memory:last_sleep_maintenance"))
    reinforce = _as_dict(kv.get("reinforce:last"))
    trainer = _as_dict(kv.get("trainer:last_correction"))
    reward_state = _as_dict(kv.get("affect:reward_state"))
    novelty_state = _as_dict(kv.get("affect:novelty_state"))
    salience_state = _as_dict(kv.get("affect:salience_state"))

    salience = max(
        _safe_float(kv.get("affect:global_salience"), 0.0),
        _safe_float(salience_state.get("level"), 0.0),
    )
    reinforce_raw = abs(
        _safe_float(reinforce.get("weight", reinforce.get("score", 0.0)), 0.0)
    ) / 10.0
    reinforce_reward = max(0.0, min(1.0, reinforce_raw)) * _age_decay(
        now,
        reinforce.get("ts"),
        ttl_s=REINFORCEMENT_PULSE_TTL_S,
    )
    reward_level = max(
        reinforce_reward,
        _safe_float(reward_state.get("level", reward_state.get("dopamine", 0.0)), 0.0),
    )
    train = _age_decay(now, trainer.get("ts"), ttl_s=TRAINER_PULSE_TTL_S)

    curiosity_boost = max(0.0, min(1.0, _safe_float(kv.get("curiosity:boost"), 0.0)))
    curiosity = max(
        curiosity_boost,
        _safe_float(novelty_state.get("level"), 0.0) * 0.35,
        _safe_float(social_exp.get("pressure"), 0.0) * 0.45,
        _safe_float(thought_momentum.get("pressure"), 0.0)
        if str(thought_momentum.get("dominant_intent", "")).lower()
        in {"curiosity", "seek_novelty", "social_experiment"}
        else 0.0,
    )

    body = {
        "power_mode": str(power_state.get("mode") or kv.get("power:mode") or "awake"),
        "charging": bool(power_state.get("charging", False)),
        "sleep": bool(power_state.get("sleep", kv.get("power:sleep", False))),
        "maintenance": str(maintenance.get("status") or maintenance.get("result") or "idle"),
        "memory_pending": int(_safe_float(kv.get("mem_cell:composer:pending_count"), 0.0)),
        "memory_composer": "on" if bool(kv.get("mem_cell:composer:started", False)) else "off",
        "read_sidecar": "on" if bool(kv.get("read:sidecar_started", False)) else "off",
        "cap_available": int(_safe_float(capability.get("available_count"), 0.0)),
        "cap_total": int(_safe_float(capability.get("component_count"), 0.0)),
        "drawer_waiting": int(_safe_float(thought_turn.get("waiting_count"), 0.0)),
        "drawer_ready": int(_safe_float(thought_turn.get("ready_count"), 0.0)),
    }

    pulse = {
        "salience": round(max(0.0, min(1.0, salience)), 3),
        "reward": round(max(0.0, min(1.0, reward_level)), 3),
        "boredom": round(max(0.0, min(1.0, _safe_float(boredom.get("level"), 0.0))), 3),
        "curiosity": round(max(0.0, min(1.0, curiosity)), 3),
        "expression": round(max(0.0, min(1.0, _safe_float(social.get("level"), 0.0))), 3),
        "trainer": round(max(0.0, min(1.0, train)), 3),
        "thought_pressure": round(
            max(0.0, min(1.0, _safe_float(thought_momentum.get("pressure"), 0.0))), 3
        ),
        "thought_intent": str(
            thought_turn.get("dominant_family")
            or thought_momentum.get("dominant_intent")
            or "idle"
        ),
        "thought_status": str(
            thought_turn.get("dominant_status")
            or ("active" if thought_momentum.get("active") else "idle")
        ),
        "novelty_delta": round(_safe_float(boredom.get("novelty_delta"), 0.0), 3),
    }

    return {"schema": PRESSURE_SCHEMA, "ts": now, "body": body, "pulse": pulse}


def extract_text_and_channels(msg: UIMessage) -> tuple[str | None, str, str]:
    payload = msg.payload
    text: str | None = None
    if isinstance(payload, dict) and "text" in payload:
        text = str(payload.get("text", ""))
    elif isinstance(payload, str):
        text = payload

    meta = msg.meta or {}
    channel = str(meta.get("channel", "") or "")
    payload_channel = str(payload.get("channel", "") or "") if isinstance(payload, dict) else ""
    payload_source = str(payload.get("source", "") or "") if isinstance(payload, dict) else ""
    raw_meta = (
        payload.get("raw_meta", {})
        if isinstance(payload, dict) and isinstance(payload.get("raw_meta"), dict)
        else {}
    )
    transport_source = str(raw_meta.get("transport_source", raw_meta.get("source", "")) or "")
    effective_channel = payload_channel or channel
    effective_source = payload_source or transport_source or str(msg.source or "")
    return text, effective_channel, effective_source


def should_show_in_conversation(msg: UIMessage, text: str | None) -> bool:
    if text is None:
        return False
    meta = msg.meta or {}
    if bool(meta.get("ui_hidden", False)) or meta.get("ui_visible") is False:
        return False
    payload = msg.payload
    if isinstance(payload, dict):
        if bool(payload.get("ui_hidden", False)) or payload.get("ui_visible") is False:
            return False
    if msg.topic in {"reason/request", "reason/output"} and not bool(meta.get("ui_visible", False)):
        return False
    return True


def flatten_mapping(value: Mapping[str, Any] | None, *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return out
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            out.update(flatten_mapping(item, prefix=name))
        else:
            out[name] = item
    return out


def runtime_tuning_candidates(kv: Mapping[str, Any]) -> dict[str, Any]:
    """Return simple runtime KV values that look intentionally adjustable."""

    out: dict[str, Any] = {}
    for key, value in kv.items():
        name = str(key)
        if not name.startswith(RUNTIME_TUNING_PREFIXES):
            continue
        lower = name.lower()
        if not any(token in lower for token in RUNTIME_TUNING_NAME_TOKENS):
            continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[name] = value
    return dict(sorted(out.items()))


def _looks_like_file_reference(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lower = text.lower()
    suffixes = (
        ".bmp",
        ".csv",
        ".flac",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".npy",
        ".ogg",
        ".png",
        ".txt",
        ".wav",
        ".webp",
    )
    return lower.endswith(suffixes) or ":\\" in text or text.startswith("/") or text.startswith("\\\\")


def extract_evidence_refs(payload: object) -> list[dict[str, str]]:
    """Extract only references explicitly carried by a cognition/sensory payload."""

    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned = 0

    def add(kind: str, ref: object, key: str) -> None:
        if ref is None:
            return
        text = str(ref).strip()
        if not text:
            return
        marker = (kind, text)
        if marker in seen:
            return
        seen.add(marker)
        refs.append({"kind": kind, "ref": text, "key": key})

    def walk(value: object, parent_key: str = "") -> None:
        nonlocal scanned
        if scanned >= EVIDENCE_SCAN_ITEM_LIMIT:
            return
        scanned += 1

        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lower = key_text.lower()
                if lower in EVIDENCE_REFERENCE_KEYS:
                    if isinstance(item, (str, Path)) and _looks_like_file_reference(str(item)):
                        add("file", item, key_text)
                elif lower in MEMORY_REFERENCE_KEYS:
                    if isinstance(item, (list, tuple, set)):
                        for cell_id in item:
                            add("memory", cell_id, key_text)
                    else:
                        add("memory", item, key_text)
                walk(item, key_text)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, parent_key)

    walk(payload)
    return refs
