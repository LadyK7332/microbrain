from __future__ import annotations

"""Touch artifact recorder.

This module is the first real writer in the sensory evidence stack for touch:
raw/high-volume touch packets become artifact files plus compact percept packets.
Object frames should see only the compact packet; deliberation can open the
artifact when proof is needed.
"""

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from microbrain.evidence.artifact_store import EvidenceArtifactStore
from microbrain.evidence.modality_frame_compactor import compact_modality_payload

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

TOUCH_ARTIFACT_PREFIX = "touch_delta"
TOUCH_COMPACT_SCHEMA = "touch.compact.v1"
TOUCH_ARTIFACT_SOURCE_SCHEMA = "touch.artifact_record.v1"
MAX_INLINE_FEATURES = 32

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

TOUCH_HEAVY_KEYS = (
    "records",
    "samples",
    "sample_values",
    "values",
    "pressure_series",
    "delta_series",
    "touch_matrix",
    "points",
    "vectors",
    "raw",
)
TOUCH_FEATURE_KEYS = (
    "pressure_peak",
    "pressure_avg",
    "pressure_mean",
    "pressure_min",
    "pressure_max",
    "slip",
    "slip_level",
    "yield",
    "yield_level",
    "texture",
    "texture_hint",
    "contact",
    "contact_state",
    "temperature",
    "vibration",
    "sensor",
    "sensor_id",
    "finger",
    "pad",
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out or out in (float("inf"), float("-inf")):
        return float(default)
    return out


def _clean_text(value: Any, *, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _list_from(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _time_range_from(payload: Mapping[str, Any], *, timestamp: float) -> list[float]:
    raw = payload.get("time_range") or payload.get("time_window") or []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and len(raw) >= 2:
        return [_safe_float(raw[0], timestamp), _safe_float(raw[1], timestamp)]
    start = payload.get("start_ts", payload.get("started_at", payload.get("ts", timestamp)))
    end = payload.get("end_ts", payload.get("ended_at", payload.get("timestamp", timestamp)))
    return [_safe_float(start, timestamp), _safe_float(end, timestamp)]


def _records_from_payload(payload: Mapping[str, Any]) -> list[Any]:
    for key in TOUCH_HEAVY_KEYS:
        value = payload.get(key)
        if value in (None, ""):
            continue
        records = _list_from(value)
        if records:
            return records
    return [dict(payload)]


def _features_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    nested_features = payload.get("features")
    if isinstance(nested_features, Mapping):
        for key, value in nested_features.items():
            if len(features) >= MAX_INLINE_FEATURES:
                break
            features[str(key)] = value
    for key in TOUCH_FEATURE_KEYS:
        if len(features) >= MAX_INLINE_FEATURES:
            break
        if key in payload and key not in features:
            features[key] = payload.get(key)
    return features


def _claims_from_features(features: Mapping[str, Any], explicit_claims: Iterable[Any] | None = None) -> list[str]:
    claims: list[str] = []
    for claim in explicit_claims or []:
        text = _clean_text(claim, limit=96)
        if text and text not in claims:
            claims.append(text)

    text_blob = " ".join(str(v).lower() for v in features.values() if v not in (None, "", [], {}))
    if any(word in text_blob for word in ("soft", "compress", "yield")):
        claims.append("touch.yielding_or_soft")
    if any(word in text_blob for word in ("fuzzy", "fur", "rough", "smooth", "texture")):
        claims.append("touch.texture_hint")
    if "low" in text_blob and "slip" in text_blob:
        claims.append("touch.low_slip")
    if "sharp" in text_blob:
        claims.append("touch.edge_or_sharpness_hint")
    if "contact" in text_blob or features.get("contact") is not None:
        claims.append("touch.contact_observed")

    # De-dupe while preserving order.
    out: list[str] = []
    for claim in claims:
        if claim and claim not in out:
            out.append(claim)
    return out[:32]


def _summary_from_payload(payload: Mapping[str, Any], features: Mapping[str, Any], sample_count: int) -> str:
    for key in ("summary", "description", "text", "label"):
        if payload.get(key) not in (None, "", [], {}):
            return _clean_text(payload.get(key), limit=240)
    parts: list[str] = []
    texture = features.get("texture_hint", features.get("texture"))
    if texture not in (None, "", [], {}):
        parts.append(f"texture={texture}")
    yield_hint = features.get("yield", features.get("yield_level"))
    if yield_hint not in (None, "", [], {}):
        parts.append(f"yield={yield_hint}")
    slip = features.get("slip", features.get("slip_level"))
    if slip not in (None, "", [], {}):
        parts.append(f"slip={slip}")
    pressure = features.get("pressure_peak", features.get("pressure_max", features.get("pressure_avg")))
    if pressure not in (None, "", [], {}):
        parts.append(f"pressure={pressure}")
    if parts:
        return _clean_text("touch contact: " + ", ".join(str(p) for p in parts), limit=240)
    return f"touch artifact with {max(0, int(sample_count))} sample record(s)"


def should_record_touch_payload(payload: Any) -> bool:
    """Return True only for raw-ish touch payloads that need an artifact."""
    data = _as_mapping(payload)
    schema = str(data.get("schema", "") or "")
    if schema == TOUCH_COMPACT_SCHEMA:
        return False
    if data.get("artifact_ref") or data.get("evidence_ref") or data.get("evidence_card"):
        # Already has a persisted proof handle; let frame compaction carry refs.
        return False
    return any(key in data and data.get(key) not in (None, "", [], {}) for key in TOUCH_HEAVY_KEYS)


def record_touch_artifact(
    base_dir: str | Path,
    payload: Any,
    *,
    timestamp: float | None = None,
    source: str = "touch_artifact_recorder",
    prefix: str = TOUCH_ARTIFACT_PREFIX,
) -> dict[str, Any]:
    """Persist raw-ish touch data and return compact percept/evidence packets."""
    stamp = float(timestamp or time.time())
    data = _as_mapping(payload)
    records = _records_from_payload(data)
    features = _features_from_payload(data)
    claims = _claims_from_features(features, data.get("claims_supported") or data.get("claims"))
    confidence = _safe_float(data.get("confidence", data.get("score", 0.0)), 0.0)
    time_range = _time_range_from(data, timestamp=stamp)
    summary = _summary_from_payload(data, features, len(records))

    store = EvidenceArtifactStore(base_dir)
    artifact_records = [
        {
            "schema": TOUCH_ARTIFACT_SOURCE_SCHEMA,
            "created_at": stamp,
            "source": source,
            "payload_meta": {
                "schema": str(data.get("schema", "") or ""),
                "sensor": str(data.get("sensor", data.get("sensor_id", "")) or ""),
                "time_range": time_range,
            },
            "features": features,
            "records": records,
        }
    ]
    card = store.write_jsonl_artifact(
        modality="touch",
        records=artifact_records,
        prefix=prefix,
        summary=summary,
        claims_supported=claims,
        confidence=confidence,
        timestamp=stamp,
        time_range=time_range,
        source=source,
        tags=["touch", "touch_delta", "raw_artifact"],
        meta={
            "raw_payload_policy": "artifact_written_reference_only",
            "source_schema": str(data.get("schema", "") or ""),
            "sample_record_count": len(records),
        },
    )
    compact_ref = store.compact_ref(card)
    compact_payload = {
        "schema": TOUCH_COMPACT_SCHEMA,
        "summary": summary,
        "features": features,
        "claims_supported": claims,
        "confidence": confidence,
        "time_range": time_range,
        "sample_count": len(records),
        "artifact_ref": str(card.get("artifact_ref", "") or ""),
        "evidence_ref": compact_ref,
        "raw_payload_policy": "artifact_written_reference_only",
        "source_schema": str(data.get("schema", "") or ""),
    }
    # Run the compact payload through the same frame compactor to make sure any
    # odd upstream feature values stay bounded before they hit object/base.
    frame_packet = compact_modality_payload(
        compact_payload,
        modality="touch",
        schema="touch.frame_ref.v1",
        keep_keys=("schema", "summary", "features", "claims_supported", "confidence", "time_range", "sample_count"),
        ref_keys=("artifact_ref", "evidence_ref"),
        extra={"artifact_written": True},
    )
    return {
        "schema": "touch.artifact_recorded.v1",
        "artifact_card": card,
        "evidence_ref": compact_ref,
        "percept_payload": compact_payload,
        "frame_packet": frame_packet,
    }
