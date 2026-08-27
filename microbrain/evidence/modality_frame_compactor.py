from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Object/base frames are fast handles.  These caps keep modality packets from
# turning frames into raw sensor archives when a pre-recorder has not compacted
# a payload yet.
MAX_INLINE_TEXT_CHARS = 420
MAX_INLINE_SEQUENCE_ITEMS = 8
MAX_INLINE_MAPPING_ITEMS = 16
MAX_INLINE_DEPTH = 3
MAX_INLINE_TOTAL_CHARS = 3600

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

MODALITY_FRAME_REF_SCHEMA = "modality.frame_ref.v1"
RAW_PAYLOAD_POLICY = "compact_reference_only"
COMMON_REF_KEYS = {
    "artifact_ref",
    "artifact_refs",
    "evidence_ref",
    "evidence_refs",
    "evidence_card_ref",
    "data_ref",
    "frame_ref",
    "image_ref",
    "image_refs",
    "audio_ref",
    "audio_refs",
    "touch_ref",
    "touch_refs",
    "fossil_ref",
    "fossil_refs",
    "ledger_ref",
    "index_ref",
    "ref_index",
    "scaffold_ref",
    "object_ref",
    "scene_ref",
    "proto_id",
    "track_id",
}
COMMON_SUMMARY_KEYS = {
    "summary",
    "description",
    "text",
    "label",
    "status",
    "kind",
    "confidence",
    "sensor",
    "time_range",
    "ts",
    "timestamp",
}
HEAVY_VALUE_KEYS = {
    "raw",
    "samples",
    "sample_values",
    "values",
    "frames",
    "frame_data",
    "pixels",
    "image",
    "image_data",
    "audio",
    "audio_data",
    "waveform",
    "pcm",
    "buffer",
    "bytes",
    "pressure_series",
    "delta_series",
    "touch_matrix",
    "points",
    "vectors",
}


def stable_digest(data: Any, *, size: int = 12) -> str:
    try:
        raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        raw = repr(data)
    return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=size).hexdigest()


def estimate_json_chars(data: Any) -> int:
    try:
        return len(json.dumps(data, sort_keys=True, default=str, separators=(",", ":")))
    except Exception:
        return len(repr(data))


def _clean_key(key: Any) -> str:
    return str(key or "").strip()


def _safe_number(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def compact_value(value: Any, *, depth: int = 0, key_hint: str = "") -> Any:
    """Return a bounded JSON-ish representation suitable for object frames."""
    key_l = str(key_hint or "").lower()
    if depth >= MAX_INLINE_DEPTH:
        if isinstance(value, Mapping):
            return {"schema": "compact.omitted_mapping.v1", "key_count": len(value)}
        if isinstance(value, (list, tuple, set)):
            return {"schema": "compact.omitted_sequence.v1", "count": len(value)}
        text = str(value)
        return text[:MAX_INLINE_TEXT_CHARS] + ("…" if len(text) > MAX_INLINE_TEXT_CHARS else "")

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _safe_number(value)
    if isinstance(value, str):
        if len(value) > MAX_INLINE_TEXT_CHARS:
            return value[:MAX_INLINE_TEXT_CHARS] + "…"
        return value

    if key_l in HEAVY_VALUE_KEYS:
        return _heavy_summary(value)

    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        omitted = 0
        for idx, (raw_key, raw_val) in enumerate(value.items()):
            if idx >= MAX_INLINE_MAPPING_ITEMS:
                omitted += 1
                continue
            key = _clean_key(raw_key)
            if not key:
                continue
            if key.lower() in HEAVY_VALUE_KEYS:
                out[key] = _heavy_summary(raw_val)
            else:
                out[key] = compact_value(raw_val, depth=depth + 1, key_hint=key)
        if len(value) > MAX_INLINE_MAPPING_ITEMS:
            out["_omitted_key_count"] = len(value) - MAX_INLINE_MAPPING_ITEMS + omitted
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        sample = [compact_value(item, depth=depth + 1, key_hint=key_hint) for item in seq[:MAX_INLINE_SEQUENCE_ITEMS]]
        if len(seq) > MAX_INLINE_SEQUENCE_ITEMS:
            return {
                "schema": "compact.sequence_sample.v1",
                "count": len(seq),
                "sample": sample,
                "omitted_count": len(seq) - MAX_INLINE_SEQUENCE_ITEMS,
            }
        return sample

    text = str(value)
    return text[:MAX_INLINE_TEXT_CHARS] + ("…" if len(text) > MAX_INLINE_TEXT_CHARS else "")


def _heavy_summary(value: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "schema": "compact.heavy_value.v1",
        "payload_digest": stable_digest(value, size=10),
        "estimated_json_chars": estimate_json_chars(value),
    }
    if isinstance(value, Mapping):
        summary["kind"] = "mapping"
        summary["key_count"] = len(value)
        summary["sample_keys"] = [str(k) for k in list(value.keys())[:MAX_INLINE_SEQUENCE_ITEMS]]
    elif isinstance(value, (list, tuple, set)):
        seq = list(value)
        summary["kind"] = "sequence"
        summary["count"] = len(seq)
        summary["sample"] = [compact_value(item, depth=MAX_INLINE_DEPTH - 1) for item in seq[:min(3, MAX_INLINE_SEQUENCE_ITEMS)]]
    else:
        summary["kind"] = type(value).__name__
    return summary


def _extract_refs(payload: Mapping[str, Any], ref_keys: set[str]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for key, value in payload.items():
        key_s = _clean_key(key)
        if not key_s:
            continue
        if key_s in ref_keys or key_s.lower() in ref_keys:
            refs[key_s] = compact_value(value, key_hint=key_s)
    return refs


def _summary_from_payload(payload: Mapping[str, Any], summary_keys: set[str]) -> str:
    for key in ("summary", "description", "text", "label", "status", "kind"):
        if key in payload and payload.get(key) not in (None, "", [], {}):
            text = str(payload.get(key))
            return text[:MAX_INLINE_TEXT_CHARS] + ("…" if len(text) > MAX_INLINE_TEXT_CHARS else "")
    for key, value in payload.items():
        if _clean_key(key).lower() in summary_keys and value not in (None, "", [], {}):
            text = str(value)
            return text[:MAX_INLINE_TEXT_CHARS] + ("…" if len(text) > MAX_INLINE_TEXT_CHARS else "")
    return ""


def compact_modality_payload(
    payload: Mapping[str, Any] | Any,
    *,
    modality: str,
    schema: str | None = None,
    keep_keys: Sequence[str] = (),
    ref_keys: Iterable[str] = (),
    summary_keys: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a compact modality packet for a base object frame.

    The packet keeps references and summaries inline, but replaces raw-ish or
    high-volume values with bounded samples/counts.  This is not a durable
    artifact writer; upstream recorder organs should still write real proof
    piles and pass artifact_ref/evidence_ref handles forward.
    """
    if isinstance(payload, Mapping):
        data = dict(payload)
    else:
        data = {"value": payload}

    all_ref_keys = set(COMMON_REF_KEYS)
    all_ref_keys.update(str(k).strip() for k in ref_keys if str(k or "").strip())
    all_summary_keys = set(COMMON_SUMMARY_KEYS)
    all_summary_keys.update(str(k).strip().lower() for k in summary_keys if str(k or "").strip())

    refs = _extract_refs(data, all_ref_keys)
    inline: Dict[str, Any] = {}
    omitted_keys: list[str] = []

    preferred = [str(k) for k in keep_keys if str(k or "").strip()]
    for key in preferred:
        if key in data:
            inline[key] = compact_value(data[key], key_hint=key)

    for key, value in data.items():
        key_s = _clean_key(key)
        if not key_s:
            continue
        key_l = key_s.lower()
        if key_s in inline or key_s in refs:
            continue
        if key_l in all_ref_keys:
            refs[key_s] = compact_value(value, key_hint=key_s)
            continue
        if key_l in all_summary_keys and key_s not in inline:
            inline[key_s] = compact_value(value, key_hint=key_s)
            continue
        if key_l in HEAVY_VALUE_KEYS:
            inline[f"{key_s}_summary"] = _heavy_summary(value)
            omitted_keys.append(key_s)
            continue
        # Keep a small bounded slice of ordinary metadata, but do not turn this
        # into a second full payload copy.
        if len(inline) < MAX_INLINE_MAPPING_ITEMS:
            inline[key_s] = compact_value(value, key_hint=key_s)
        else:
            omitted_keys.append(key_s)

    if extra:
        for key, value in dict(extra).items():
            key_s = _clean_key(key)
            if key_s:
                inline[key_s] = compact_value(value, key_hint=key_s)

    packet: Dict[str, Any] = {
        "schema": str(schema or MODALITY_FRAME_REF_SCHEMA),
        "modality": str(modality or "unknown"),
        "raw_payload_policy": RAW_PAYLOAD_POLICY,
        "payload_digest": stable_digest(data, size=12),
        "payload_estimated_json_chars": estimate_json_chars(data),
        "summary": _summary_from_payload(data, all_summary_keys),
        "refs": refs,
        "inline": inline,
        "omitted_keys": sorted(set(omitted_keys))[:64],
    }
    packet["truncated"] = bool(packet["omitted_keys"] or packet["payload_estimated_json_chars"] > MAX_INLINE_TOTAL_CHARS)
    return packet


def flatten_compat_fields(packet: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Expose selected inline/ref fields at top-level for older callers/tests."""
    out = dict(packet or {})
    inline = out.get("inline", {}) if isinstance(out.get("inline", {}), Mapping) else {}
    refs = out.get("refs", {}) if isinstance(out.get("refs", {}), Mapping) else {}
    for key in keys:
        key_s = str(key or "").strip()
        if not key_s:
            continue
        if key_s in inline and key_s not in out:
            out[key_s] = inline[key_s]
        elif key_s in refs and key_s not in out:
            out[key_s] = refs[key_s]
    return out
