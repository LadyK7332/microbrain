from __future__ import annotations

"""Bind bounded loaded evidence back into deliberation-friendly observations.

The evidence loader opens a tiny window into an artifact.  This module turns that
loaded sample into a compact observation event that hypothesis/review/safety
organs can consume without normal memory swallowing the raw proof pile.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

MAX_OBSERVATION_ITEMS = 4
MAX_OBSERVATION_ITEM_KEYS = 14
MAX_OBSERVATION_TEXT_CHARS = 420
MAX_SUMMARY_CHARS = 360

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

EVIDENCE_OBSERVATION_SCHEMA = "evidence.observation.v1"
EVIDENCE_OBSERVATION_ERROR_SCHEMA = "evidence.observation_error.v1"
EVIDENCE_STATUS = "bounded_sample_not_truth"
RAW_POLICY = "observation_only_no_raw_memory_ingest"

ROUTE_TOPIC_BY_TRIGGER_PREFIX: tuple[tuple[str, str], ...] = (
    ("hypothesis/", "hypothesis/evidence_observation"),
    ("review/", "review/evidence_observation"),
    ("trainer/", "trainer/evidence_observation"),
    ("scene/", "scene/evidence_observation"),
    ("recognition/", "recognition/evidence_observation"),
    ("safety/", "safety/evidence_observation"),
    ("object/", "object/evidence_observation"),
    ("memory/", "memory/evidence_observation"),
    ("thought/", "thought/evidence_sample"),
)


def _stable_id(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _short_text(value: Any, *, limit: int = MAX_OBSERVATION_TEXT_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _bound_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _short_text(value)
    if depth >= 2:
        return _short_text(_json_text(value), limit=MAX_OBSERVATION_TEXT_CHARS)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= MAX_OBSERVATION_ITEM_KEYS:
                out["_omitted_key_count"] = len(value) - MAX_OBSERVATION_ITEM_KEYS
                break
            out[str(key)] = _bound_value(val, depth=depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seq = list(value)
        sample = [_bound_value(item, depth=depth + 1) for item in seq[:6]]
        if len(seq) > 6:
            return {"schema": "evidence.observation.sequence_sample.v1", "count": len(seq), "sample": sample}
        return sample
    return _short_text(str(value))


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _items_sample(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(list(items)[:MAX_OBSERVATION_ITEMS]):
        if isinstance(item, Mapping):
            compact = dict(_bound_value(item))
        else:
            compact = {"value": _bound_value(item)}
        compact.setdefault("sample_ordinal", idx)
        out.append(compact)
    return out


def _summarize_loaded(loaded: Mapping[str, Any], *, ok: bool, item_count: int, mode: str, ref: str) -> str:
    if not ok:
        err = _short_text(loaded.get("error", "evidence load failed"), limit=220)
        return _short_text(f"evidence load failed for {ref or 'artifact'}: {err}", limit=MAX_SUMMARY_CHARS)
    kind = str(loaded.get("artifact_kind", "artifact") or "artifact")
    query = _short_text(loaded.get("query", ""), limit=120)
    if item_count <= 0:
        if mode == "directed" and query:
            return _short_text(f"no matching {kind} evidence items found for query '{query}'", limit=MAX_SUMMARY_CHARS)
        return _short_text(f"no {kind} evidence items loaded from {ref or 'artifact'}", limit=MAX_SUMMARY_CHARS)
    if mode == "scatter":
        return _short_text(f"sampled {item_count} {kind} evidence item(s) from {ref or 'artifact'}", limit=MAX_SUMMARY_CHARS)
    if mode == "walk":
        return _short_text(f"walked {item_count} {kind} evidence item(s) from {ref or 'artifact'}", limit=MAX_SUMMARY_CHARS)
    if mode == "directed":
        if query:
            return _short_text(f"loaded {item_count} matching {kind} evidence item(s) for query '{query}'", limit=MAX_SUMMARY_CHARS)
        return _short_text(f"loaded {item_count} directed {kind} evidence item(s)", limit=MAX_SUMMARY_CHARS)
    return _short_text(f"loaded {item_count} {kind} evidence item(s) from {ref or 'artifact'}", limit=MAX_SUMMARY_CHARS)


def build_evidence_observation(loaded: Mapping[str, Any] | Any, *, event_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a compact observation from an ``evidence/loaded`` payload."""
    if not isinstance(loaded, Mapping):
        loaded = {"schema": EVIDENCE_OBSERVATION_ERROR_SCHEMA, "ok": False, "error": "loaded payload was not a mapping", "value": loaded}
    meta = dict(event_meta or {})
    ok = bool(loaded.get("ok", False))
    mode = str(loaded.get("mode", "summary") or "summary")
    artifact_ref = str(loaded.get("artifact_ref", "") or "")
    item_count = int(loaded.get("item_count", 0) or 0)
    trigger_topic = str(loaded.get("trigger_topic", "") or meta.get("trigger_topic", "") or "")

    seed = {
        "artifact_ref": artifact_ref,
        "request_id": loaded.get("request_id", ""),
        "trigger_topic": trigger_topic,
        "mode": mode,
        "query": loaded.get("query", ""),
        "ok": ok,
    }
    observation: dict[str, Any] = {
        "schema": EVIDENCE_OBSERVATION_SCHEMA,
        "observation_id": "evidence_obs:" + _stable_id(seed),
        "ok": ok,
        "status": EVIDENCE_STATUS,
        "raw_policy": RAW_POLICY,
        "artifact_ref": artifact_ref,
        "artifact_kind": str(loaded.get("artifact_kind", "") or ""),
        "mode": mode,
        "query": _short_text(loaded.get("query", ""), limit=MAX_SUMMARY_CHARS),
        "limit": int(loaded.get("limit", 0) or 0),
        "offset": int(loaded.get("offset", 0) or 0),
        "item_count": item_count,
        "scanned_count": int(loaded.get("scanned_count", 0) or 0),
        "byte_count": int(loaded.get("byte_count", 0) or 0),
        "matched_terms": list(loaded.get("matched_terms", []) or [])[:16] if isinstance(loaded.get("matched_terms", []), Sequence) else [],
        "items_sample": _items_sample(loaded.get("items", [])),
        "summary": _summarize_loaded(loaded, ok=ok, item_count=item_count, mode=mode, ref=artifact_ref),
        "request_id": str(loaded.get("request_id", "") or ""),
        "request_topic": str(loaded.get("request_topic", "") or ""),
        "requested_by": str(loaded.get("requested_by", "") or ""),
        "trigger_topic": trigger_topic,
        "trigger_source": str(loaded.get("trigger_source", "") or ""),
        "route_reason": str(loaded.get("route_reason", "") or ""),
        "priority": float(loaded.get("priority", 0.5) or 0.5),
        "confidence_hint": _confidence_hint(ok=ok, item_count=item_count, mode=mode),
    }
    if not ok and loaded.get("error"):
        observation["error"] = _short_text(loaded.get("error", ""), limit=MAX_SUMMARY_CHARS)
    if isinstance(loaded.get("ref_card"), Mapping):
        ref_card = dict(loaded.get("ref_card") or {})
        observation["ref_card"] = _bound_value(ref_card)
    return observation


def _confidence_hint(*, ok: bool, item_count: int, mode: str) -> float:
    if not ok:
        return 0.0
    if item_count <= 0:
        return 0.12
    base = 0.42
    if mode == "directed":
        base = 0.52
    elif mode == "walk":
        base = 0.48
    elif mode == "scatter":
        base = 0.28
    return min(0.82, base + min(0.18, item_count * 0.025))


def route_topic_for_observation(observation: Mapping[str, Any] | Any) -> str:
    if not isinstance(observation, Mapping):
        return ""
    explicit = str(observation.get("emit_observation_topic", "") or "").strip()
    if explicit:
        return explicit
    trigger = str(observation.get("trigger_topic", "") or "").strip()
    for prefix, topic in ROUTE_TOPIC_BY_TRIGGER_PREFIX:
        if trigger.startswith(prefix):
            return topic
    return ""
