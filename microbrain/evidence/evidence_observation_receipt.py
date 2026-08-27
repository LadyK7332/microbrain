from __future__ import annotations

"""Compact receipts for evidence observations.

Evidence observations are bounded windows opened from artifact/index ledgers.
This module creates tiny receipt rows that can be staged into memcells without
copying proof samples into normal memory.
"""

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

RECEIPT_MIN_PRIORITY = 0.55
SCATTER_RECEIPT_MIN_PRIORITY = 0.72
MAX_RECEIPT_TEXT_CHARS = 360
MAX_RECEIPT_QUERY_CHARS = 180
MAX_LINKS_INLINE = 8

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

EVIDENCE_OBSERVATION_RECEIPT_SCHEMA = "evidence.observation_receipt.v1"
MEMCELL_EVIDENCE_RECEIPT_SCHEMA = "mem_cell.evidence_observation_receipt.v1"
RECEIPT_STATUS = "bounded_sample_receipt_not_truth"
RAW_POLICY = "receipt_only_no_sample_items"

IMPORTANT_TRIGGER_PREFIXES = (
    "safety/",
    "hypothesis/",
    "review/",
    "trainer/",
    "scene/",
    "recognition/",
)

IMPORTANT_ROUTE_REASONS = frozenset(
    {
        "safety_uncertainty",
        "hypothesis_contradiction",
        "contradiction",
        "trainer_correction",
        "review_repair_candidate",
        "scene_anomaly",
        "recognition_conflict",
        "hazard_review",
    }
)

MEMCELL_KIND = "evidence.observation_receipt"


def _stable_id(value: Any, *, size: int = 8) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=size).hexdigest()


def _short_text(value: Any, *, limit: int = MAX_RECEIPT_TEXT_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _maybe_list(value: Any, *, limit: int = MAX_LINKS_INLINE) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [str(item) for item in value if item not in (None, "", [], {})]
    else:
        items = [str(value)]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = _short_text(item, limit=220)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _sample_digest(observation: Mapping[str, Any]) -> str:
    sample = observation.get("items_sample", [])
    if sample in (None, "", [], {}):
        return ""
    return "sample:" + _stable_id(sample, size=10)


def _summary_for(observation: Mapping[str, Any]) -> str:
    summary = _short_text(observation.get("summary", ""))
    if summary:
        return summary
    artifact_ref = _short_text(observation.get("artifact_ref", "artifact"), limit=160)
    item_count = _safe_int(observation.get("item_count", 0), 0)
    mode = _short_text(observation.get("mode", "summary"), limit=40)
    if item_count <= 0:
        return _short_text(f"evidence observation opened no items from {artifact_ref}")
    return _short_text(f"evidence observation loaded {item_count} item(s) from {artifact_ref} using {mode}")


def should_stage_observation_receipt(observation: Mapping[str, Any] | Any) -> bool:
    """Return True when an evidence observation deserves a tiny memory receipt."""
    if not isinstance(observation, Mapping):
        return False
    artifact_ref = str(observation.get("artifact_ref", "") or "").strip()
    if not artifact_ref:
        return False

    priority = _safe_float(observation.get("priority", 0.5), 0.5)
    mode = str(observation.get("mode", "summary") or "summary").strip().lower()
    trigger_topic = str(observation.get("trigger_topic", "") or "").strip()
    route_reason = str(observation.get("route_reason", "") or "").strip()
    item_count = _safe_int(observation.get("item_count", 0), 0)

    if route_reason in IMPORTANT_ROUTE_REASONS:
        return True
    if any(trigger_topic.startswith(prefix) for prefix in IMPORTANT_TRIGGER_PREFIXES):
        return True
    if mode in {"scatter", "shotgun"}:
        return priority >= SCATTER_RECEIPT_MIN_PRIORITY and item_count > 0
    return priority >= RECEIPT_MIN_PRIORITY or item_count > 0


def tier_for_observation_receipt(observation: Mapping[str, Any] | Any) -> str:
    """Choose a memory tier for a compact receipt, not for raw evidence."""
    if not isinstance(observation, Mapping):
        return "now"
    priority = _safe_float(observation.get("priority", 0.5), 0.5)
    trigger_topic = str(observation.get("trigger_topic", "") or "")
    route_reason = str(observation.get("route_reason", "") or "")
    if trigger_topic.startswith("safety/") or route_reason == "safety_uncertainty":
        return "short"
    if route_reason in IMPORTANT_ROUTE_REASONS:
        return "short"
    if priority >= 0.84:
        return "short"
    return "now"


def build_evidence_observation_receipt(
    observation: Mapping[str, Any] | Any,
    *,
    event_meta: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Build a compact receipt from an evidence observation.

    The receipt intentionally stores only summaries, counts, references, and a
    digest of the bounded sample.  It never carries ``items_sample``.
    """
    if not isinstance(observation, Mapping):
        observation = {"ok": False, "error": "observation payload was not a mapping", "value_type": type(observation).__name__}
    meta = dict(event_meta or {})
    now_ts = time.time() if now is None else float(now)

    artifact_ref = str(observation.get("artifact_ref", "") or "")
    observation_id = str(observation.get("observation_id", "") or "")
    request_id = str(observation.get("request_id", "") or "")
    trigger_topic = str(observation.get("trigger_topic", "") or meta.get("trigger_topic", "") or "")
    route_reason = str(observation.get("route_reason", "") or meta.get("route_reason", "") or "")
    mode = str(observation.get("mode", "summary") or "summary")
    priority = _safe_float(observation.get("priority", meta.get("priority", 0.5)), 0.5)
    item_count = _safe_int(observation.get("item_count", 0), 0)
    scanned_count = _safe_int(observation.get("scanned_count", 0), 0)
    byte_count = _safe_int(observation.get("byte_count", 0), 0)
    confidence_hint = _safe_float(observation.get("confidence_hint", 0.0), 0.0)

    seed = {
        "artifact_ref": artifact_ref,
        "observation_id": observation_id,
        "request_id": request_id,
        "trigger_topic": trigger_topic,
        "mode": mode,
        "route_reason": route_reason,
    }
    receipt_id = "evidence_receipt:" + _stable_id(seed, size=10)
    sample_digest = _sample_digest(observation)
    tier = tier_for_observation_receipt(observation)
    should_stage = should_stage_observation_receipt(observation)

    refs = {
        "artifact_ref": artifact_ref,
        "observation_id": observation_id,
        "request_id": request_id,
    }
    refs = {key: value for key, value in refs.items() if value}
    if isinstance(observation.get("ref_card"), Mapping):
        ref_card = observation.get("ref_card") or {}
        for key in ("artifact_ref", "index_ref", "ledger_ref"):
            if ref_card.get(key) and key not in refs:
                refs[key] = str(ref_card.get(key))

    return {
        "schema": EVIDENCE_OBSERVATION_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "created_at": now_ts,
        "status": RECEIPT_STATUS,
        "raw_policy": RAW_POLICY,
        "ok": bool(observation.get("ok", False)),
        "summary": _summary_for(observation),
        "artifact_ref": artifact_ref,
        "artifact_kind": _short_text(observation.get("artifact_kind", ""), limit=80),
        "observation_id": observation_id,
        "request_id": request_id,
        "request_topic": _short_text(observation.get("request_topic", ""), limit=120),
        "requested_by": _short_text(observation.get("requested_by", ""), limit=120),
        "trigger_topic": trigger_topic,
        "trigger_source": _short_text(observation.get("trigger_source", ""), limit=120),
        "route_reason": route_reason,
        "mode": mode,
        "query": _short_text(observation.get("query", ""), limit=MAX_RECEIPT_QUERY_CHARS),
        "item_count": item_count,
        "scanned_count": scanned_count,
        "byte_count": byte_count,
        "sample_digest": sample_digest,
        "sample_item_count_seen": len(observation.get("items_sample", []) or []) if isinstance(observation.get("items_sample", []), Sequence) else 0,
        "matched_terms": _maybe_list(observation.get("matched_terms", []), limit=16),
        "priority": priority,
        "confidence_hint": confidence_hint,
        "refs": refs,
        "stage_decision": {
            "stage": should_stage,
            "tier": tier,
            "reason": _stage_reason(observation, should_stage=should_stage, tier=tier),
        },
    }


def _stage_reason(observation: Mapping[str, Any], *, should_stage: bool, tier: str) -> str:
    if not should_stage:
        return "receipt skipped: no artifact ref or priority below receipt threshold"
    trigger_topic = str(observation.get("trigger_topic", "") or "")
    route_reason = str(observation.get("route_reason", "") or "")
    if trigger_topic.startswith("safety/") or route_reason == "safety_uncertainty":
        return f"{tier} receipt: safety proof-demand observation"
    if route_reason in IMPORTANT_ROUTE_REASONS:
        return f"{tier} receipt: important proof-demand route"
    return f"{tier} receipt: bounded evidence sample opened"


def build_memcell_for_evidence_receipt(receipt: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Create a tiny memcell row from an evidence observation receipt."""
    if not isinstance(receipt, Mapping):
        receipt = build_evidence_observation_receipt(receipt)
    links: list[str] = []
    refs = receipt.get("refs", {})
    if isinstance(refs, Mapping):
        for value in refs.values():
            links.extend(_maybe_list(value, limit=MAX_LINKS_INLINE))
    links = _maybe_list(links, limit=MAX_LINKS_INLINE)

    text = _short_text(receipt.get("summary", ""), limit=MAX_RECEIPT_TEXT_CHARS)
    activation = min(1.0, max(0.05, 0.25 + _safe_float(receipt.get("priority", 0.5), 0.5) * 0.45))
    trust = min(1.0, max(0.05, _safe_float(receipt.get("confidence_hint", 0.0), 0.0)))

    return {
        "schema": MEMCELL_EVIDENCE_RECEIPT_SCHEMA,
        "id": str(receipt.get("receipt_id", "") or ("evidence_receipt:" + _stable_id(receipt, size=10))),
        "kind": MEMCELL_KIND,
        "text": text,
        "summary": text,
        "created_at": float(receipt.get("created_at", time.time()) or time.time()),
        "activation": activation,
        "trust": trust,
        "priority": _safe_float(receipt.get("priority", 0.5), 0.5),
        "evidence_status": RECEIPT_STATUS,
        "raw_policy": RAW_POLICY,
        "artifact_ref": str(receipt.get("artifact_ref", "") or ""),
        "observation_id": str(receipt.get("observation_id", "") or ""),
        "request_id": str(receipt.get("request_id", "") or ""),
        "trigger_topic": str(receipt.get("trigger_topic", "") or ""),
        "route_reason": str(receipt.get("route_reason", "") or ""),
        "mode": str(receipt.get("mode", "summary") or "summary"),
        "item_count": _safe_int(receipt.get("item_count", 0), 0),
        "sample_digest": str(receipt.get("sample_digest", "") or ""),
        "links_explicit": links,
        "evidence_receipt": {
            "receipt_id": str(receipt.get("receipt_id", "") or ""),
            "schema": EVIDENCE_OBSERVATION_RECEIPT_SCHEMA,
            "refs": dict(receipt.get("refs", {}) or {}) if isinstance(receipt.get("refs", {}), Mapping) else {},
            "sample_digest": str(receipt.get("sample_digest", "") or ""),
            "item_count": _safe_int(receipt.get("item_count", 0), 0),
            "status": RECEIPT_STATUS,
        },
    }
