from __future__ import annotations

"""Route proof-demand events into bounded evidence-load requests.

The evidence loader is intentionally passive: it opens artifacts only when a
reasoning/review path asks.  This module is the small bridge between events that
mean "check the proof" and the concrete ``memory/evidence_request`` messages the
loader understands.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Keep router output bounded.  One contradiction can contain many refs, but the
# loader should not get a flood of requests from a single turn.
MAX_ROUTED_REQUESTS_PER_EVENT = 4
MAX_REF_SCAN_DEPTH = 5
MAX_REF_SCAN_ITEMS = 96
MAX_QUERY_CHARS = 360
MAX_QUERY_TOKENS = 24

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

EVIDENCE_REQUEST_ROUTE_SCHEMA = "evidence.request_route.v1"
EVIDENCE_REQUEST_SCHEMA = "evidence.request.v1"
TOKEN_RE = re.compile(r"[a-z0-9']+")

# Topics that imply a proof pile may need to be opened if their payload carries
# an evidence/artifact ref.  Mode names match microbrain.evidence.evidence_loader.
TRIGGER_RULES: dict[str, dict[str, Any]] = {
    "trainer/correction": {
        "mode": "walk",
        "limit": 12,
        "reason": "trainer_correction_needs_ordered_evidence",
        "priority": 1.0,
    },
    "safety/uncertain_action": {
        "mode": "directed",
        "limit": 10,
        "reason": "safety_uncertainty_needs_targeted_evidence",
        "priority": 1.0,
    },
    "hypothesis/contradiction": {
        "mode": "walk",
        "limit": 10,
        "reason": "hypothesis_contradiction_needs_ordered_evidence",
        "priority": 0.92,
    },
    "hypothesis/conflict": {
        "mode": "walk",
        "limit": 10,
        "reason": "hypothesis_conflict_needs_ordered_evidence",
        "priority": 0.9,
    },
    "recognition/conflict": {
        "mode": "directed",
        "limit": 8,
        "reason": "recognition_conflict_needs_targeted_evidence",
        "priority": 0.86,
    },
    "scene/anomaly": {
        "mode": "walk",
        "limit": 8,
        "reason": "scene_anomaly_needs_nearby_evidence",
        "priority": 0.78,
    },
    "review/repair_candidate": {
        "mode": "directed",
        "limit": 6,
        "reason": "review_repair_candidate_needs_supporting_evidence",
        "priority": 0.72,
    },
    "object/evidence_challenge": {
        "mode": "directed",
        "limit": 10,
        "reason": "object_evidence_challenge",
        "priority": 0.88,
    },
    "memory/evidence_needed": {
        "mode": "directed",
        "limit": 8,
        "reason": "memory_path_requested_evidence",
        "priority": 0.8,
    },
    # Low-urgency curiosity can sample an existing ledger/index without turning
    # every idle thought into a full search.  It only fires when refs are present.
    "thought/probe": {
        "mode": "scatter",
        "limit": 4,
        "reason": "idle_probe_samples_existing_evidence",
        "priority": 0.35,
    },
}

REF_KEYS = frozenset({
    "artifact_ref",
    "artifact_refs",
    "evidence_ref",
    "evidence_refs",
    "index_ref",
    "index_refs",
    "ledger_ref",
    "ledger_refs",
    "data_ref",
    "source_ref",
    "source_refs",
})

REF_PREFIXES = (
    "evidence/",
    "mem_cell_links/",
    "mem_cell/",
    "mem_cell_derived/",
)

QUERY_KEYS = (
    "query",
    "text",
    "summary",
    "description",
    "reason",
    "claim",
    "claims_supported",
    "label",
    "object_label",
    "interpretation",
    "repair_surface",
    "expected",
    "observed",
)


def normalize_trigger_topic(topic: Any) -> str:
    return str(topic or "").strip()


def trigger_rule_for(topic: Any) -> dict[str, Any] | None:
    topic_s = normalize_trigger_topic(topic)
    rule = TRIGGER_RULES.get(topic_s)
    return dict(rule) if rule else None


def _short_text(value: Any, *, limit: int = MAX_QUERY_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _stable_id(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _looks_like_artifact_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().replace("\\", "/")
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in REF_PREFIXES):
        return True
    # Absolute refs are allowed later by the loader only if they resolve under
    # the memory directory.  The router accepts obvious artifact-ish file refs.
    lower = text.lower()
    return lower.endswith((".jsonl", ".json", ".txt", ".md", ".log", ".csv")) and "/" in text


def _coerce_ref_card(value: Any, *, key_hint: str = "") -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        for key in ("artifact_ref", "index_ref", "ledger_ref", "data_ref", "source_ref", "path", "ref"):
            ref = value.get(key)
            if _looks_like_artifact_ref(ref):
                card = dict(value)
                card["artifact_ref"] = str(ref).strip().replace("\\", "/")
                card.setdefault("ref_source_key", key_hint or key)
                return card
        nested = value.get("evidence_ref") or value.get("card") or value.get("index_card")
        if nested is not value:
            return _coerce_ref_card(nested, key_hint=key_hint or "evidence_ref")
        return None
    if _looks_like_artifact_ref(value):
        return {"artifact_ref": str(value).strip().replace("\\", "/"), "ref_source_key": key_hint or "value"}
    return None


def extract_evidence_refs(value: Any, *, max_refs: int = MAX_ROUTED_REQUESTS_PER_EVENT, max_depth: int = MAX_REF_SCAN_DEPTH) -> list[dict[str, Any]]:
    """Return compact artifact/index/ledger cards found inside a payload.

    This is intentionally conservative.  Generic cell ids are not enough; the
    router only opens things that look like memory artifact paths or explicit
    evidence cards.
    """
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0

    def add(card: dict[str, Any] | None) -> None:
        if not card:
            return
        ref = str(card.get("artifact_ref", "") or "").strip().replace("\\", "/")
        if not ref or ref in seen:
            return
        seen.add(ref)
        refs.append(card)

    def walk(node: Any, *, depth: int, key_hint: str = "") -> None:
        nonlocal scanned
        if len(refs) >= max_refs or depth > max_depth or scanned >= MAX_REF_SCAN_ITEMS:
            return
        scanned += 1
        if isinstance(node, Mapping):
            # First treat this mapping as a possible evidence card.
            add(_coerce_ref_card(node, key_hint=key_hint))
            if len(refs) >= max_refs:
                return
            for key, child in node.items():
                key_s = str(key)
                if key_s in REF_KEYS:
                    if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                        for item in list(child)[:max_refs]:
                            add(_coerce_ref_card(item, key_hint=key_s))
                            if len(refs) >= max_refs:
                                return
                    else:
                        add(_coerce_ref_card(child, key_hint=key_s))
                    if len(refs) >= max_refs:
                        return
                walk(child, depth=depth + 1, key_hint=key_s)
                if len(refs) >= max_refs:
                    return
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in list(node)[:MAX_REF_SCAN_ITEMS]:
                walk(child, depth=depth + 1, key_hint=key_hint)
                if len(refs) >= max_refs:
                    return
        else:
            # Only accept bare strings when they are under an explicit ref-ish key.
            if key_hint in REF_KEYS or key_hint in {"ref", "path"}:
                add(_coerce_ref_card(node, key_hint=key_hint))

    walk(value, depth=0)
    return refs


def _query_parts(value: Any, *, depth: int = 0, parts: list[str] | None = None) -> list[str]:
    if parts is None:
        parts = []
    if depth > 4 or len(parts) >= MAX_QUERY_TOKENS:
        return parts
    if isinstance(value, Mapping):
        for key in QUERY_KEYS:
            if key in value and value.get(key) not in (None, "", [], {}):
                item = value.get(key)
                if isinstance(item, (list, tuple)):
                    text = " ".join(str(x) for x in item[:8])
                elif isinstance(item, Mapping):
                    text = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                else:
                    text = str(item)
                parts.append(_short_text(text, limit=120))
                if len(parts) >= MAX_QUERY_TOKENS:
                    return parts
        for child in value.values():
            _query_parts(child, depth=depth + 1, parts=parts)
            if len(parts) >= MAX_QUERY_TOKENS:
                break
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in list(value)[:16]:
            _query_parts(child, depth=depth + 1, parts=parts)
            if len(parts) >= MAX_QUERY_TOKENS:
                break
    return parts


def build_query_text(payload: Any, *, fallback: str = "") -> str:
    parts = [part for part in _query_parts(payload) if part]
    if not parts and fallback:
        parts = [fallback]
    raw = " ".join(parts)
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(raw.lower()):
        if len(tok) < 2:
            continue
        if tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= MAX_QUERY_TOKENS:
            break
    if tokens:
        return _short_text(" ".join(tokens), limit=MAX_QUERY_CHARS)
    return _short_text(raw or fallback, limit=MAX_QUERY_CHARS)


def route_evidence_requests(
    topic: Any,
    payload: Any,
    *,
    source: str = "",
    event_meta: Mapping[str, Any] | None = None,
    correlation_id: str = "",
) -> dict[str, Any]:
    """Build loader requests for a trigger event.

    Returns a route envelope with zero or more request payloads.  The caller can
    emit each request on ``memory/evidence_request``.
    """
    topic_s = normalize_trigger_topic(topic)
    rule = trigger_rule_for(topic_s)
    if not rule:
        return {
            "schema": EVIDENCE_REQUEST_ROUTE_SCHEMA,
            "routed": False,
            "trigger_topic": topic_s,
            "requests": [],
            "reason": "topic_not_registered_for_evidence_routing",
        }

    refs = extract_evidence_refs(payload, max_refs=MAX_ROUTED_REQUESTS_PER_EVENT)
    if not refs:
        return {
            "schema": EVIDENCE_REQUEST_ROUTE_SCHEMA,
            "routed": False,
            "trigger_topic": topic_s,
            "requests": [],
            "reason": "no_evidence_refs_found",
            "route_reason": rule.get("reason", ""),
        }

    meta = dict(event_meta or {})
    query = build_query_text(payload, fallback=str(meta.get("reason", "") or rule.get("reason", "")))
    mode = str(meta.get("evidence_mode") or rule.get("mode") or "summary")
    limit = int(meta.get("evidence_limit") or rule.get("limit") or 8)
    requests: list[dict[str, Any]] = []
    for ordinal, ref_card in enumerate(refs):
        artifact_ref = str(ref_card.get("artifact_ref", "") or "").strip().replace("\\", "/")
        if not artifact_ref:
            continue
        request_seed = {
            "trigger_topic": topic_s,
            "source": source,
            "correlation_id": correlation_id,
            "artifact_ref": artifact_ref,
            "ordinal": ordinal,
        }
        request = {
            "schema": EVIDENCE_REQUEST_SCHEMA,
            "request_id": "evidence_req:" + _stable_id(request_seed),
            "artifact_ref": artifact_ref,
            "mode": mode,
            "limit": limit,
            "query": query,
            "trigger_topic": topic_s,
            "trigger_source": source,
            "route_reason": str(rule.get("reason", "") or "evidence_trigger"),
            "priority": float(rule.get("priority", 0.5) or 0.5),
            "ref_card": ref_card,
            "emit_topic": "evidence/loaded",
        }
        if mode == "scatter":
            request["seed"] = _stable_id(request_seed)
        requests.append(request)

    return {
        "schema": EVIDENCE_REQUEST_ROUTE_SCHEMA,
        "routed": bool(requests),
        "trigger_topic": topic_s,
        "route_reason": str(rule.get("reason", "") or "evidence_trigger"),
        "mode": mode,
        "limit": limit,
        "query": query,
        "request_count": len(requests),
        "requests": requests,
    }
