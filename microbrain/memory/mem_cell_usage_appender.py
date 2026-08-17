from __future__ import annotations

"""Temporary usage appendices for mem-cells.

Words, patterns, and sentence frames are stable-ish memory cells. A single use of
one cell inside one sentence is not stable truth; it is temporary evidence about
how that cell behaved in context.  This module appends small, expiring usage
atoms to the cells created by ``MemCellStore.ingest_text`` so later composers and
language organs can see *why* a word/frame was used without permanently
rebuilding the concept on the first encounter.
"""

import hashlib
import json
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

USAGE_APPENDIX_TTL_S = 3.0 * 60.0 * 60.0
USAGE_APPENDIX_LIMIT = 16
USAGE_APPENDIX_SAMPLE_LIMIT = 180

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

USAGE_APPENDIX_SCHEMA = "mem_cell.usage_appendix.v1"
USAGE_APPENDIX_KEY = "usage_appendix"
USAGE_APPENDIX_STATE_KEY = "usage_appendix_state"
USAGE_APPENDIX_SUMMARY_KEY = "usage_appendix_summary"

SHORT_RESPONSE_TERMS = {
    "short",
    "quick",
    "brief",
    "terse",
    "concise",
    "yesno",
    "yes/no",
    "oneword",
    "one-word",
}

EXPLANATION_TERMS = {
    "why",
    "how",
    "explain",
    "explanation",
    "think",
    "thought",
    "opinion",
    "feedback",
    "review",
    "because",
}

APPROVAL_TERMS = {"approve", "approval", "agree", "okay", "ok", "valid", "good"}


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split()).strip()


def _safe_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _fingerprint(payload: Mapping[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(material.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()


def _token_ref(cell: Mapping[str, Any]) -> str:
    anchor = _safe_dict(cell.get("anchor"))
    return str(anchor.get("ref") or "").strip().lower()


def _slot_refs(cell: Mapping[str, Any]) -> dict[str, Any]:
    refs = _safe_list(cell.get("refs"))
    slots: dict[str, Any] = {}
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        if str(ref.get("kind") or "") != "slot":
            continue
        name = str(ref.get("name") or "").strip()
        if not name:
            continue
        slots[name] = ref.get("value")
    meta = _safe_dict(cell.get("meta"))
    meta_slots = _safe_dict(meta.get("slots"))
    for key, value in meta_slots.items():
        if value not in (None, "", [], False):
            slots.setdefault(str(key), value)
    return slots


def classify_response_request_slots(text: str) -> dict[str, Any]:
    """Return expected response slots implied by the user's wording.

    This is intentionally simple and trace-focused.  It does not compose the
    final answer; it tags the utterance so later response logic can avoid
    collapsing compound requests into a single word.
    """
    raw = str(text or "").strip()
    norm = _norm_text(raw.replace("?", " ? ").replace("/", " / "))
    tokens = [t for t in norm.replace("?", " ? ").split() if t]
    token_set = set(tokens)
    slots: list[dict[str, Any]] = []

    asks_approval = bool(token_set & APPROVAL_TERMS) or "do you approve" in norm
    asks_explanation = bool(token_set & EXPLANATION_TERMS) or "what do you think" in norm
    asks_short = bool(token_set & SHORT_RESPONSE_TERMS)
    is_question = "?" in raw or (tokens[:1] and tokens[0] in {"what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does", "did", "is", "are"})

    if asks_approval:
        slots.append({
            "slot": "approval_judgment",
            "expected_shape": "yes_no_or_conditional",
            "bare_word_allowed": bool(asks_short and not asks_explanation),
        })
    if asks_explanation:
        slots.append({
            "slot": "evaluation_explanation",
            "expected_shape": "reasoning_or_caveat_frame",
            "bare_word_allowed": False,
        })
    if is_question and not slots:
        slots.append({
            "slot": "answer_with_reason",
            "expected_shape": "answer_plus_explanation_unless_short_typified",
            "bare_word_allowed": bool(asks_short),
        })
    if not slots:
        slots.append({
            "slot": "contextual_reply",
            "expected_shape": "explanatory_by_default",
            "bare_word_allowed": bool(asks_short),
        })

    return {
        "schema": "language.response_request_slots.v1",
        "slots": slots[:4],
        "short_typified": bool(asks_short),
        "explanation_expected": any(not bool(slot.get("bare_word_allowed")) for slot in slots),
        "compound_request": len(slots) > 1,
    }


def _source_confidence(*, role: str, topic: str, source: str, meta: Mapping[str, Any] | None = None) -> float:
    meta = _safe_dict(meta)
    raw_lane = str(meta.get("lane") or meta.get("mode") or meta.get("source_mode") or "").lower()
    topic_l = str(topic or "").lower()
    source_l = str(source or "").lower()
    if raw_lane in {"trainer", "teaching", "/t", "t"} or topic_l.startswith("trainer/") or "trainer" in source_l:
        return 0.92
    if role == "user":
        return 0.78
    if role == "assistant":
        return 0.56
    return 0.50


def make_usage_atom(
    *,
    cell: Mapping[str, Any],
    text: str,
    role: str,
    topic: str,
    source: str,
    parent_id: str = "",
    token_index: int | None = None,
    left_context: str = "",
    right_context: str = "",
    structure: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    kind = str(cell.get("kind") or "unknown")
    anchor = _safe_dict(cell.get("anchor"))
    cell_meta = _safe_dict(cell.get("meta"))
    slots = _slot_refs(cell)
    usage_payload = {
        "schema": USAGE_APPENDIX_SCHEMA,
        "cell_kind": kind,
        "anchor_kind": str(anchor.get("kind") or ""),
        "anchor_ref": str(anchor.get("ref") or "")[:120],
        "role": str(role or ""),
        "topic": str(topic or ""),
        "source": str(source or ""),
        "parent_id": str(parent_id or cell_meta.get("parent_id") or ""),
        "token_index": token_index if token_index is not None else cell_meta.get("token_index"),
        "left_context": str(left_context or "")[:64],
        "right_context": str(right_context or "")[:64],
        "functional_role": str(cell_meta.get("functional_role") or ""),
        "tool_role": str(cell_meta.get("tool_role") or ""),
        "pattern_type": str(cell_meta.get("pattern_type") or ""),
        "slots": slots,
        "response_request": classify_response_request_slots(text) if kind == "utterance_anchor" else {},
        "surface_sample": str(text or "")[:USAGE_APPENDIX_SAMPLE_LIMIT],
        "epistemic_status": "temporary_usage_evidence",
        "source_confidence": _source_confidence(role=role, topic=topic, source=source, meta=meta),
        "created_ts": now,
        "expires_ts": now + USAGE_APPENDIX_TTL_S,
    }
    usage_payload["id"] = "ua" + _fingerprint({
        "cell_id": str(cell.get("id") or ""),
        "anchor": usage_payload["anchor_ref"],
        "role": usage_payload["role"],
        "topic": usage_payload["topic"],
        "token_index": usage_payload.get("token_index"),
        "left": usage_payload["left_context"],
        "right": usage_payload["right_context"],
        "surface": usage_payload["surface_sample"],
    })
    return usage_payload


def merge_usage_appendices(
    old_items: Sequence[Mapping[str, Any]] | None,
    new_items: Sequence[Mapping[str, Any]] | None,
    *,
    now_ts: float | None = None,
    ttl_s: float = USAGE_APPENDIX_TTL_S,
    limit: int = USAGE_APPENDIX_LIMIT,
) -> list[dict[str, Any]]:
    """Merge old/new temporary usage atoms by ID, prune expired, cap recent."""
    now = float(now_ts if now_ts is not None else time.time())
    floor = now - max(60.0, float(ttl_s))
    by_id: dict[str, dict[str, Any]] = {}
    for item in list(old_items or []) + list(new_items or []):
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        created = float(row.get("created_ts", row.get("ts", now)) or now)
        expires = float(row.get("expires_ts", created + ttl_s) or created + ttl_s)
        if expires < now or created < floor:
            continue
        row.setdefault("schema", USAGE_APPENDIX_SCHEMA)
        row.setdefault("created_ts", created)
        row.setdefault("expires_ts", expires)
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            row_id = "ua" + _fingerprint(row)
            row["id"] = row_id
        prior = by_id.get(row_id)
        if prior is None or float(prior.get("created_ts", 0.0) or 0.0) <= created:
            by_id[row_id] = row
    rows = sorted(by_id.values(), key=lambda item: float(item.get("created_ts", 0.0) or 0.0), reverse=True)
    return rows[: max(1, int(limit))]


def usage_appendix_summary(items: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    rows = [dict(item) for item in list(items or []) if isinstance(item, Mapping)]
    kinds: dict[str, int] = {}
    roles: dict[str, int] = {}
    response_slots: set[str] = set()
    for item in rows:
        kind = str(item.get("cell_kind") or "")
        role = str(item.get("role") or "")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
        if role:
            roles[role] = roles.get(role, 0) + 1
        rr = item.get("response_request") if isinstance(item.get("response_request"), Mapping) else {}
        for slot in _safe_list(rr.get("slots")):
            if isinstance(slot, Mapping) and str(slot.get("slot") or ""):
                response_slots.add(str(slot.get("slot")))
    return {
        "schema": "mem_cell.usage_appendix_summary.v1",
        "count": len(rows),
        "kinds": kinds,
        "roles": roles,
        "response_slots": sorted(response_slots),
        "epistemic_status": "temporary_usage_summary",
    }


def merge_meta_with_usage_appendix(
    old_meta: Mapping[str, Any] | None,
    new_meta: Mapping[str, Any] | None,
    *,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Merge normal metadata while appending temporary usage evidence."""
    old = _safe_dict(old_meta)
    new = _safe_dict(new_meta)
    old_items = _safe_list(old.get(USAGE_APPENDIX_KEY))
    new_items = _safe_list(new.get(USAGE_APPENDIX_KEY))
    merged = dict(old)
    merged.update({k: v for k, v in new.items() if k not in {USAGE_APPENDIX_KEY, USAGE_APPENDIX_STATE_KEY, USAGE_APPENDIX_SUMMARY_KEY}})
    items = merge_usage_appendices(old_items, new_items, now_ts=now_ts)
    if items:
        merged[USAGE_APPENDIX_KEY] = items
        merged[USAGE_APPENDIX_STATE_KEY] = {
            "schema": "mem_cell.usage_appendix_state.v1",
            "state": "temporary_pending_generalization",
            "promotes_directly_to_truth": False,
            "requires_repetition_or_trainer_confidence": True,
        }
        merged[USAGE_APPENDIX_SUMMARY_KEY] = usage_appendix_summary(items)
    else:
        merged.pop(USAGE_APPENDIX_KEY, None)
        merged.pop(USAGE_APPENDIX_STATE_KEY, None)
        merged.pop(USAGE_APPENDIX_SUMMARY_KEY, None)
    return merged


def attach_temporary_usage_appendix(
    *,
    utterance: Mapping[str, Any],
    token_cells: Sequence[Mapping[str, Any]],
    word_role_cells: Sequence[Mapping[str, Any]],
    thought_template_cells: Sequence[Mapping[str, Any]],
    clause_frame_cells: Sequence[Mapping[str, Any]],
    learning_frame_cells: Sequence[Mapping[str, Any]],
    general_pattern_cells: Sequence[Mapping[str, Any]],
    linker_cells: Sequence[Mapping[str, Any]],
    text: str,
    role: str,
    topic: str,
    source: str,
    structure: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return copies of cells with a temporary usage atom appended.

    This does not mutate canonical meaning.  It records how each cell was used in
    this sentence so later stable patterns can self-update by repetition.
    """
    tokens = [_token_ref(cell) for cell in token_cells]
    parent_id = str(utterance.get("id") or "")
    now = time.time()
    groups: list[Mapping[str, Any]] = []
    groups.append(utterance)
    for seq in (
        token_cells,
        word_role_cells,
        thought_template_cells,
        clause_frame_cells,
        learning_frame_cells,
        general_pattern_cells,
        linker_cells,
    ):
        groups.extend([cell for cell in seq if isinstance(cell, Mapping)])

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for cell in groups:
        cell_id = str(cell.get("id") or "").strip()
        if not cell_id or cell_id in seen_ids:
            continue
        seen_ids.add(cell_id)
        cell_meta = _safe_dict(cell.get("meta"))
        raw_index = cell_meta.get("token_index")
        token_index = int(raw_index) if isinstance(raw_index, int) else None
        left = ""
        right = ""
        if token_index is not None and tokens:
            left = tokens[token_index - 1] if token_index > 0 and token_index - 1 < len(tokens) else ""
            right = tokens[token_index + 1] if token_index + 1 < len(tokens) else ""
        atom = make_usage_atom(
            cell=cell,
            text=text,
            role=role,
            topic=topic,
            source=source,
            parent_id=parent_id,
            token_index=token_index,
            left_context=left,
            right_context=right,
            structure=structure,
            meta=meta,
            now_ts=now,
        )
        updated = dict(cell)
        updated_meta = merge_meta_with_usage_appendix(cell.get("meta", {}), {USAGE_APPENDIX_KEY: [atom]}, now_ts=now)
        updated["meta"] = updated_meta
        updated["usage_count"] = max(int(updated.get("usage_count", 0) or 0), int(cell.get("usage_count", 0) or 0)) + 1
        updated["last_used_ts"] = max(float(updated.get("last_used_ts", 0.0) or 0.0), now)
        out.append(updated)
    return out
