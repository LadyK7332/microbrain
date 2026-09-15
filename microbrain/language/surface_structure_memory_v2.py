from __future__ import annotations

"""Safe, durable surface-shape memory for explicit response obligations."""

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from typing import Any

from microbrain.language.response_obligation import outward_target_surface

STORE_KV_KEY = "language:surface_structures_v2"
LAST_PLAN_KV_KEY = "language:last_surface_plan_v2"
LAST_CANDIDATE_KV_KEY = "language:last_surface_candidate_v2"
TARGET_TOKEN = "{target}"
SIGNAL_TOKEN = "{signal}"
SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
VOBJ_RE = re.compile(r"\bvobj:[a-z0-9_.:-]+\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
MAX_STRUCTURES = 256
MAX_CANDIDATES_PER_TEXT = 8
SURFACE_FLOOR_MAX_LEN = 120


def _clean(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[:limit]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if number != number else number


def _stable(value: Any, prefix: str) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(value)
    return f"{prefix}:{hashlib.blake2b(raw.encode('utf-8', errors='replace'), digest_size=8).hexdigest()}"


def _tokens(value: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(value or ""))]


def infer_surface_pattern(surface: Any, kind: Any = "", slots: Mapping[str, Any] | None = None) -> str:
    text = _clean(surface)
    if not text:
        return ""
    if SLOT_RE.search(text):
        return text
    if VOBJ_RE.search(text):
        return VOBJ_RE.sub(TARGET_TOKEN, text, count=1)
    words = _tokens(text)
    for demonstrative in ("that", "this"):
        if demonstrative in words:
            return re.sub(rf"\b{demonstrative}\b", TARGET_TOKEN, text, count=1, flags=re.IGNORECASE)
    if isinstance(slots, Mapping) and "target" in slots:
        for pronoun in ("it", "there"):
            if pronoun in words:
                return re.sub(rf"\b{pronoun}\b", TARGET_TOKEN, text, count=1, flags=re.IGNORECASE)
    kind_text = _clean(kind, 80).lower()
    if kind_text in {"unknown_identity_question", "question_structure"} and text.lower().startswith("what is "):
        return f"What is {TARGET_TOKEN}{'?' if text.endswith('?') else ''}"
    return text


def normalize_structure_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None
    example = _clean(
        candidate.get("surface_example")
        or candidate.get("surface_pattern")
        or candidate.get("quote")
        or candidate.get("text")
    )
    kind = _clean(candidate.get("structure_kind") or candidate.get("kind") or "surface_structure", 80).lower()
    slots = dict(candidate.get("slots") or {}) if isinstance(candidate.get("slots"), Mapping) else {}
    pattern = _clean(candidate.get("surface_pattern") or infer_surface_pattern(example, kind, slots), 180)
    if not pattern:
        return None
    required_slots = sorted(set(SLOT_RE.findall(pattern)))
    if "target" in required_slots and "target" not in slots:
        slots["target"] = {"role": "current_obligation_target"}
    structure_id = str(candidate.get("structure_id") or _stable([kind, pattern], "lstruct2"))
    return {
        "schema": "language.surface_structure.v2",
        "structure_id": structure_id,
        "structure_kind": kind,
        "speech_act": _clean(candidate.get("speech_act") or "clarify", 40).lower(),
        "surface_pattern": pattern,
        "surface_example": example,
        "slots": slots,
        "required_slots": required_slots,
        "allow_slotless_reuse": bool(candidate.get("allow_slotless_reuse", False)),
        "source": _clean(candidate.get("source") or "surface_candidate", 96),
        "evidence_refs": [
            str(value)
            for value in (
                candidate.get("evidence_ref"),
                candidate.get("source_quote_id"),
                candidate.get("cell_id"),
            )
            if value
        ][:8],
        "confidence": max(0.0, min(1.0, _float(candidate.get("confidence"), 0.52))),
        "created_ts": _float(candidate.get("created_ts"), time.time()),
        "last_used_ts": _float(candidate.get("last_used_ts"), 0.0),
        "use_count": int(candidate.get("use_count", 0) or 0),
        "truth_status": "structure_shape_not_answer_truth",
        "not_canned_response": True,
    }


def structure_candidates_from_text(
    text: Any,
    *,
    source: Any,
    evidence_ref: Any = "",
    confidence: float = 0.46,
) -> list[dict[str, Any]]:
    raw = str(text or "")
    if not raw.strip() or len(raw) > 8000:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for piece in re.split(r"(?<=[.!?])\s+|[\r\n]+", raw):
        surface = _clean(piece)
        if len(surface) < 4 or any(marker in surface for marker in ("```", "<html", "</", "def ", "class ", "=>")):
            continue
        pattern = infer_surface_pattern(surface, "question_structure")
        if TARGET_TOKEN not in pattern or pattern.lower() in seen:
            continue
        seen.add(pattern.lower())
        kind = "unknown_identity_question" if surface.endswith("?") else "unknown_identity_statement"
        out.append(
            {
                "schema": "language.structure_candidate.v2",
                "structure_id": _stable([kind, pattern], "lstruct2"),
                "structure_kind": kind,
                "speech_act": "clarify",
                "surface_example": surface,
                "surface_pattern": pattern,
                "slots": {"target": {"role": "current_obligation_target"}},
                "source": _clean(source, 96),
                "evidence_ref": _clean(evidence_ref, 120),
                "confidence": max(0.0, min(1.0, _float(confidence, 0.46))),
            }
        )
        if len(out) >= MAX_CANDIDATES_PER_TEXT:
            break
    return out


def merge_structure(store: Any, structure: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    current = {
        str(key): dict(value)
        for key, value in (store.items() if isinstance(store, Mapping) else [])
        if isinstance(value, Mapping)
    }
    sid = str(structure.get("structure_id") or "")
    if sid:
        prior = current.get(sid, {})
        merged = dict(prior)
        merged.update(dict(structure))
        merged["use_count"] = int(prior.get("use_count", structure.get("use_count", 0)) or 0)
        merged["confidence"] = max(_float(prior.get("confidence"), 0.0), _float(structure.get("confidence"), 0.0))
        current[sid] = merged
    ranked = sorted(
        current.items(),
        key=lambda item: (int(item[1].get("use_count", 0) or 0), _float(item[1].get("last_used_ts"), 0.0)),
        reverse=True,
    )
    return dict(ranked[:MAX_STRUCTURES])


def structure_to_mem_cell(structure: Mapping[str, Any], tier: str = "now") -> dict[str, Any]:
    payload = dict(structure)
    pattern = _clean(payload.get("surface_pattern"), 180)
    now = time.time()
    return {
        "schema": "mem_cell.v1",
        "id": str(payload.get("structure_id") or _stable(payload, "lstruct2")),
        "kind": "surface_structure_v2",
        "tier": tier,
        "anchor": {"kind": "language/surface_structure_v2", "ref": pattern, "norm": " ".join(_tokens(pattern))},
        "refs": [{"kind": "evidence", "value": ref} for ref in list(payload.get("evidence_refs", []) or [])[:8]],
        "modalities": ["text"],
        "links_explicit": list(payload.get("evidence_refs", []) or [])[:8],
        "activation": 0.34,
        "promotion": 0.02,
        "trust": max(0.0, min(1.0, _float(payload.get("confidence"), 0.46))),
        "meta": {"surface_structure_v2": payload, "truth_status": "structure_shape_not_answer_truth"},
        "ts": _float(payload.get("created_ts"), now),
        "last_seen": now,
    }


def structure_from_mem_cell(cell: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(cell.get("kind") or "") != "surface_structure_v2":
        return None
    meta = cell.get("meta") if isinstance(cell.get("meta"), Mapping) else {}
    payload = meta.get("surface_structure_v2") if isinstance(meta.get("surface_structure_v2"), Mapping) else {}
    normalized = normalize_structure_candidate(payload)
    if normalized:
        normalized["use_count"] = int(cell.get("usage_count", normalized.get("use_count", 0)) or 0)
        normalized["last_used_ts"] = _float(cell.get("last_used_ts"), normalized.get("last_used_ts", 0.0))
    return normalized


def _target_ref(payload: Mapping[str, Any], fallback: Any = "") -> str:
    visual = payload.get("visual_attention_ref") if isinstance(payload.get("visual_attention_ref"), Mapping) else {}
    for value in (
        payload.get("target_ref"),
        payload.get("target"),
        visual.get("track_id"),
        visual.get("selected_track_id"),
        fallback,
    ):
        text = _clean(value, 90)
        if text:
            return text
    return ""


def select_structure(
    obligation: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    target_surface: str,
    signal: str,
) -> dict[str, Any] | None:
    values = structures.values() if isinstance(structures, Mapping) else structures
    required_act = _clean(obligation.get("required_act"), 40).lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        pattern = _clean(candidate.get("surface_pattern"), 180)
        required_slots = set(SLOT_RE.findall(pattern))
        structure_act = _clean(candidate.get("speech_act"), 40).lower()
        if not pattern or (required_act and structure_act and required_act != structure_act):
            continue
        if not required_slots and not bool(candidate.get("allow_slotless_reuse", False)):
            continue
        if "target" in required_slots and not target_surface:
            continue
        if "signal" in required_slots and not signal:
            continue
        score = _float(candidate.get("confidence"), 0.45) + min(0.18, int(candidate.get("use_count", 0) or 0) * 0.01)
        if required_act == "clarify" and pattern.rstrip().endswith("?"):
            score += 0.35
        scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def render_pattern(pattern: Any, *, target: Any = "", signal: Any = "") -> str:
    text = _clean(pattern, 180)
    values = {TARGET_TOKEN: _clean(target, 90), SIGNAL_TOKEN: _clean(signal, 90)}
    for token, value in values.items():
        if token in text:
            if not value:
                return ""
            text = text.replace(token, value)
    return "" if SLOT_RE.search(text) or VOBJ_RE.search(text) or "??" in text else text


def _sentence_floor(text: Any) -> str:
    surface = _clean(text, SURFACE_FLOOR_MAX_LEN).strip(" ")
    if not surface or SLOT_RE.search(surface) or VOBJ_RE.search(surface) or "??" in surface:
        return ""
    if not surface.endswith((".", "!", "?")):
        surface = f"{surface}."
    return surface[:1].upper() + surface[1:]


def _source_after_say(source_text: Any) -> str:
    text = _clean(source_text, SURFACE_FLOOR_MAX_LEN)
    match = re.match(r"^\s*say\s+(.+)$", text, flags=re.IGNORECASE)
    return _sentence_floor(match.group(1)) if match else ""


def required_surface_complexity(obligation: Mapping[str, Any]) -> int:
    act = _clean(obligation.get("required_act"), 40).lower()
    kind = _clean(obligation.get("statement_kind"), 80).lower()
    if act in {"acknowledge", "continue"} or kind in {"greeting", "agreement", "closure"}:
        return 1
    if act in {"respond", "reflect"}:
        return 2
    if act in {"answer", "clarify"}:
        return 3
    if act in {"investigate", "act_or_reply"}:
        return 4
    return 2


def build_surface_floor(obligation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a minimal non-learning surface for a required but unformed reply."""

    payload = obligation if isinstance(obligation, Mapping) else {}
    act = _clean(payload.get("required_act"), 40).lower()
    kind = _clean(payload.get("statement_kind"), 80).lower()
    source_text = _clean(payload.get("source_text"), 240)
    source_norm = " ".join(_tokens(source_text))

    direct_say = _source_after_say(source_text)
    if direct_say:
        surface = direct_say
        status = "speech_floor_direct_say"
    elif kind == "greeting" or source_norm in {"good morning", "good evening", "hello", "hi"}:
        surface = _sentence_floor(source_text or "hello")
        status = "speech_floor_greeting"
    elif act == "answer":
        surface = "Reply not ready." if "no reply" in source_norm else "Unknown."
        status = "speech_floor_answer_unknown"
    elif act == "clarify":
        surface = "What?"
        status = "speech_floor_minimal_clarify"
    elif act in {"acknowledge", "continue", "respond", "reflect"}:
        surface = "I follow."
        status = "speech_floor_acknowledge"
    else:
        surface = "Reply not ready."
        status = "speech_floor_reply_not_ready"

    complexity = required_surface_complexity(payload)
    return {
        "surface": surface,
        "surface_status": status,
        "surface_floor": True,
        "surface_floor_policy": "minimal_non_learning_discharge",
        "capability_mismatch": {
            "schema": "language.capability_mismatch.v1",
            "required_response_complexity": complexity,
            "available_surface_complexity": 1,
            "mismatch": max(0, complexity - 1),
            "babble_dependency": False,
            "correction_dependency": False,
            "self_question": "why can I recognize a reply is needed but not form one?",
            "self_question_step_budget": 2,
        },
        "store_in_memory": False,
        "reinforcement_eligible": False,
    }


def build_surface_plan(
    obligation: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    *,
    fallback_target: Any = "",
) -> dict[str, Any]:
    payload = obligation if isinstance(obligation, Mapping) else {}
    target_ref = _target_ref(payload, fallback_target)
    visual = payload.get("visual_attention_ref") if isinstance(payload.get("visual_attention_ref"), Mapping) else {}
    target_surface = _clean(payload.get("target_surface"), 90) or outward_target_surface(target_ref, visual)
    signal = _clean(payload.get("signal") or payload.get("target_signal"), 90)
    selected = select_structure(payload, structures or {}, target_surface=target_surface, signal=signal)
    surface = render_pattern(selected.get("surface_pattern", ""), target=target_surface, signal=signal) if selected else ""
    status = "constructed_from_learned_structure" if surface else ""
    if not surface:
        options = [
            _clean(item, 180)
            for item in list(payload.get("surface_options", []) or [])
            if _clean(item, 180) and not SLOT_RE.search(_clean(item, 180)) and not VOBJ_RE.search(_clean(item, 180))
        ]
        if options:
            surface, status = options[0], "obligation_supplied_surface"
        elif _clean(payload.get("required_act"), 40).lower() == "clarify" and (target_surface or signal):
            surface = f"{target_surface or signal}?"
            status = "primitive_contextual_question_handle"
        else:
            floor = build_surface_floor(payload) if bool(payload.get("required", False)) else {}
            surface = str(floor.get("surface", "") or "")
            status = str(floor.get("surface_status", "") or "expression_gap_no_compatible_structure")
    floor = build_surface_floor(payload) if status.startswith("speech_floor_") else {}
    complete = bool(surface and not SLOT_RE.search(surface) and not VOBJ_RE.search(surface) and "??" not in surface)
    obligation_id = _clean(payload.get("obligation_id"), 96)
    selected_id = str(selected.get("structure_id") or "") if selected else ""
    return {
        "schema": "language.surface_plan.v2",
        "plan_id": _stable([obligation_id, selected_id, target_ref, target_surface, surface], "surface_plan2"),
        "obligation_id": obligation_id,
        "required": bool(payload.get("required", False)),
        "required_act": _clean(payload.get("required_act"), 40).lower(),
        "gap_kind": _clean(payload.get("gap_kind") or payload.get("obligation_kind"), 80).lower(),
        "target_ref": target_ref,
        "target_surface": target_surface,
        "signal": signal,
        "selected_structure_id": selected_id,
        "surface": surface,
        "surface_status": status,
        "surface_complete": complete,
        "surface_floor": bool(floor.get("surface_floor", False)),
        "surface_floor_policy": str(floor.get("surface_floor_policy", "") or ""),
        "capability_mismatch": dict(floor.get("capability_mismatch", {}) or {}),
        "store_in_memory": False if bool(floor.get("surface_floor", False)) else None,
        "reinforcement_eligible": False if bool(floor.get("surface_floor", False)) else None,
        "truth_status": "surface_candidate_not_answer_truth",
        "not_canned_response": True,
    }


def build_surface_candidate(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(plan),
        "schema": "language.surface_candidate.v2",
        "requires_review_by_mouth": True,
    }
