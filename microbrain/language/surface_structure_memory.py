from __future__ import annotations

"""Learned surface-structure memory for language realization.

This module keeps language in the shape MB has been moving toward:

* reading provides reusable sentence molds;
* current perception/context supplies the slot handles;
* a gap decides which mold should be tried;
* the resulting surface is a candidate, not hard truth and not a canned answer.

The module is intentionally small and inspectable.  It does not try to be a full
parser.  It generalizes common quote/reading examples into slot-bearing surface
patterns and renders them against active refs such as ``vobj:07``.
"""

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

MAX_SURFACE_CHARS = 220
MAX_CONTEXT_CHARS = 260
MAX_STRUCTURES_IN_KV = 256
MAX_RENDERED_SURFACE_CHARS = 180
MAX_STRUCTURE_EVIDENCE_REFS = 8

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

SURFACE_STRUCTURE_SCHEMA = "language.surface_structure.v1"
SURFACE_PLAN_SCHEMA = "language.surface_plan.v1"
SURFACE_CANDIDATE_SCHEMA = "language.surface_candidate.v1"
STORE_KV_KEY = "language:surface_structures"
LAST_SURFACE_PLAN_KV_KEY = "language:last_surface_plan"
LAST_SURFACE_CANDIDATE_KV_KEY = "language:last_surface_candidate"

TARGET_TOKEN = "{target}"
SUBJECT_TOKEN = "{subject}"
PREDICATE_TOKEN = "{predicate}"
SIGNAL_TOKEN = "{signal}"

WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
VOBJ_RE = re.compile(r"\bvobj:[a-z0-9_.:-]+\b", re.IGNORECASE)

DEMONSTRATIVE_TARGETS = ("that", "this", "it", "there")
UNKNOWN_IDENTITY_KINDS = {
    "object_identity_unknown",
    "identity_unknown",
    "visual_identity_unknown",
    "unknown_visual_object",
    "recognition_gap",
}
GESTURE_GAP_KINDS = {
    "gesture_meaning_unknown",
    "paralinguistic_signal_unknown",
    "intent_ambiguous",
    "minimal_signal",
}


@dataclass(slots=True)
class SurfaceStructure:
    """A learned/reusable language mold with contextual slots."""

    structure_id: str
    structure_kind: str
    surface_pattern: str
    slots: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    surface_example: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = 0.0
    use_count: int = 0
    confidence: float = 0.5
    truth_status: str = "structure_shape_not_answer_truth"
    not_canned_response: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SURFACE_STRUCTURE_SCHEMA,
            "structure_id": self.structure_id,
            "structure_kind": self.structure_kind,
            "surface_pattern": self.surface_pattern,
            "slots": dict(self.slots),
            "source": self.source,
            "surface_example": self.surface_example,
            "evidence_refs": list(self.evidence_refs[:MAX_STRUCTURE_EVIDENCE_REFS]),
            "created_ts": self.created_ts,
            "last_used_ts": self.last_used_ts,
            "use_count": self.use_count,
            "confidence": self.confidence,
            "truth_status": self.truth_status,
            "not_canned_response": self.not_canned_response,
            "requires_contextual_slot_fill": True,
        }


def _clean_text(value: Any, *, limit: int = MAX_SURFACE_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(1, int(limit) - 1)].rstrip() + "…"
    return text


def _tokens(text: Any) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))]


def _stable_id(value: Any, *, prefix: str = "lstruct") -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _extract_evidence_refs(payload: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("quote_id", "source_quote_id", "artifact_ref", "evidence_ref", "cell_id", "structure_id"):
        value = payload.get(key)
        if value:
            refs.append(str(value))
    return refs[:MAX_STRUCTURE_EVIDENCE_REFS]


def _replace_first_word_case_preserving(surface: str, word: str, replacement: str) -> str:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return pattern.sub(replacement, surface, count=1)


def infer_surface_pattern(surface_example: Any, structure_kind: Any = "", slots: Mapping[str, Any] | None = None) -> str:
    """Turn an encountered sentence into a slot-bearing mold.

    The replacement is intentionally conservative.  It handles common examples
    from reading such as ``What is that?`` and ``I don't know what that is.`` by
    replacing only a demonstrative target.  If no slot-like target is found, the
    surface is returned unchanged and will not be chosen for target rendering.
    """

    surface = _clean_text(surface_example)
    kind = _clean_text(structure_kind, limit=80).lower()
    if not surface:
        return ""

    # Already generalized by another organ.
    if "{" in surface and "}" in surface:
        return surface

    # Explicit visual handle in a read/training sentence can be generalized.
    if VOBJ_RE.search(surface):
        return VOBJ_RE.sub(TARGET_TOKEN, surface, count=1)

    words = _tokens(surface)
    if any(token in words for token in DEMONSTRATIVE_TARGETS):
        for target_word in DEMONSTRATIVE_TARGETS:
            if target_word in words:
                return _replace_first_word_case_preserving(surface, target_word, TARGET_TOKEN)

    lower = surface.lower()
    if kind in {"unknown_identity_question", "question_structure"} and lower.startswith("what is "):
        return "What is " + TARGET_TOKEN + ("?" if surface.endswith("?") else "")
    if "don't know" in lower and " what " in f" {lower} ":
        # Best-effort mold; only used when reading supplied this style.
        return "I don't know what " + TARGET_TOKEN + " is."
    return surface


def normalize_structure_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert quote/reference structure candidates into a durable-ish mold."""

    if not isinstance(candidate, Mapping):
        return None
    surface_example = _clean_text(
        candidate.get("surface_example")
        or candidate.get("surface_pattern")
        or candidate.get("quote")
        or candidate.get("text")
    )
    structure_kind = _clean_text(candidate.get("structure_kind") or candidate.get("kind") or "surface_structure", limit=80)
    pattern = _clean_text(candidate.get("surface_pattern") or infer_surface_pattern(surface_example, structure_kind, _as_mapping(candidate.get("slots"))))
    if not pattern:
        return None
    slots = dict(_as_mapping(candidate.get("slots")))
    if TARGET_TOKEN in pattern and "target" not in slots:
        slots["target"] = {"role": "current_gap_target", "allowed_refs": ["vobj", "object", "sound", "signal", "concept"]}
    evidence_refs = _extract_evidence_refs(candidate)
    structure = SurfaceStructure(
        structure_id=str(candidate.get("structure_id") or _stable_id([structure_kind, pattern, surface_example])),
        structure_kind=structure_kind,
        surface_pattern=pattern,
        slots=slots,
        source=_clean_text(candidate.get("source") or "reading_structure_candidate", limit=80),
        surface_example=surface_example,
        evidence_refs=evidence_refs,
        confidence=float(candidate.get("confidence", 0.58) or 0.58),
        truth_status="structure_shape_not_answer_truth",
        not_canned_response=True,
    )
    payload = structure.to_payload()
    payload["learned_from_context"] = {
        "source_quote_id": candidate.get("source_quote_id") or candidate.get("quote_id") or "",
        "learning_use": candidate.get("learning_use", ""),
        "truth_status": candidate.get("truth_status", "source_framed_revisable_not_hard_truth"),
    }
    return payload


def merge_structure_into_store(store: Any, structure: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a bounded KV-safe structure store with ``structure`` merged in."""

    current: dict[str, dict[str, Any]] = {}
    if isinstance(store, Mapping):
        for sid, payload in store.items():
            if isinstance(payload, Mapping):
                current[str(sid)] = dict(payload)
    sid = str(structure.get("structure_id") or "")
    if sid:
        prior = current.get(sid, {})
        merged = dict(prior)
        merged.update(dict(structure))
        if prior:
            merged["use_count"] = int(prior.get("use_count", 0) or 0)
            merged["created_ts"] = float(prior.get("created_ts", structure.get("created_ts", time.time())) or time.time())
            merged["confidence"] = max(float(prior.get("confidence", 0.0) or 0.0), float(structure.get("confidence", 0.0) or 0.0))
        current[sid] = merged

    if len(current) <= MAX_STRUCTURES_IN_KV:
        return current
    # Keep most-used/recent structures.  This is runtime KV, not the final
    # durable memory design.
    ranked = sorted(
        current.items(),
        key=lambda item: (int(item[1].get("use_count", 0) or 0), float(item[1].get("last_used_ts", item[1].get("created_ts", 0.0)) or 0.0)),
        reverse=True,
    )
    return dict(ranked[:MAX_STRUCTURES_IN_KV])


def _gap_kind(gap_payload: Mapping[str, Any]) -> str:
    return _clean_text(
        gap_payload.get("gap_kind")
        or gap_payload.get("kind")
        or gap_payload.get("response_obligation")
        or gap_payload.get("obligation_kind"),
        limit=90,
    )


def _candidate_score(structure: Mapping[str, Any], gap_kind: str, target_ref: str) -> float:
    kind = _clean_text(structure.get("structure_kind"), limit=90).lower()
    pattern = _clean_text(structure.get("surface_pattern"), limit=MAX_RENDERED_SURFACE_CHARS).lower()
    score = float(structure.get("confidence", 0.45) or 0.45)
    gap = gap_kind.lower()
    if TARGET_TOKEN in pattern:
        score += 0.2
    if gap in UNKNOWN_IDENTITY_KINDS:
        if kind == "unknown_identity_question":
            score += 0.55
        if "what is" in pattern:
            score += 0.35
        if "don't know" in pattern:
            score += 0.2
    if gap in GESTURE_GAP_KINDS:
        if kind in {"question_structure", "unknown_identity_question"}:
            score += 0.15
        if SIGNAL_TOKEN in pattern:
            score += 0.4
    if target_ref and target_ref.lower().startswith("vobj:") and "target" in str(structure.get("slots", {})).lower():
        score += 0.1
    score += min(0.15, int(structure.get("use_count", 0) or 0) * 0.01)
    return score


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                text = _clean_text(item, limit=90)
                if text:
                    return text
            continue
        text = _clean_text(value, limit=90)
        if text:
            return text
    return ""


def target_from_gap(gap_payload: Mapping[str, Any], *, fallback_target: Any = "") -> str:
    """Extract the active handle the surface structure should speak about."""

    payload = _as_mapping(gap_payload)
    target = _first_nonempty(
        payload.get("target"),
        payload.get("target_ref"),
        payload.get("unknown_target"),
        payload.get("object_ref"),
        payload.get("vobj_id"),
        payload.get("current_focus"),
        payload.get("current_vobj_id"),
        fallback_target,
    )
    if target and not target.startswith("vobj:") and re.fullmatch(r"[0-9]+", target):
        target = f"vobj:{target}"
    if not target:
        missing = payload.get("missing")
        if isinstance(missing, Iterable) and not isinstance(missing, (str, bytes, Mapping)):
            for item in missing:
                text = _clean_text(item, limit=90)
                if VOBJ_RE.fullmatch(text):
                    return text
    return target


def select_structure_for_gap(
    gap_payload: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    target_ref: Any = "",
) -> dict[str, Any] | None:
    """Choose the best learned mold for the current gap."""

    if isinstance(structures, Mapping):
        candidates = [dict(v) for v in structures.values() if isinstance(v, Mapping)]
    else:
        candidates = [dict(v) for v in structures if isinstance(v, Mapping)]
    target = _clean_text(target_ref or target_from_gap(gap_payload), limit=90)
    gap = _gap_kind(gap_payload)
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        pattern = _clean_text(candidate.get("surface_pattern"), limit=MAX_RENDERED_SURFACE_CHARS)
        if not pattern:
            continue
        # Only select target-bearing structures for target-bearing gaps.  This
        # prevents example quotes from becoming canned replies.
        if target and TARGET_TOKEN not in pattern and SIGNAL_TOKEN not in pattern:
            continue
        scored.append((_candidate_score(candidate, gap, target), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return dict(scored[0][1])


def render_pattern(pattern: Any, *, target_ref: Any = "", signal: Any = "", subject: Any = "", predicate: Any = "") -> str:
    surface = _clean_text(pattern, limit=MAX_RENDERED_SURFACE_CHARS)
    replacements = {
        TARGET_TOKEN: _clean_text(target_ref, limit=90),
        SIGNAL_TOKEN: _clean_text(signal, limit=90),
        SUBJECT_TOKEN: _clean_text(subject, limit=90),
        PREDICATE_TOKEN: _clean_text(predicate, limit=120),
    }
    for token, value in replacements.items():
        if token in surface:
            surface = surface.replace(token, value or "?")
    surface = " ".join(surface.split())
    if len(surface) > MAX_RENDERED_SURFACE_CHARS:
        surface = surface[: MAX_RENDERED_SURFACE_CHARS - 1].rstrip() + "…"
    return surface


def primitive_gap_surface(target_ref: Any = "", *, signal: Any = "") -> str:
    target = _clean_text(target_ref, limit=90)
    sig = _clean_text(signal, limit=90)
    if target:
        return f"{target}?"
    if sig:
        return f"{sig}?"
    return "?"


def build_surface_plan_for_gap(
    gap_payload: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    *,
    fallback_target: Any = "",
) -> dict[str, Any]:
    """Build an inspectable language plan from a cognition gap."""

    payload = _as_mapping(gap_payload)
    target = target_from_gap(payload, fallback_target=fallback_target)
    signal = _first_nonempty(payload.get("signal"), payload.get("surface"), payload.get("text"), payload.get("target_signal"))
    gap_kind = _gap_kind(payload)
    store = structures or {}
    selected = select_structure_for_gap(payload, store, target_ref=target)
    if selected:
        surface = render_pattern(selected.get("surface_pattern", ""), target_ref=target, signal=signal)
        status = "constructed_from_learned_structure"
    else:
        surface = primitive_gap_surface(target, signal=signal)
        status = "primitive_placeholder_no_learned_structure"
    plan_id = _stable_id([gap_kind, target, signal, selected.get("structure_id") if selected else "primitive", time.time()], prefix="surface_plan")
    return {
        "schema": SURFACE_PLAN_SCHEMA,
        "plan_id": plan_id,
        "gap_kind": gap_kind,
        "target": target,
        "signal": signal,
        "selected_structure_id": selected.get("structure_id", "") if selected else "",
        "selected_structure_kind": selected.get("structure_kind", "") if selected else "",
        "surface_status": status,
        "surface": surface,
        "slots_filled": {
            "target": target,
            "signal": signal,
        },
        "truth_status": "surface_candidate_not_answer_truth",
        "not_canned_response": True,
        "requires_review_by_mouth": True,
    }


def build_surface_candidate_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    return {
        "schema": SURFACE_CANDIDATE_SCHEMA,
        "plan_id": payload.get("plan_id", ""),
        "surface": payload.get("surface", "?"),
        "surface_status": payload.get("surface_status", ""),
        "gap_kind": payload.get("gap_kind", ""),
        "target": payload.get("target", ""),
        "signal": payload.get("signal", ""),
        "selected_structure_id": payload.get("selected_structure_id", ""),
        "truth_status": "surface_candidate_not_answer_truth",
        "not_canned_response": True,
        "requires_review_by_mouth": True,
    }
