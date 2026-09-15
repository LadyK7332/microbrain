from __future__ import annotations

"""Explicit response obligations shared by policy, realization, and speech.

An obligation is created only after the hypothesis/release path has decided
that an outward response is warranted.  It therefore does not turn every user
statement into speech; it prevents a response already selected by cognition
from disappearing between planning and the mouth.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

RESPONSE_OBLIGATION_SCHEMA = "speech.response_obligation.v2"


def _clean(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[:limit]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if number != number else number


def _stable_id(value: Any, prefix: str = "obligation") -> str:
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(value)
    digest = hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"


def speech_act_for_action(action: Any, statement_kind: Any = "") -> str:
    selected = _clean(action, 80).lower()
    kind = _clean(statement_kind, 80).lower()
    if selected in {"clarify", "ask_followup"}:
        return "clarify"
    if selected in {"acknowledge", "acknowledge_revision"}:
        return "acknowledge"
    if selected in {"continue_thread", "reflect"}:
        return "continue"
    if selected in {"decline", "refuse"}:
        return "decline"
    if selected in {"answer", "investigate", "respond"}:
        return "answer"
    if kind in {"correction", "disagreement", "status_update", "personal_state", "agreement", "greeting"}:
        return "acknowledge"
    if kind in {"question", "request"}:
        return "answer"
    return "respond"


def visual_binding_from_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    ctx = context if isinstance(context, Mapping) else {}
    attention = ctx.get("attention") if isinstance(ctx.get("attention"), Mapping) else {}
    visual = attention.get("visual_ref") if isinstance(attention.get("visual_ref"), Mapping) else {}
    if not visual:
        input_block = ctx.get("input") if isinstance(ctx.get("input"), Mapping) else {}
        raw_meta = input_block.get("raw_meta") if isinstance(input_block.get("raw_meta"), Mapping) else {}
        visual = raw_meta.get("visual_attention_ref") if isinstance(raw_meta.get("visual_attention_ref"), Mapping) else {}
    return dict(visual)


def outward_target_surface(target_ref: Any, visual_ref: Mapping[str, Any] | None = None) -> str:
    """Return a speakable target while retaining the internal ref separately."""

    visual = visual_ref if isinstance(visual_ref, Mapping) else {}
    label = _clean(visual.get("label_hint") or visual.get("label"), 80)
    if label.lower() not in {"", "unknown", "unknown_object", "object", "thing", "unclassified"}:
        return label
    target = _clean(target_ref, 90)
    if target.lower().startswith("vobj:"):
        return "that"
    return target


def build_response_obligation(
    *,
    context: Mapping[str, Any] | None,
    hypothesis: Mapping[str, Any] | None,
    trigger: Mapping[str, Any] | None,
    correlation_id: Any = "",
) -> dict[str, Any] | None:
    """Create the contract for a response already released by cognition."""

    ctx = context if isinstance(context, Mapping) else {}
    hyp = hypothesis if isinstance(hypothesis, Mapping) else {}
    trig = trigger if isinstance(trigger, Mapping) else {}
    if not bool(trig.get("should_release", hyp.get("should_respond", False))):
        return None

    pattern = hyp.get("pattern_analysis") if isinstance(hyp.get("pattern_analysis"), Mapping) else {}
    action = _clean(trig.get("recommended_action") or hyp.get("recommended_action") or "respond", 80).lower()
    if action == "silence":
        return None
    statement_kind = _clean(pattern.get("statement_kind") or "statement", 80).lower()
    speech_act = speech_act_for_action(action, statement_kind)
    input_block = ctx.get("input") if isinstance(ctx.get("input"), Mapping) else {}
    visual_ref = visual_binding_from_context(ctx)
    attention = ctx.get("attention") if isinstance(ctx.get("attention"), Mapping) else {}
    target_ref = _clean(
        visual_ref.get("track_id")
        or visual_ref.get("selected_track_id")
        or attention.get("deictic_binding_hint"),
        90,
    )
    turn_id = _clean(input_block.get("correlation_id") or correlation_id, 96)
    hypothesis_id = _clean(hyp.get("hypothesis_id"), 96)
    obligation_id = _stable_id([turn_id, hypothesis_id, speech_act])
    return {
        "schema": RESPONSE_OBLIGATION_SCHEMA,
        "obligation_id": obligation_id,
        "turn_id": turn_id,
        "hypothesis_id": hypothesis_id,
        "required": True,
        "required_act": speech_act,
        "selected_action": action,
        "statement_kind": statement_kind,
        "response_demand": _safe_float(hyp.get("response_demand"), 0.0),
        "silence_allowed": False,
        "source_text": _clean(input_block.get("text"), 240),
        "target_ref": target_ref,
        "target_surface": outward_target_surface(target_ref, visual_ref),
        "visual_attention_ref": visual_ref,
        "surface_options": [],
        "reason": "hypothesis_release_selected_outward_response",
    }


def obligation_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    value = payload.get("response_obligation")
    return dict(value) if isinstance(value, Mapping) else {}
