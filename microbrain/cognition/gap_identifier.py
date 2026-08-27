from __future__ import annotations

"""Identify missing evidence/intent gaps before speech/action selection finalizes.

The Gap Identifier is deliberately not a general responder.  It labels what is
missing, decides whether a question/perception request is a safe way to close the
missing piece, and treats silence as a last-resort answer rather than the default
for ambiguity.
"""

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

GAP_UNCERTAINTY_THRESHOLD = 0.62
MINIMAL_SIGNAL_RESPONSE_DEMAND = 0.25
CLARIFICATION_PRIORITY = 0.58
EVIDENCE_NEED_PRIORITY = 0.54
MAX_SURFACE_CHARS = 160
MAX_TEXT_CHARS = 240
MAX_MISSING_ITEMS = 8
AUTO_CLARIFY_SILENT_USER_GAPS = True

# Silence is still valid when speech would be harmful/noisy/stale.
STALE_REPEAT_LIMIT = 2

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

GAP_IDENTIFIED_SCHEMA = "cognition.gap_identified.v1"
CLARIFICATION_NEED_SCHEMA = "cognition.clarification_need.v1"
EVIDENCE_NEED_SCHEMA = "cognition.evidence_need.v1"
SPEECH_OBLIGATION_SCHEMA = "speech.response_obligation.v1"
GAP_SPEECH_PAYLOAD_SCHEMA = "speech.gap_clarification.v1"
TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

PARALINGUISTIC_SURFACES = {
    "o.o",
    "o_o",
    "o o",
    "o-o",
    "o0",
    "0o",
    ":o",
    ":0",
    "._.",
    "-_-",
    "x.x",
}

UNKNOWN_LABELS = {
    "",
    "unknown",
    "unknown_object",
    "unknown object",
    "uncertain",
    "unclassified",
    "unidentified",
    "object",
    "thing",
}

QUESTION_BLOCK_REASONS = {
    "crisis_mode": "speech_may_delay_urgent_safety_action",
    "human_uplift_negative": "question_may_worsen_human_outcome",
    "self_damage_unnecessary": "question_or_action_may_cause_unnecessary_self_damage",
    "privacy_risk": "question_may_increase_privacy_or_surveillance_harm",
    "stale_repetition": "same_gap_repeated_without_new_information",
    "not_user_originated": "gap_did_not_originate_from_user_intent",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _clean_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(1, int(limit) - 1)].rstrip() + "…"
    return text


def _float01(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not number == number:
        return default
    return max(0.0, min(1.0, number))


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    if value is None:
        return default
    return bool(value)


def _as_list(value: Any, *, limit: int = 16) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)[: max(0, int(limit))]
    if value is None:
        return []
    return [value]


def _stable_id(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _get_path(root: Any, *keys: str, default: Any = None) -> Any:
    node = root
    for key in keys:
        if not isinstance(node, Mapping):
            return default
        node = node.get(key)
    return default if node is None else node


def _topic_modality(topic: str, payload: Any, context: Mapping[str, Any]) -> str:
    topic_s = str(topic or "")
    if "vision" in topic_s:
        return "vision"
    if "audio" in topic_s or "sound" in topic_s:
        return "audio"
    if "touch" in topic_s:
        return "touch"
    if "text" in topic_s or "language" in topic_s:
        return "text"
    channel = _clean_text(_get_path(context, "input", "channel", default=""), limit=40).lower()
    if channel == "textual":
        return "text"
    if channel:
        return channel
    if isinstance(payload, Mapping):
        channel = _clean_text(payload.get("channel", ""), limit=40).lower()
        if channel == "textual":
            return "text"
        if channel:
            return channel
    return "unknown"


def _input_packet(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        context = payload.get("context")
        if isinstance(context, Mapping) and isinstance(context.get("input"), Mapping):
            return context.get("input") or {}
        if isinstance(payload.get("input"), Mapping):
            return payload.get("input") or {}
    return {}


def _context_packet(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("context"), Mapping):
        return payload.get("context") or {}
    return {}


def extract_user_text(payload: Any) -> str:
    if isinstance(payload, str):
        return _clean_text(payload)
    if not isinstance(payload, Mapping):
        return ""
    input_packet = _input_packet(payload)
    for candidate in (
        input_packet.get("text"),
        _get_path(payload, "trigger", "text", default=""),
        payload.get("text"),
        payload.get("utterance"),
    ):
        text = _clean_text(candidate)
        if text:
            return text
    return ""


def extract_tokens(payload: Any) -> list[str]:
    input_packet = _input_packet(payload)
    raw = input_packet.get("tokens") if isinstance(input_packet, Mapping) else None
    if not raw and isinstance(payload, Mapping):
        raw = payload.get("tokens") or _get_path(payload, "hypothesis", "pattern_analysis", "tokens", default=[])
    out: list[str] = []
    for token in _as_list(raw, limit=64):
        text = _clean_text(token, limit=48).lower()
        if text:
            out.append(text)
    if out:
        return out
    return [m.group(0).lower() for m in TOKEN_RE.finditer(extract_user_text(payload))]


def extract_meaningful_tokens(payload: Any) -> list[str]:
    input_packet = _input_packet(payload)
    raw = input_packet.get("meaningful_tokens") if isinstance(input_packet, Mapping) else None
    if raw is None and isinstance(payload, Mapping):
        raw = _get_path(payload, "hypothesis", "pattern_analysis", "meaningful_tokens", default=[])
    out: list[str] = []
    for token in _as_list(raw, limit=64):
        text = _clean_text(token, limit=48).lower()
        if text:
            out.append(text)
    return out


def is_user_originated_text(payload: Any, *, source: Any = "") -> bool:
    text = extract_user_text(payload)
    if not text:
        return False
    context = _context_packet(payload)
    input_packet = _input_packet(payload)
    channel = _clean_text(input_packet.get("channel") or _get_path(context, "input", "channel", default=""), limit=40).lower()
    raw_meta = input_packet.get("raw_meta") if isinstance(input_packet, Mapping) else None
    input_source = _clean_text(input_packet.get("source"), limit=80).lower()
    transport_source = ""
    if isinstance(raw_meta, Mapping):
        if not input_source:
            input_source = _clean_text(raw_meta.get("source"), limit=80).lower()
        transport_source = _clean_text(raw_meta.get("transport_source") or raw_meta.get("frontend"), limit=80).lower()
    source_s = _clean_text(source or input_source or transport_source, limit=80).lower()
    if channel in {"textual", "text", "chat"}:
        return True
    return source_s in {"ui", "dashboard", "user", "frontend"}


def is_paralinguistic_signal(text: str, tokens: Sequence[str] | None = None) -> bool:
    norm = _clean_text(text, limit=80).lower().replace(" ", "")
    spaced = _clean_text(text, limit=80).lower().replace(".", " ").replace("_", " ").replace("-", " ")
    if norm in PARALINGUISTIC_SURFACES or spaced in PARALINGUISTIC_SURFACES:
        return True
    raw_tokens = [str(t).lower() for t in (tokens or [])]
    return bool(raw_tokens and len(raw_tokens) <= 3 and all(t in {"o", "0", "x"} for t in raw_tokens))


def extract_uncertainty(payload: Any) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    for candidate in (
        _get_path(payload, "hypothesis", "pattern_analysis", "uncertainty", default=None),
        _get_path(payload, "pattern_analysis", "uncertainty", default=None),
        payload.get("uncertainty"),
        _get_path(payload, "analysis", "uncertainty", default=None),
    ):
        if candidate is not None:
            return _float01(candidate, default=0.0)
    interpretations = _get_path(payload, "hypothesis", "interpretations", default=[])
    for item in _as_list(interpretations, limit=16):
        if isinstance(item, Mapping) and "uncertain" in _clean_text(item.get("interpretation"), limit=120).lower():
            return max(0.0, _float01(item.get("confidence"), default=0.0))
    return 0.0


def extract_statement_kind(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for candidate in (
        _get_path(payload, "hypothesis", "pattern_analysis", "statement_kind", default=""),
        _get_path(payload, "hypothesis", "ddna_tuning", "statement_kind", default=""),
        payload.get("statement_kind", ""),
    ):
        text = _clean_text(candidate, limit=80).lower()
        if text:
            return text
    return ""


def extract_selected_action(payload: Any, event_meta: Mapping[str, Any] | None = None) -> str:
    meta = event_meta or {}
    for candidate in (
        meta.get("selected_action"),
        _get_path(payload, "meta", "selected_action", default="") if isinstance(payload, Mapping) else "",
        _get_path(payload, "trigger", "recommended_action", default="") if isinstance(payload, Mapping) else "",
        _get_path(payload, "hypothesis", "recommended_action", default="") if isinstance(payload, Mapping) else "",
        _get_path(payload, "trigger", "kind", default="") if isinstance(payload, Mapping) else "",
    ):
        text = _clean_text(candidate, limit=80).lower()
        if text:
            return text
    return ""


def extract_response_demand(payload: Any) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    for candidate in (
        _get_path(payload, "hypothesis", "response_demand", default=None),
        _get_path(payload, "trigger", "response_demand", default=None),
        payload.get("response_demand"),
        _get_path(payload, "pattern_analysis", "response_expectation", default=None),
    ):
        if candidate is not None:
            return _float01(candidate, default=0.0)
    return 0.0


def human_uplift_question_gate(payload: Any, *, source: Any = "") -> dict[str, Any]:
    """Return whether a clarification question is safe/useful right now.

    This is intentionally conservative but not mute.  Questions are normally
    valid for user-originated ambiguity; blocks must have concrete harm/noise
    reasons instead of falling back to silence.
    """
    context = _context_packet(payload)
    constraints = context.get("constraints") if isinstance(context, Mapping) else {}
    constraints = constraints if isinstance(constraints, Mapping) else {}
    drives = context.get("drives") if isinstance(context, Mapping) else {}
    drives = drives if isinstance(drives, Mapping) else {}

    reasons: list[str] = []
    if _bool(constraints.get("crisis_mode"), default=False):
        reasons.append("crisis_mode")
    if _bool(constraints.get("human_uplift_negative"), default=False):
        reasons.append("human_uplift_negative")
    if _bool(constraints.get("self_damage_unnecessary"), default=False):
        reasons.append("self_damage_unnecessary")
    if _bool(constraints.get("privacy_risk"), default=False):
        reasons.append("privacy_risk")

    boredom = drives.get("boredom") if isinstance(drives.get("boredom"), Mapping) else {}
    same_user = int(_float01(boredom.get("same_user_repetitions"), default=0.0) * 10) if boredom else 0
    try:
        same_user = int(boredom.get("same_user_repetitions", same_user)) if boredom else 0
    except (TypeError, ValueError):
        same_user = 0
    if same_user >= STALE_REPEAT_LIMIT:
        reasons.append("stale_repetition")

    if not is_user_originated_text(payload, source=source):
        # Non-user sensory uncertainty should usually request a sense/action
        # rather than ask the user, unless another organ has already framed it.
        reasons.append("not_user_originated")

    allowed = not reasons
    return {
        "schema": "human_uplift.question_gate.v1",
        "allowed": allowed,
        "blocked_reasons": reasons,
        "blocked_reason_text": [QUESTION_BLOCK_REASONS.get(r, r) for r in reasons],
        "principle": "questions_are_valid_unless_they_worsen_human_uplift_or_unnecessary_self_damage",
    }


def _visual_label(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in ("label", "object_label", "class", "kind", "candidate_label", "recognized_as"):
        text = _clean_text(payload.get(key), limit=80).lower()
        if text:
            return text
    for nested_key in ("visual_ref", "object", "candidate", "recognition", "classification"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            text = _visual_label(nested)
            if text:
                return text
    return ""


def _confidence_from_payload(payload: Any) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("confidence", "score", "identity_confidence", "objecthood_confidence", "match_confidence"):
        if key in payload:
            return _float01(payload.get(key), default=0.0)
    for nested_key in ("visual_ref", "object", "candidate", "recognition", "classification", "audio", "sound"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            value = _confidence_from_payload(nested)
            if value is not None:
                return value
    return None


def _audio_direction_known(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for key in ("direction", "bearing", "azimuth", "source_direction", "stereo_direction"):
        value = payload.get(key)
        if value not in (None, "", "unknown", "uncertain"):
            return True
    nested = payload.get("audio") or payload.get("sound") or payload.get("source")
    return _audio_direction_known(nested) if isinstance(nested, Mapping) else False


def _audio_kind_known(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for key in ("kind", "label", "classification", "sound_type", "source_type"):
        value = _clean_text(payload.get(key), limit=80).lower()
        if value and value not in UNKNOWN_LABELS:
            return True
    nested = payload.get("audio") or payload.get("sound") or payload.get("source")
    return _audio_kind_known(nested) if isinstance(nested, Mapping) else False


def _evidence_need_for_modality(topic: str, payload: Any, modality: str) -> dict[str, Any] | None:
    topic_s = str(topic or "")
    if modality == "vision":
        conf = _confidence_from_payload(payload)
        label = _visual_label(payload)
        uncertain = (conf is not None and conf < GAP_UNCERTAINTY_THRESHOLD) or label in UNKNOWN_LABELS or "uncertain" in topic_s
        if uncertain:
            return {
                "modality": "vision",
                "question": "what is the object/scene change and does the view need readjustment?",
                "suggested_action": "vision_readjust_refocus_or_recenter",
                "reason": "vision_identity_or_scene_normalization_gap",
            }
    if modality == "audio":
        direction_known = _audio_direction_known(payload)
        kind_known = _audio_kind_known(payload)
        if not direction_known or not kind_known:
            missing = []
            if not kind_known:
                missing.append("sound_type")
            if not direction_known:
                missing.append("direction")
            return {
                "modality": "audio",
                "question": "what made the sound and which way did it come from?",
                "suggested_action": "stereo_direction_check_and_sound_classify",
                "reason": "audio_source_or_direction_gap",
                "missing": missing,
            }
    if modality == "text":
        text = extract_user_text(payload)
        tokens = extract_tokens(payload)
        meaningful = extract_meaningful_tokens(payload)
        if text and not meaningful and is_paralinguistic_signal(text, tokens):
            return None
    return None


def _suggested_clarification_surface(text: str, *, paralinguistic: bool = False) -> str:
    if paralinguistic:
        return "Something catch your attention?"
    if text:
        return "I'm not sure how to read that — what changed?"
    return "What should I clarify?"


def identify_gap(
    topic: Any,
    payload: Any,
    *,
    source: Any = "",
    event_meta: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Build a gap frame for a stimulus, or return ``identified=False``.

    The most important path is user-originated minimal/ambiguous text that would
    otherwise become deliberate silence.  The organ can also label sensory gaps
    so perception can request the right modality instead of guessing.
    """
    now_f = float(time.time() if now is None else now)
    topic_s = _clean_text(topic, limit=120)
    event_meta = event_meta or {}
    context = _context_packet(payload)
    modality = _topic_modality(topic_s, payload, context)
    text = extract_user_text(payload)
    tokens = extract_tokens(payload)
    meaningful = extract_meaningful_tokens(payload)
    uncertainty = extract_uncertainty(payload)
    statement_kind = extract_statement_kind(payload)
    selected_action = extract_selected_action(payload, event_meta)
    response_demand = extract_response_demand(payload)
    user_origin = is_user_originated_text(payload, source=source)
    paralinguistic = is_paralinguistic_signal(text, tokens)
    deliberate_silence = _bool(event_meta.get("deliberate_silence"), default=False)
    if not deliberate_silence and isinstance(payload, Mapping):
        deliberate_silence = _bool(_get_path(payload, "trigger", "deliberate_silence", default=False), default=False)

    evidence_need = _evidence_need_for_modality(topic_s, payload, modality)
    missing: list[str] = []
    gap_kind = ""
    closure_options: list[dict[str, Any]] = []
    response_demand_recommended = response_demand

    user_ambiguous = bool(
        user_origin
        and text
        and uncertainty >= GAP_UNCERTAINTY_THRESHOLD
        and (
            not meaningful
            or paralinguistic
            or statement_kind in {"minimal_statement", "fragment", "unknown"}
            or selected_action == "silence"
            or deliberate_silence
        )
    )

    if user_ambiguous:
        gap_kind = "intent_ambiguous"
        missing = ["intent", "desired_response"]
        if paralinguistic:
            missing.append("paralinguistic_meaning")
        gate = human_uplift_question_gate(payload, source=source)
        surface = _suggested_clarification_surface(text, paralinguistic=paralinguistic)
        closure_options.append(
            {
                "kind": "clarification_question",
                "safe": bool(gate.get("allowed")),
                "modality": "text",
                "suggested_surface": surface[:MAX_SURFACE_CHARS],
                "reason": "user_originated_ambiguity_needs_curiosity_before_silence",
            }
        )
        response_demand_recommended = max(response_demand, MINIMAL_SIGNAL_RESPONSE_DEMAND)
    elif evidence_need:
        gap_kind = f"{modality}_evidence_gap"
        missing = list(evidence_need.get("missing") or ["normalization"])
        closure_options.append(
            {
                "kind": "evidence_request",
                "safe": True,
                "modality": evidence_need.get("modality", modality),
                "suggested_action": evidence_need.get("suggested_action", "gather_more_evidence"),
                "question": evidence_need.get("question", "what evidence is missing?"),
                "reason": evidence_need.get("reason", "modality_needs_normalization"),
            }
        )
        response_demand_recommended = max(response_demand, 0.0)
    else:
        return {
            "schema": GAP_IDENTIFIED_SCHEMA,
            "identified": False,
            "source_topic": topic_s,
            "source_modality": modality,
            "reason": "no_gap_identified",
        }

    gate = human_uplift_question_gate(payload, source=source)
    blocks = list(gate.get("blocked_reasons") or [])
    silence_allowed = bool(blocks)
    if evidence_need and not user_ambiguous:
        # Non-speech evidence requests are allowed without making silence a
        # social answer.  The gap still exists, but no question may be owed.
        silence_allowed = False

    gap = {
        "schema": GAP_IDENTIFIED_SCHEMA,
        "identified": True,
        "gap_id": f"gap:{_stable_id([topic_s, text, gap_kind, selected_action, now_f // 5])}",
        "created_at": now_f,
        "source_topic": topic_s,
        "source": _clean_text(source, limit=80),
        "source_modality": modality,
        "gap_kind": gap_kind,
        "stimulus": {
            "text": text,
            "tokens": list(tokens)[:16],
            "meaningful_tokens": list(meaningful)[:16],
            "user_originated": user_origin,
            "paralinguistic": paralinguistic,
            "intentional_change": user_origin,
            "normalized": False if gap_kind else True,
        },
        "analysis": {
            "uncertainty": uncertainty,
            "statement_kind": statement_kind,
            "selected_action_seen": selected_action,
            "deliberate_silence_seen": deliberate_silence,
            "response_demand_seen": response_demand,
            "response_demand_recommended": response_demand_recommended,
        },
        "missing": missing[:MAX_MISSING_ITEMS],
        "evidence_need": evidence_need or {"needed": False},
        "closure_options": closure_options,
        "human_uplift_gate": gate,
        "silence_allowed": silence_allowed,
        "silence_reason": ";".join(gate.get("blocked_reason_text") or []) if silence_allowed else "",
        "principle": "silence_is_last_resort_questions_are_valid_unless_they_harm_uplift_or_unnecessary_self_integrity",
    }
    return gap


def build_clarification_need(gap: Mapping[str, Any]) -> dict[str, Any] | None:
    if not gap.get("identified"):
        return None
    for option in gap.get("closure_options", []) or []:
        if not isinstance(option, Mapping):
            continue
        if option.get("kind") != "clarification_question" or not option.get("safe"):
            continue
        surface = _clean_text(option.get("suggested_surface"), limit=MAX_SURFACE_CHARS)
        if not surface:
            continue
        return {
            "schema": CLARIFICATION_NEED_SCHEMA,
            "gap_id": gap.get("gap_id", ""),
            "gap_kind": gap.get("gap_kind", ""),
            "source_modality": gap.get("source_modality", ""),
            "question_surface": surface,
            "priority": CLARIFICATION_PRIORITY,
            "reason": option.get("reason", "gap_needs_clarification"),
            "silence_allowed": bool(gap.get("silence_allowed")),
            "missing": list(gap.get("missing") or []),
        }
    return None


def build_evidence_need(gap: Mapping[str, Any]) -> dict[str, Any] | None:
    if not gap.get("identified"):
        return None
    evidence_need = gap.get("evidence_need")
    if not isinstance(evidence_need, Mapping) or evidence_need.get("needed") is False:
        # The helper stores concrete fields directly when a modality request is
        # needed.  Absence or explicit needed=False means no perception request.
        return None
    modality = _clean_text(evidence_need.get("modality"), limit=40)
    if not modality:
        return None
    return {
        "schema": EVIDENCE_NEED_SCHEMA,
        "gap_id": gap.get("gap_id", ""),
        "gap_kind": gap.get("gap_kind", ""),
        "modality": modality,
        "question": _clean_text(evidence_need.get("question"), limit=MAX_SURFACE_CHARS),
        "suggested_action": _clean_text(evidence_need.get("suggested_action"), limit=120),
        "reason": _clean_text(evidence_need.get("reason"), limit=160),
        "priority": EVIDENCE_NEED_PRIORITY,
        "missing": list(evidence_need.get("missing") or gap.get("missing") or []),
    }


def build_speech_obligation(gap: Mapping[str, Any]) -> dict[str, Any] | None:
    need = build_clarification_need(gap)
    if not need:
        return None
    return {
        "schema": SPEECH_OBLIGATION_SCHEMA,
        "gap_id": gap.get("gap_id", ""),
        "obligation_kind": "clarification_question",
        "what_to_say": "ask_missing_intent_or_desired_response",
        "how_to_say": "short_safe_clarification_surface",
        "surface_options": [need.get("question_surface", "")],
        "minimum_surface_complete": True,
        "silence_allowed": bool(gap.get("silence_allowed")),
        "human_uplift_gate": gap.get("human_uplift_gate", {}),
        "priority": need.get("priority", CLARIFICATION_PRIORITY),
        "reason": need.get("reason", "gap_needs_clarification"),
    }


def build_gap_speech_payload(gap: Mapping[str, Any]) -> dict[str, Any] | None:
    obligation = build_speech_obligation(gap)
    if not obligation:
        return None
    surfaces = [s for s in (obligation.get("surface_options") or []) if _clean_text(s)]
    if not surfaces:
        return None
    text = _clean_text(surfaces[0], limit=MAX_SURFACE_CHARS)
    if not text:
        return None
    return {
        "schema": GAP_SPEECH_PAYLOAD_SCHEMA,
        "text": text,
        "channel": "textual",
        "transport": "textual",
        "suppress_tts": True,
        "kind": "gap_clarification",
        "gap_id": gap.get("gap_id", ""),
        "gap_kind": gap.get("gap_kind", ""),
        "response_obligation": obligation,
        "surface_complete": True,
        "reason": "gap_identifier_curiosity_clarification",
    }
