from __future__ import annotations

"""
Conversation / recollection review helpers for MicroBrain.

This module is intentionally small and deterministic.  It does not try to be a
full parser, and it does not decide truth.  It turns conversation turns,
assistant replies, and random recollection fragments into review anchors that a
trainer, dashboard, or later memory composer can inspect.

Law:
    Interaction speaks.
    Review explains the speech.
    Recollection may teach, but may not testify.
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping

CONNECTOR_WORDS = {
    "with", "to", "from", "for", "of", "in", "on", "at", "by", "over", "under",
    "about", "into", "onto", "through", "between", "because", "but", "and", "or",
}

QUESTION_WORDS = {"what", "why", "how", "where", "when", "who", "which", "can", "do", "does", "did", "is", "are", "will"}
CONCEPT_LABELS = {"question", "query", "unknown", "subject", "object", "gap", "inquiry", "memory", "work"}


def norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text or ""))


def new_anchor_id(prefix: str = "utt") -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def source_trust(source: str) -> str:
    src = str(source or "").lower()
    if src in {"thought_probe", "thought/probe", "recollection", "memory_drift", "idle_wander"}:
        return "low_internal_weather"
    if src in {"assistant", "native_responder", "demi", "act/speech"}:
        return "assistant_output"
    if src in {"user", "human", "textual", "cli", "mic"}:
        return "user_observed"
    return "observed"


def is_fragment(text: str) -> bool:
    token_count = len(words(text))
    if token_count == 0:
        return True
    if token_count == 1:
        return True
    n = norm_text(text)
    return n in {"what is", "do you", "can we", "with work"}


def repair_for_fragment(text: str, *, previous_user_text: str = "") -> tuple[str, str, str]:
    """Return (status, reason, repair_surface) for a likely fragment."""
    n = norm_text(text)
    toks = n.split()
    previous = norm_text(previous_user_text)

    if not toks:
        return "invalid_fragment", "empty_output", "question?"

    if len(toks) == 1:
        token = toks[0]
        if token in CONNECTOR_WORDS:
            return "invalid_fragment", "connector_without_object", f"{token} what?"
        if token == "question":
            if "subject" in previous:
                return "repair_handle", "bare_question_label_missing_subject", "subject?"
            return "repair_handle", "bare_concept_label", "question?"
        if token == "subject":
            return "repair_handle", "bare_slot_label", "subject?"
        if token == "object":
            return "repair_handle", "bare_slot_label", "object?"
        if token in CONCEPT_LABELS:
            return "repair_handle", "bare_concept_label", f"{token}?"
        return "repair_handle", "single_word_needs_context", f"{token}?"

    if toks == ["what", "is"]:
        # If the user asked about work, expose the missing/retrieval target.
        if "work" in previous:
            return "invalid_fragment", "incomplete_question_echo", "work?"
        return "invalid_fragment", "incomplete_question_echo", "what is what?"

    if toks and toks[0] in QUESTION_WORDS and not str(text).strip().endswith("?"):
        return "repair_handle", "unfinished_question_surface", str(text).strip() + "?"

    return "ok", "", ""


def infer_user_frame(text: str) -> dict[str, Any]:
    n = norm_text(text)
    frame = "statement"
    obligations: list[str] = []
    topic = ""

    if n.endswith("?") or str(text or "").strip().endswith("?") or (n.split()[:2] in [["what", "is"], ["what", "are"]]):
        frame = "question"
        obligations = ["answer_or_clarify"]
    if re.search(r"\bwhat is my work\b", n):
        frame = "user_fact_query"
        topic = "work"
        obligations = ["memory_lookup", "answer_subject_fact"]
    elif re.search(r"\bi work at\b", n):
        frame = "user_fact_teaching_claim"
        topic = "work"
        obligations = ["acknowledgement", "memory_candidate"]
    elif re.search(r"\bhalfway\b.*\bwo?k\b|\bhalfway\b.*\bwork\b", n):
        frame = "social_progress_update"
        topic = "work"
        obligations = ["acknowledgement", "topic_reference"]
    elif re.search(r"\bguess what\b", n):
        frame = "social_prompt"
        obligations = ["invite_disclosure"]
    elif re.search(r"\bquestion\b", n) and re.search(r"\bsubject\b", n):
        frame = "language_repair_teaching"
        topic = "question_subject"
        obligations = ["surface_missing_slot"]

    return {"frame": frame, "topic": topic, "response_obligations": obligations}


def analyze_anchor(
    *,
    text: str,
    source: str,
    role: str = "observed",
    previous_user_text: str = "",
    event_topic: str = "",
    correlation_id: str = "",
    ts: float | None = None,
) -> dict[str, Any]:
    raw = str(text or "").strip()
    n = norm_text(raw)
    frame = infer_user_frame(raw) if role == "user" else {"frame": "output_or_recollection", "topic": "", "response_obligations": []}
    status, reason, repair = repair_for_fragment(raw, previous_user_text=previous_user_text)
    if role == "user" and raw:
        # User fragments are observations, not bad assistant outputs.
        status = "observed"
        reason = ""
        repair = ""

    trust = source_trust(source)
    memory_eligible = False
    promotion_requires = ["trainer_confirmation", "repeat_useful_pattern"]
    if role == "user" and frame.get("frame") in {"user_fact_teaching_claim"}:
        memory_eligible = True
        promotion_requires = ["review", "trainer_or_user_confirmation"]
    elif role == "assistant" and status == "ok":
        memory_eligible = True
        promotion_requires = ["review", "successful_turn_satisfaction"]

    return {
        "anchor_id": new_anchor_id("utt" if role != "recollection" else "rec"),
        "ts": float(time.time() if ts is None else ts),
        "source": source,
        "role": role,
        "event_topic": event_topic,
        "correlation_id": correlation_id,
        "text": raw,
        "norm": n,
        "trust": trust,
        "frame": frame.get("frame", "output_or_recollection"),
        "topic": frame.get("topic", ""),
        "response_obligations": list(frame.get("response_obligations", []) or []),
        "status": status,
        "reason": reason,
        "repair_surface": repair,
        "memory_eligible": memory_eligible,
        "promotion_requires": promotion_requires,
        "epistemic_status": "observed_not_truth" if role != "recollection" else "low_trust_internal_weather",
    }


def review_pair(user_anchor: Mapping[str, Any] | None, reply_anchor: Mapping[str, Any] | None) -> dict[str, Any]:
    user_anchor = user_anchor if isinstance(user_anchor, Mapping) else {}
    reply_anchor = reply_anchor if isinstance(reply_anchor, Mapping) else {}
    user_frame = str(user_anchor.get("frame", "") or "")
    obligations = list(user_anchor.get("response_obligations", []) or [])
    reply_status = str(reply_anchor.get("status", "") or "")
    reply_text = str(reply_anchor.get("text", "") or "")
    reply_reason = str(reply_anchor.get("reason", "") or "")

    satisfied = False
    missing: list[str] = []
    repair = str(reply_anchor.get("repair_surface", "") or "")

    if not reply_text:
        reply_status = "missing_reply"
        reply_reason = "no_assistant_output_to_review"
        missing = obligations or ["response"]
        repair = "question?"
    elif reply_status in {"invalid_fragment", "repair_handle"}:
        satisfied = False
        missing = obligations or ["complete_surface"]
    elif user_frame == "social_progress_update":
        rnorm = norm_text(reply_text)
        satisfied = bool(("work" in rnorm or "halfway" in rnorm) and len(rnorm.split()) >= 2)
        if not satisfied:
            missing = ["acknowledgement", "topic_reference"]
            repair = repair or "Nice, halfway done with work."
    elif user_frame == "user_fact_query":
        rnorm = norm_text(reply_text)
        satisfied = bool("work" in rnorm and len(rnorm.split()) >= 4)
        if not satisfied:
            missing = ["memory_lookup", "answer_subject_fact"]
            repair = repair or "work?"
    elif obligations:
        satisfied = reply_status == "ok"
        if not satisfied:
            missing = obligations
    else:
        satisfied = reply_status == "ok"

    return {
        "review_id": new_anchor_id("review"),
        "ts": time.time(),
        "kind": "conversation_turn_review",
        "user_anchor_id": str(user_anchor.get("anchor_id", "") or ""),
        "reply_anchor_id": str(reply_anchor.get("anchor_id", "") or ""),
        "user_frame": user_frame,
        "expected_obligations": obligations,
        "reply_text": reply_text,
        "reply_status": reply_status,
        "reply_reason": reply_reason,
        "satisfied_turn": satisfied,
        "missing_slots": missing,
        "repair_surface": repair,
        "memory_eligible": bool(satisfied and reply_text),
        "promotion_requires": ["trainer_confirmation"] if not satisfied else ["successful_reuse"],
        "epistemic_status": "review_candidate_not_truth",
    }


def trim_ring(items: Iterable[Mapping[str, Any]], limit: int = 64) -> list[dict[str, Any]]:
    clean = [dict(item) for item in items if isinstance(item, Mapping)]
    if len(clean) > limit:
        clean = clean[-limit:]
    return clean
