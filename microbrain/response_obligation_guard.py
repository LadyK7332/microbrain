from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# This helper is deliberately narrow.  It is not a second answer composer.
# It only asks: "does this final surface satisfy the turn, or is it an orphan
# fragment that should be converted into a question handle?"
MINIMUM_QUESTION_FALLBACK = "question?"
DEFAULT_MAX_REPAIR_WORDS = 9

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

RESPONSE_OBLIGATION_SCHEMA = "response_obligation_guard.v1"
QUESTION_MARK = "?"

ORPHAN_SINGLE_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "question",  # label-only, unless shaped as "question?"
        "subject",   # label-only, unless shaped as "subject?"
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

ORPHAN_PHRASES = frozenset(
    {
        "what is",
        "what are",
        "do you",
        "can you",
        "is my",
        "my work",
        "with work",
        "with wok",
        "question subject",
        "subject question",
    }
)

CONTENT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "done",
        "for",
        "from",
        "have",
        "hey",
        "how",
        "i",
        "im",
        "i'm",
        "if",
        "in",
        "is",
        "it",
        "like",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

VOBJ_RE = re.compile(r"\bvobj[:_\-]?[0-9A-Za-z]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class ResponseObligationResult:
    schema: str
    text: str
    action: str
    reason: str
    turn_type: str
    subject: str
    required_slots: tuple[str, ...]
    original_text: str
    proposed_text: str
    repaired: bool
    ts: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_slots"] = list(self.required_slots)
        return data


def _norm(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _clean_subject(subject: Any) -> str:
    text = str(subject or "").strip().strip(".?!,;: ")
    if not text:
        return ""
    # Common live-typing typo from line testing.  Keep this tiny and explicit;
    # this helper must not become a general spellchecker.
    if text.lower() == "wok":
        return "work"
    return text[:80]


def _question_surface(subject: Any) -> str:
    clean = _clean_subject(subject)
    if not clean:
        return MINIMUM_QUESTION_FALLBACK
    if clean.endswith(QUESTION_MARK):
        return clean
    return f"{clean}?"


def _ends_as_question(text: Any) -> bool:
    return str(text or "").strip().endswith(QUESTION_MARK)


def _looks_like_valid_question_handle(text: Any) -> bool:
    raw = str(text or "").strip()
    if not raw.endswith(QUESTION_MARK):
        return False
    before = raw[:-1].strip()
    return bool(before)


def _extract_subject_from_payload(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in (
        "question_subject",
        "subject",
        "target",
        "object",
        "object_id",
        "track_id",
        "selected_track_id",
        "vobj_id",
        "visual_track_id",
        "visual_object_id",
        "focus",
        "focus_ref",
        "ref",
    ):
        value = payload.get(key)
        if value:
            return _clean_subject(value)
    for nested_key in ("pressure", "evidence_mesh", "visual_attention_ref", "recognition_claim"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            value = _extract_subject_from_payload(nested)
            if value:
                return value
    return ""


def _extract_subject(text: Any, payload: Mapping[str, Any] | None = None) -> str:
    payload_subject = _extract_subject_from_payload(payload)
    if payload_subject:
        return payload_subject

    raw = str(text or "")
    vobj = VOBJ_RE.search(raw)
    if vobj:
        return vobj.group(0)

    norm = _norm(raw)
    if re.search(r"\b(subject|object|target)\b", norm):
        # For training phrases like "question, subject?" the missing slot is
        # more useful than the high-level label "question".
        if "subject" in norm:
            return "subject"
        if "object" in norm:
            return "object"
        if "target" in norm:
            return "target"

    if re.search(r"\b(work|job|ems|callcenter|call center|wok)\b", norm):
        return "work"
    if re.search(r"\bquestion|query|inquiry|unknown\b", norm):
        return "question"

    toks = _tokens(raw)
    for token in reversed(toks):
        if token not in CONTENT_STOPWORDS and len(token) > 1:
            return _clean_subject(token)
    return ""


def _classify_turn(text: Any, payload: Mapping[str, Any] | None = None) -> tuple[str, tuple[str, ...], str]:
    raw = str(text or "").strip()
    norm = _norm(raw)
    subject = _extract_subject(raw, payload)

    if not raw:
        return "empty", tuple(), subject
    if "guess what" in norm:
        return "prompted_followup", ("followup_question",), "what"
    if norm in {"?", "question", "question subject", "subject question"} or "question subject" in norm:
        return "question_shape_training", ("question_subject",), subject or "subject"
    if re.fullmatch(r"what is my (work|job)", norm):
        return "user_fact_query", ("answer_known_fact",), "work"
    if _ends_as_question(raw):
        return "question", ("answer_or_question_handle",), subject or "question"
    if re.search(r"\bhalfway done with\b", norm):
        return "social_progress_update", ("acknowledge", "topic_reference"), subject or "work"
    if re.search(r"\byes with (work|wok)\b", norm):
        return "correction_ack", ("acknowledge", "topic_reference"), "work"
    return "statement", tuple(), subject


def _word_count(text: Any) -> int:
    return len(_tokens(text))


def _is_orphan_fragment(reply: Any, *, user_text: Any, turn_type: str) -> tuple[bool, str]:
    raw = str(reply or "").strip()
    norm = _norm(raw)
    user_norm = _norm(user_text)
    if not norm:
        return True, "empty_reply"

    # "what" is valid as the social follow-up to "guess what".
    if turn_type == "prompted_followup" and norm == "what":
        return False, "valid_prompted_followup"

    if _looks_like_valid_question_handle(raw):
        return False, "valid_question_handle"

    if norm in ORPHAN_SINGLE_WORDS:
        return True, "orphan_single_word"
    if norm in ORPHAN_PHRASES:
        return True, "orphan_phrase"

    if _word_count(raw) <= 2 and norm and user_norm:
        # Echoed fragments like "what is" or "with" are not answers.  This is
        # intentionally lenient for real short acknowledgements such as "yes".
        if norm in user_norm and norm != user_norm:
            return True, "input_fragment_echo"

    if turn_type in {"question", "user_fact_query"} and norm in {"yes", "no", "maybe"}:
        # A bare yes/no can be valid for yes/no questions, but not for content
        # questions like "what is my work?".
        if user_norm.startswith(("what ", "who ", "where ", "when ", "why ", "how ")):
            return True, "bare_answer_to_content_question"

    if turn_type in {"question", "user_fact_query", "question_shape_training"} and _word_count(raw) <= 2:
        if not _ends_as_question(raw) and norm in {"question", "subject", "work", "job", "what is"}:
            return True, "label_without_question_surface"

    return False, "valid"


def _repair_reply(
    *,
    user_text: Any,
    proposed_reply: Any,
    turn_type: str,
    subject: str,
    required_slots: Sequence[str],
) -> tuple[str, str]:
    raw = str(user_text or "").strip()
    norm = _norm(raw)
    proposed_norm = _norm(proposed_reply)

    if turn_type == "prompted_followup":
        return "what?", "repair_prompted_followup"

    if turn_type == "question_shape_training":
        return _question_surface(subject or "subject"), "repair_question_shape_training"

    if turn_type == "social_progress_update":
        # Minimal but actually conversational.  This is a repair for fragment
        # collapse, not a broad canned personality layer.
        if "halfway done" in norm and ("work" in norm or "wok" in norm):
            return "Nice, halfway done with work.", "repair_social_progress_ack"
        return _question_surface(subject), "repair_social_update_subject_question"

    if turn_type == "correction_ack":
        return "work?", "repair_correction_ack_subject_question"

    if proposed_norm in {"question", "subject", "object", "target", "work", "job"}:
        # Preserve the label, but shape it as a question handle.
        return _question_surface(proposed_norm), "repair_label_to_question_handle"

    if turn_type in {"question", "user_fact_query"}:
        return _question_surface(subject or "question"), "repair_question_subject_handle"

    if subject:
        return _question_surface(subject), "repair_subject_handle"
    return MINIMUM_QUESTION_FALLBACK, "repair_minimum_question"


def guard_native_response(
    *,
    user_text: Any,
    proposed_reply: Any,
    shape: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    raw_meta: Mapping[str, Any] | None = None,
    syntax_guidance: Mapping[str, Any] | None = None,
) -> ResponseObligationResult:
    """Accept, repair, or drop a final native response surface.

    This guard is designed for the exact failure seen in line testing:
    function-word fragments such as ``with`` and stem fragments such as
    ``what is`` escaping as final speech.  It does not claim to understand the
    answer; it only checks whether the surface satisfies the immediate turn.
    """
    payload = payload if isinstance(payload, Mapping) else {}
    turn_type, required_slots, subject = _classify_turn(user_text, payload)
    proposed = str(proposed_reply or "").strip()
    invalid, invalid_reason = _is_orphan_fragment(proposed, user_text=user_text, turn_type=turn_type)

    if not invalid:
        return ResponseObligationResult(
            schema=RESPONSE_OBLIGATION_SCHEMA,
            text=proposed,
            action="accept",
            reason=invalid_reason,
            turn_type=turn_type,
            subject=subject,
            required_slots=tuple(required_slots),
            original_text=str(user_text or ""),
            proposed_text=proposed,
            repaired=False,
            ts=time.time(),
        )

    # Statements without response obligations may still stay silent.  Do not
    # force chatter just because the native composer had no reply.
    if turn_type == "statement" and not proposed:
        return ResponseObligationResult(
            schema=RESPONSE_OBLIGATION_SCHEMA,
            text="",
            action="drop",
            reason="empty_statement_no_obligation",
            turn_type=turn_type,
            subject=subject,
            required_slots=tuple(required_slots),
            original_text=str(user_text or ""),
            proposed_text=proposed,
            repaired=False,
            ts=time.time(),
        )

    repaired_text, repair_reason = _repair_reply(
        user_text=user_text,
        proposed_reply=proposed,
        turn_type=turn_type,
        subject=subject,
        required_slots=required_slots,
    )
    return ResponseObligationResult(
        schema=RESPONSE_OBLIGATION_SCHEMA,
        text=repaired_text,
        action="repair",
        reason=f"{invalid_reason}:{repair_reason}",
        turn_type=turn_type,
        subject=subject,
        required_slots=tuple(required_slots),
        original_text=str(user_text or ""),
        proposed_text=proposed,
        repaired=True,
        ts=time.time(),
    )
