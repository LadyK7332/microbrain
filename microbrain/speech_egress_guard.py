from __future__ import annotations

"""Final-mouth guard for MB speech events.

This module is deliberately small and dependency-light.  It is meant to sit at
MicroBrain's final speech boundary, after any neuron has emitted ``act/speech``
but before the UI/mouth sees that event.

The guard does not try to be a full language brain.  It catches the worst
failure mode seen in line-live runs: orphan fragments such as ``with``, ``what
is``, and bare labels such as ``question`` escaping as if they satisfied the
turn.

Tiny law:
    A word is not an answer unless it satisfies the turn.
"""

import re
import time
from typing import Any, Dict, Mapping, MutableMapping, Optional

from microbrain.orchestrator.neuron_base import Event

_CONTEXT_KEY = "speech_egress:context"
_LAST_DECISION_KEY = "speech_egress:last_decision"
_FAULTS_KEY = "speech_egress:faults"

_CONNECTOR_FRAGMENTS = {
    "with",
    "to",
    "for",
    "from",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "and",
    "or",
    "but",
    "if",
    "because",
    "than",
    "then",
    "that",
    "this",
    "these",
    "those",
}

_BARE_LABEL_FRAGMENTS = {
    "question",
    "subject",
    "object",
    "target",
    "relation",
    "query",
    "unknown",
    "inquiry",
    "learning",
    "conversation",
    "gap",
    "answer",
    "reply",
    "work",  # one-word fallback for "what is my work" is usually not enough
}

_PARTIAL_PHRASES = {
    "what is",
    "what are",
    "who is",
    "where is",
    "when is",
    "why is",
    "how is",
    "do you",
    "can we",
    "can you",
    "i am",
    "i'm",
    "we can",
}

_ALLOWED_SHORT = {
    "yes",
    "no",
    "ok",
    "okay",
    "sure",
    "yep",
    "yeah",
    "nope",
    "what",
    "why",
    "who",
    "where",
    "when",
    "how",
    "hello",
    "hey",
    "hi",
    "thanks",
    "ty",
}


def _norm(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _text_from_payload(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("text", "") or payload.get("message", "") or "").strip()
    return str(payload or "").strip()


def _set_payload_text(event: Event, text: str) -> Event:
    if isinstance(event.payload, dict):
        event.payload = dict(event.payload)
        event.payload["text"] = text
    else:
        event.payload = {"text": text}
    return event


def _copy_meta(event: Event) -> Dict[str, Any]:
    return dict(event.meta or {}) if isinstance(event.meta, Mapping) else {}


def _context(store: MutableMapping[str, Any]) -> Dict[str, Any]:
    ctx = store.get(_CONTEXT_KEY)
    if not isinstance(ctx, dict):
        ctx = {}
        store[_CONTEXT_KEY] = ctx
    return ctx


def _remember_user_work(ctx: MutableMapping[str, Any], text: str) -> None:
    """Capture simple user work declarations for repair of work queries.

    This is not durable truth.  It is a local, reviewable speech-context hint so
    the final guard can repair obvious failures like answering "what is my work"
    with "what is" or "work".
    """
    raw = str(text or "").strip()
    norm = _norm(raw)
    match = re.search(r"\bi\s+work\s+at\s+(.+?)(?:\s+as\s+(?:a|an)?\s*(.+))?$", raw, flags=re.I)
    if not match:
        return
    org = re.sub(r"[.!?]+$", "", str(match.group(1) or "").strip())
    role = re.sub(r"[.!?]+$", "", str(match.group(2) or "").strip()) if match.group(2) else ""
    if not org:
        return
    phrase = org
    if role:
        phrase = f"{org} as {role}"
    ctx["user_work_hint"] = phrase
    ctx["user_work_hint_ts"] = time.time()
    ctx["user_work_source"] = raw
    ctx["user_work_source_norm"] = norm


def observe_speech_context(event: Event, store: MutableMapping[str, Any]) -> None:
    """Observe context that can help final-mouth repairs later.

    The guard only keeps small, recent hints.  It does not promote anything to
    durable memory and should never be treated as truth by itself.
    """
    if not isinstance(event, Event):
        return
    ctx = _context(store)
    topic = str(event.topic or "")
    text = _text_from_payload(event.payload)
    if not text:
        return

    if topic in {"percept/text", "reason/request", "user/text", "input/text"}:
        # reason/request can be emitted by an internal stage; keep only likely
        # direct user material unless no better context exists.
        source = str(event.source or "").lower()
        meta = event.meta if isinstance(event.meta, Mapping) else {}
        raw_source = ""
        if isinstance(event.payload, Mapping):
            raw_source = str(event.payload.get("source", "") or "").lower()
        userish = (
            topic != "reason/request"
            or source in {"user", "repl", "textual", "ui", "cli"}
            or raw_source in {"user", "repl", "textual", "ui", "cli"}
            or meta.get("source") in {"user", "repl", "textual", "ui", "cli"}
        )
        if userish:
            ctx["last_user_text"] = text
            ctx["last_user_norm"] = _norm(text)
            ctx["last_user_ts"] = time.time()
            _remember_user_work(ctx, text)
        return

    if topic == "act/speech":
        ctx["last_speech_seen_ts"] = time.time()


def _looks_like_work_query(norm_user: str) -> bool:
    if not norm_user:
        return False
    return norm_user in {"what is my work", "whats my work", "what is my job", "whats my job"} or (
        "my work" in norm_user and any(q in norm_user for q in {"what is", "whats", "what's"})
    )


def _looks_like_progress_update(norm_user: str) -> bool:
    if not norm_user:
        return False
    return (
        ("halfway" in norm_user or "half way" in norm_user)
        and ("work" in norm_user or "wok" in norm_user)
    )


def _question_subject_repair(norm_user: str) -> str:
    if "subject" in norm_user:
        return "subject?"
    if "object" in norm_user or "target" in norm_user:
        return "object?"
    if "question" in norm_user or "query" in norm_user or "inquiry" in norm_user:
        return "question?"
    return "question?"


def classify_speech_fault(reply_text: str, *, last_user_text: str = "") -> Dict[str, Any]:
    """Classify whether a speech text is likely an orphan fragment.

    Returns a decision dict.  ``status`` is one of ``allow`` or ``repair``.
    """
    raw = str(reply_text or "").strip()
    norm = _norm(raw)
    user_norm = _norm(last_user_text)
    words = norm.split()

    if not norm:
        return {"status": "drop", "reason": "empty_speech"}

    if norm in _ALLOWED_SHORT:
        # "what" is legitimate after "guess what".  One-word yes/no/etc are
        # also valid conversationally, so do not overcorrect them.
        return {"status": "allow", "reason": "allowed_short"}

    if len(words) == 1:
        if norm in _CONNECTOR_FRAGMENTS:
            return {"status": "repair", "reason": "connector_without_object", "missing_slot": "object"}
        if norm in _BARE_LABEL_FRAGMENTS:
            return {"status": "repair", "reason": "bare_label_fragment", "missing_slot": "subject"}
        if _looks_like_work_query(user_norm):
            return {"status": "repair", "reason": "work_query_not_satisfied", "missing_slot": "known_user_work"}
        # Unknown one-word replies are suspicious but not always wrong.  Do not
        # rewrite proper names / unknown labels here; let review sidecar judge.
        return {"status": "allow", "reason": "unclassified_one_word"}

    if norm in _PARTIAL_PHRASES or any(norm == phrase for phrase in _PARTIAL_PHRASES):
        return {"status": "repair", "reason": "partial_phrase", "missing_slot": "completion"}

    if len(words) <= 3 and user_norm and norm and user_norm.startswith(norm):
        return {"status": "repair", "reason": "question_prefix_echo", "missing_slot": "answer"}

    if len(words) <= 2 and _looks_like_work_query(user_norm):
        return {"status": "repair", "reason": "work_query_not_satisfied", "missing_slot": "known_user_work"}

    return {"status": "allow", "reason": "looks_complete"}


def repair_speech(reply_text: str, *, last_user_text: str = "", context: Mapping[str, Any] | None = None) -> str:
    ctx = context if isinstance(context, Mapping) else {}
    norm_reply = _norm(reply_text)
    user_norm = _norm(last_user_text)

    if _looks_like_work_query(user_norm):
        work_hint = str(ctx.get("user_work_hint", "") or "").strip()
        if work_hint:
            return f"You work at {work_hint}."
        return "Your work?"

    if _looks_like_progress_update(user_norm):
        return "Nice, halfway done with work."

    if norm_reply in _CONNECTOR_FRAGMENTS:
        return f"{norm_reply} what?"

    if norm_reply in {"question", "query", "inquiry", "unknown"}:
        return _question_subject_repair(user_norm)

    if norm_reply in {"subject", "object", "target", "relation"}:
        return f"{norm_reply}?"

    if norm_reply.startswith("what is"):
        return "what is what?"

    if norm_reply.startswith("can we"):
        return "can we what?"

    if norm_reply.startswith("do you"):
        return "do I what?"

    if norm_reply:
        return f"{reply_text.strip()}?"
    return "question?"


def guard_speech_event(event: Event, store: MutableMapping[str, Any]) -> Optional[Event]:
    """Repair or drop a final ``act/speech`` event before it reaches the mouth."""
    if not isinstance(event, Event):
        return None
    if str(event.topic or "") != "act/speech":
        return event

    ctx = _context(store)
    text = _text_from_payload(event.payload)
    last_user = str(ctx.get("last_user_text", "") or "")
    decision = classify_speech_fault(text, last_user_text=last_user)
    decision_record = {
        "ts": time.time(),
        "source": event.source,
        "original_text": text,
        "last_user_text": last_user,
        **decision,
    }

    if decision.get("status") == "allow":
        store[_LAST_DECISION_KEY] = decision_record
        return event

    if decision.get("status") == "drop":
        store[_LAST_DECISION_KEY] = decision_record
        faults = list(store.get(_FAULTS_KEY, []) or [])
        faults.append(decision_record)
        store[_FAULTS_KEY] = faults[-64:]
        return None

    repaired = repair_speech(text, last_user_text=last_user, context=ctx)
    meta = _copy_meta(event)
    meta["speech_egress_guard"] = {
        "status": "repaired",
        "reason": decision.get("reason"),
        "missing_slot": decision.get("missing_slot"),
        "original_text": text,
        "repaired_text": repaired,
        "last_user_text": last_user,
    }
    meta.setdefault("kind", "guarded_speech")
    event.meta = meta
    _set_payload_text(event, repaired)

    decision_record["status"] = "repaired"
    decision_record["repaired_text"] = repaired
    store[_LAST_DECISION_KEY] = decision_record
    faults = list(store.get(_FAULTS_KEY, []) or [])
    faults.append(decision_record)
    store[_FAULTS_KEY] = faults[-64:]
    return event
