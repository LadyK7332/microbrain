from __future__ import annotations

from microbrain.orchestrator.neuron_base import Event
from microbrain.speech_egress_guard import (
    classify_speech_fault,
    guard_speech_event,
    observe_speech_context,
    repair_speech,
)


def test_allows_guess_what_reply():
    assert classify_speech_fault("what", last_user_text="Demi, guess what")["status"] == "allow"


def test_repairs_connector_fragment():
    decision = classify_speech_fault("with", last_user_text="I'm halfway done with work!")
    assert decision["status"] == "repair"
    assert decision["reason"] == "connector_without_object"
    assert repair_speech("with", last_user_text="Tell me with") == "with what?"


def test_repairs_progress_update_context():
    event = Event(topic="act/speech", payload={"text": "with"}, source="test")
    store = {}
    observe_speech_context(Event(topic="percept/text", payload={"text": "I'm halfway done with wok!"}, source="user"), store)
    guarded = guard_speech_event(event, store)
    assert guarded is not None
    assert guarded.payload["text"] == "Nice, halfway done with work."
    assert guarded.meta["speech_egress_guard"]["reason"] == "connector_without_object"


def test_remembers_work_hint_and_repairs_work_query():
    store = {}
    observe_speech_context(Event(topic="percept/text", payload={"text": "I work at EMS as a callcenter agent"}, source="user"), store)
    observe_speech_context(Event(topic="percept/text", payload={"text": "what is my work"}, source="user"), store)
    guarded = guard_speech_event(Event(topic="act/speech", payload={"text": "what is"}, source="test"), store)
    assert guarded is not None
    assert guarded.payload["text"] == "You work at EMS as callcenter agent."


def test_repairs_bare_question_with_subject_context():
    store = {}
    observe_speech_context(Event(topic="percept/text", payload={"text": "question, subject?"}, source="user"), store)
    guarded = guard_speech_event(Event(topic="act/speech", payload={"text": "question"}, source="test"), store)
    assert guarded is not None
    assert guarded.payload["text"] == "subject?"


def test_drops_empty_speech():
    assert guard_speech_event(Event(topic="act/speech", payload={"text": ""}, source="test"), {}) is None


def test_allows_unclassified_one_word_name():
    assert classify_speech_fault("EMS", last_user_text="where do I work?")["status"] == "allow"


def test_repairs_question_prefix_echo():
    decision = classify_speech_fault("what is", last_user_text="what is my work")
    assert decision["status"] == "repair"
    assert decision["reason"] in {"partial_phrase", "question_prefix_echo", "work_query_not_satisfied"}


def test_guard_records_faults():
    store = {}
    observe_speech_context(Event(topic="percept/text", payload={"text": "question, subject?"}, source="user"), store)
    guard_speech_event(Event(topic="act/speech", payload={"text": "question"}, source="test"), store)
    assert store["speech_egress:last_decision"]["status"] == "repaired"
    assert store["speech_egress:faults"]


def test_non_speech_passes_through():
    event = Event(topic="thought/internal", payload={"text": "with"}, source="test")
    assert guard_speech_event(event, {}) is event
