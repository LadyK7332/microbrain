from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from microbrain.conversation_recollection_review import analyze_anchor, review_pair
from microbrain.neurons.conversation_recollection_review_neuron import ConversationRecollectionReviewNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


def test_connector_fragment_becomes_repair_handle():
    anchor = analyze_anchor(text="with", source="assistant", role="assistant", previous_user_text="I'm halfway done with wok!")
    assert anchor["status"] == "invalid_fragment"
    assert anchor["reason"] == "connector_without_object"
    assert anchor["repair_surface"] == "with what?"
    assert anchor["memory_eligible"] is False


def test_bare_question_after_subject_teaching_becomes_subject_question():
    anchor = analyze_anchor(text="question", source="assistant", role="assistant", previous_user_text="question, subject?")
    assert anchor["status"] == "repair_handle"
    assert anchor["reason"] == "bare_question_label_missing_subject"
    assert anchor["repair_surface"] == "subject?"


def test_what_is_work_collapse_exposes_work_target():
    anchor = analyze_anchor(text="what is", source="assistant", role="assistant", previous_user_text="what is my work")
    assert anchor["status"] == "invalid_fragment"
    assert anchor["reason"] == "incomplete_question_echo"
    assert anchor["repair_surface"] == "work?"


def test_recollection_is_low_trust_and_not_memory_eligible():
    anchor = analyze_anchor(text="city in fact", source="thought_probe", role="recollection")
    assert anchor["trust"] == "low_internal_weather"
    assert anchor["epistemic_status"] == "low_trust_internal_weather"
    assert anchor["memory_eligible"] is False


def test_social_progress_pair_detects_failed_with_reply():
    user = analyze_anchor(text="I'm halfway done with wok!", source="user", role="user")
    reply = analyze_anchor(text="with", source="assistant", role="assistant", previous_user_text=user["text"])
    review = review_pair(user, reply)
    assert review["user_frame"] == "social_progress_update"
    assert review["satisfied_turn"] is False
    assert "acknowledgement" in review["missing_slots"]
    assert review["repair_surface"] == "with what?"


def test_social_progress_pair_accepts_topic_ack():
    user = analyze_anchor(text="I'm halfway done with work!", source="user", role="user")
    reply = analyze_anchor(text="Nice, halfway done with work.", source="assistant", role="assistant", previous_user_text=user["text"])
    review = review_pair(user, reply)
    assert review["satisfied_turn"] is True
    assert review["memory_eligible"] is True


@dataclass
class FakeCtx:
    kv: dict[str, Any] = field(default_factory=dict)
    emitted: list[Event] = field(default_factory=list)

    async def emit(self, event: Event) -> None:
        self.emitted.append(event)

    async def log_debug(self, msg: str, **kwargs: Any) -> None:
        pass

    async def log_info(self, msg: str, **kwargs: Any) -> None:
        pass

    async def log_warn(self, msg: str, **kwargs: Any) -> None:
        pass

    async def log_error(self, msg: str, **kwargs: Any) -> None:
        pass

    async def get_kv(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    async def set_kv(self, key: str, value: Any) -> None:
        self.kv[key] = value


def make_neuron() -> ConversationRecollectionReviewNeuron:
    cfg = NeuronConfig(
        name="conversation_recollection_review_neuron",
        subscribed_topics=["reason/request", "act/speech", "thought/probe"],
        output_topics=["review/utterance_anchor", "review/conversation_turn", "review/repair_candidate", "review/recollection_anchor"],
    )
    return ConversationRecollectionReviewNeuron(cfg)


def test_neuron_reviews_user_then_bad_reply():
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        user_outputs = list(await neuron.process(Event(topic="reason/request", payload={"text": "I'm halfway done with wok!", "channel": "repl"}, source="user", correlation_id="c1"), ctx))
        assert user_outputs[0].topic == "review/utterance_anchor"
        reply_outputs = list(await neuron.process(Event(topic="act/speech", payload={"text": "with"}, source="native_responder", correlation_id="c1"), ctx))
        topics = [ev.topic for ev in reply_outputs]
        assert "review/conversation_turn" in topics
        assert "review/repair_candidate" in topics
        review = ctx.kv["conversation_review:last_turn_review"]
        assert review["satisfied_turn"] is False
        assert review["reply_reason"] == "connector_without_object"
    asyncio.run(run())


def test_neuron_reviews_recollection_without_memory_eligibility():
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        outputs = list(await neuron.process(Event(topic="thought/probe", payload={"text": "question"}, source="thought_probe"), ctx))
        assert outputs[0].topic == "review/recollection_anchor"
        anchor = ctx.kv["conversation_review:last_recollection_anchor"]
        assert anchor["memory_eligible"] is False
        assert "daylight_review" in anchor["promotion_requires"]
    asyncio.run(run())
