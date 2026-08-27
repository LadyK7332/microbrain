from __future__ import annotations

import pytest

from microbrain.neurons.gap_identifier_neuron import GapIdentifierNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    pass


def _event():
    return Event(
        topic="hypothesis/action_committed",
        source="desire_trigger_neuron",
        correlation_id="abc123",
        meta={"selected_action": "silence", "deliberate_silence": True},
        payload={
            "context": {
                "constraints": {"allow_babble": True, "crisis_mode": False},
                "drives": {"boredom": {"same_user_repetitions": 0}},
                "input": {
                    "channel": "textual",
                    "source": "ui",
                    "text": "o.o",
                    "tokens": ["o", "o"],
                    "meaningful_tokens": [],
                    "raw_meta": {"source": "ui", "transport_source": "ui", "frontend": "dashboard"},
                },
            },
            "hypothesis": {
                "pattern_analysis": {
                    "uncertainty": 0.84,
                    "statement_kind": "minimal_statement",
                    "meaningful_tokens": [],
                    "tokens": ["o", "o"],
                },
                "recommended_action": "silence",
                "response_demand": 0.0,
                "should_respond": False,
            },
            "trigger": {"deliberate_silence": True, "recommended_action": "silence", "response_demand": 0.0},
        },
    )


@pytest.mark.asyncio
async def test_neuron_emits_gap_obligation_and_clarification_speech():
    neuron = GapIdentifierNeuron(
        NeuronConfig(
            name="gap_identifier_neuron",
            subscribed_topics=["hypothesis/action_committed"],
            output_topics=[],
        )
    )
    outputs = list(await neuron.process(_event(), DummyCtx()))
    topics = [event.topic for event in outputs]
    assert "cognition/gap_identified" in topics
    assert "cognition/clarification_need" in topics
    assert "speech/response_obligation" in topics
    assert "act/speech" in topics
    speech = next(event for event in outputs if event.topic == "act/speech")
    assert speech.payload["text"] == "Something catch your attention?"
    assert speech.payload["surface_complete"] is True
    assert speech.meta["kind"] == "gap_clarification"


@pytest.mark.asyncio
async def test_neuron_does_not_emit_for_clear_empty_event():
    neuron = GapIdentifierNeuron(
        NeuronConfig(name="gap_identifier_neuron", subscribed_topics=["context/built"], output_topics=[])
    )
    event = Event(topic="context/built", source="ui", payload={"context": {"input": {"text": "", "channel": "textual"}}})
    outputs = list(await neuron.process(event, DummyCtx()))
    assert outputs == []
