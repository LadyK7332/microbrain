import asyncio

from microbrain.neurons.language_surface_realizer_neuron import (
    LanguageSurfaceRealizerNeuron,
    validate_surface_candidate,
)
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    def __init__(self):
        self.kv = {}

    async def set_kv(self, key, value):
        self.kv[key] = value


def _neuron():
    return LanguageSurfaceRealizerNeuron(
        NeuronConfig(name="language_surface_realizer_neuron", subscribed_topics=["language/surface_candidate"])
    )


def test_realizer_emits_complete_candidate_with_trace_ids():
    event = Event(
        topic="language/surface_candidate",
        correlation_id="turn",
        payload={
            "schema": "language.surface_candidate.v2",
            "surface": "What is that?",
            "surface_complete": True,
            "obligation_id": "obligation:1",
            "plan_id": "plan:1",
            "selected_structure_id": "lstruct:1",
            "required_act": "clarify",
        },
    )
    outputs = list(asyncio.run(_neuron().process(event, DummyCtx())))
    assert len(outputs) == 1
    assert outputs[0].topic == "act/speech"
    assert outputs[0].payload["obligation_id"] == "obligation:1"
    assert outputs[0].meta["language_surface_v2"] is True


def test_realizer_rejects_internal_handle_and_unresolved_slot():
    base = {"schema": "language.surface_candidate.v2", "surface_complete": True, "required_act": "clarify"}
    assert validate_surface_candidate({**base, "surface": "What is vobj:07?"})[0] is False
    assert validate_surface_candidate({**base, "surface": "What is {target}?"})[0] is False


def test_realizer_marks_floor_speech_non_learning():
    event = Event(
        topic="language/surface_candidate",
        correlation_id="turn",
        payload={
            "schema": "language.surface_candidate.v2",
            "surface": "Reply not ready.",
            "surface_complete": True,
            "surface_floor": True,
            "surface_floor_policy": "minimal_non_learning_discharge",
            "obligation_id": "obligation:floor",
            "plan_id": "plan:floor",
            "required_act": "answer",
        },
    )
    outputs = list(asyncio.run(_neuron().process(event, DummyCtx())))
    assert len(outputs) == 1
    assert outputs[0].topic == "act/speech"
    assert outputs[0].meta["kind"] == "language_surface_floor_realization"
    assert outputs[0].meta["store_in_memory"] is False
    assert outputs[0].payload["store_in_memory"] is False
