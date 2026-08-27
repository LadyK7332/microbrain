import asyncio
from dataclasses import dataclass, field

from microbrain.neurons.language_surface_structure_neuron import LanguageSurfaceStructureNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value


async def _run(neuron, event, ctx):
    result = await neuron.process(event, ctx)
    return list(result)


def _neuron():
    return LanguageSurfaceStructureNeuron(
        NeuronConfig(
            name="language_surface_structure_neuron",
            subscribed_topics=["language/structure_candidate", "cognition/gap_identified"],
            output_topics=["language/surface_structure", "language/surface_plan", "language/surface_candidate"],
        )
    )


def test_neuron_stores_structure_then_builds_surface_candidate():
    ctx = DummyCtx()
    neuron = _neuron()
    structure_events = asyncio.run(
        _run(
            neuron,
            Event(
                topic="language/structure_candidate",
                payload={
                    "structure_id": "lstruct:test",
                    "structure_kind": "unknown_identity_question",
                    "surface_example": "What is that?",
                },
                correlation_id="corr1",
            ),
            ctx,
        )
    )
    assert [e.topic for e in structure_events] == ["language/surface_structure"]
    assert "lstruct:test" in ctx.kv["language:surface_structures"]

    gap_events = asyncio.run(
        _run(
            neuron,
            Event(
                topic="cognition/gap_identified",
                payload={"gap_kind": "object_identity_unknown", "target": "vobj:07"},
                correlation_id="corr2",
            ),
            ctx,
        )
    )
    topics = [e.topic for e in gap_events]
    assert topics == ["language/surface_plan", "language/surface_candidate"]
    assert gap_events[-1].payload["surface"] == "What is vobj:07?"
    assert gap_events[-1].payload["not_canned_response"] is True
    assert ctx.kv["language:last_surface_candidate"]["surface"] == "What is vobj:07?"


def test_neuron_uses_current_focus_when_gap_has_no_target():
    ctx = DummyCtx()
    ctx.kv["vision:focus"] = "vobj:22"
    neuron = _neuron()
    events = asyncio.run(
        _run(
            neuron,
            Event(
                topic="cognition/gap_identified",
                payload={"gap_kind": "object_identity_unknown"},
                correlation_id="corr3",
            ),
            ctx,
        )
    )
    assert events[-1].payload["surface"] == "vobj:22?"
