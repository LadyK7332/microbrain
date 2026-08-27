from __future__ import annotations

import asyncio

from microbrain.neurons.evidence_loaded_binder_neuron import EvidenceLoadedBinderNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


async def _run(neuron, event):
    result = await neuron.process(event, None)
    return list(result or [])


def _neuron() -> EvidenceLoadedBinderNeuron:
    return EvidenceLoadedBinderNeuron(
        NeuronConfig(
            name="evidence_loaded_binder_neuron",
            subscribed_topics=["evidence/loaded"],
            output_topics=["evidence/observation", "hypothesis/evidence_observation"],
        )
    )


def test_binder_emits_generic_and_routed_observation() -> None:
    event = Event(
        topic="evidence/loaded",
        payload={
            "ok": True,
            "artifact_ref": "evidence/touch/day/contact.jsonl",
            "artifact_kind": "jsonl",
            "mode": "walk",
            "item_count": 2,
            "items": [{"summary": "first"}, {"summary": "second"}],
            "trigger_topic": "hypothesis/contradiction",
            "route_reason": "hypothesis_contradiction_needs_ordered_evidence",
        },
        source="evidence_loader_neuron",
        correlation_id="chain-9",
    )
    outputs = asyncio.run(_run(_neuron(), event))
    assert [out.topic for out in outputs] == ["evidence/observation", "hypothesis/evidence_observation"]
    assert outputs[0].correlation_id == "chain-9"
    assert outputs[0].payload["raw_policy"] == "observation_only_no_raw_memory_ingest"
    assert outputs[1].meta["cognitive_visible"] is True
    assert outputs[1].meta["store_in_memory"] is False


def test_binder_only_emits_generic_when_no_route_matches() -> None:
    event = Event(topic="evidence/loaded", payload={"ok": True, "item_count": 1, "items": []})
    outputs = asyncio.run(_run(_neuron(), event))
    assert [out.topic for out in outputs] == ["evidence/observation"]
