from __future__ import annotations

import asyncio
from pathlib import Path

from microbrain.evidence.artifact_store import EvidenceArtifactStore
from microbrain.neurons.evidence_loader_neuron import EvidenceLoaderNeuron, attach_request_context_to_loaded
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    def __init__(self, memdir: Path):
        self.memdir = memdir

    async def get_kv(self, key, default=None):
        if key == "memory:base_dir":
            return str(self.memdir)
        return default


async def _run(neuron, event, ctx):
    result = await neuron.process(event, ctx)
    return list(result or [])


def _neuron() -> EvidenceLoaderNeuron:
    return EvidenceLoaderNeuron(
        NeuronConfig(
            name="evidence_loader_neuron",
            subscribed_topics=["memory/evidence_request"],
            output_topics=["evidence/loaded"],
        )
    )


def test_attach_request_context_to_loaded() -> None:
    loaded = {"ok": True, "artifact_ref": "evidence/x.jsonl"}
    request = {
        "request_id": "evidence_req:123",
        "trigger_topic": "review/repair_candidate",
        "trigger_source": "review_neuron",
        "route_reason": "review_repair_candidate_needs_supporting_evidence",
        "priority": 0.72,
    }
    out = attach_request_context_to_loaded(loaded, request, request_topic="memory/evidence_request", requested_by="router")
    assert out["request_id"] == "evidence_req:123"
    assert out["trigger_topic"] == "review/repair_candidate"
    assert out["request_topic"] == "memory/evidence_request"
    assert out["requested_by"] == "router"
    assert out["load_context_schema"] == "evidence.loaded_context.v1"


def test_loader_neuron_preserves_request_context(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    card = store.write_jsonl_artifact(
        modality="touch",
        records=[{"summary": "soft fuzzy contact"}],
        timestamp=50.0,
        summary="touch row",
    )
    event = Event(
        topic="memory/evidence_request",
        payload={
            "artifact_ref": card["artifact_ref"],
            "mode": "directed",
            "query": "fuzzy",
            "request_id": "evidence_req:ctx",
            "trigger_topic": "hypothesis/contradiction",
            "trigger_source": "hypothesis_engine",
            "route_reason": "hypothesis_contradiction_needs_ordered_evidence",
            "priority": 0.92,
        },
        source="evidence_request_router_neuron",
        correlation_id="chain-1",
    )
    outputs = asyncio.run(_run(_neuron(), event, DummyCtx(tmp_path)))
    assert len(outputs) == 1
    out = outputs[0]
    assert out.topic == "evidence/loaded"
    assert out.payload["ok"] is True
    assert out.payload["request_id"] == "evidence_req:ctx"
    assert out.payload["trigger_topic"] == "hypothesis/contradiction"
    assert out.payload["route_reason"] == "hypothesis_contradiction_needs_ordered_evidence"
    assert out.meta["store_in_memory"] is False
