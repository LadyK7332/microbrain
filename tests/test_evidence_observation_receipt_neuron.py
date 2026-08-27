import json
from pathlib import Path

import pytest

from microbrain.neurons.evidence_observation_receipt_neuron import EvidenceObservationReceiptNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self, memdir: Path):
        self.memdir = memdir

    async def get_kv(self, key, default=None):
        if key == "memory:base_dir":
            return str(self.memdir)
        return default


@pytest.mark.asyncio
async def test_neuron_stages_short_receipt_without_sample_payload(tmp_path):
    neuron = EvidenceObservationReceiptNeuron(
        NeuronConfig(
            name="evidence_observation_receipt_neuron",
            subscribed_topics=["evidence/observation"],
            output_topics=["memory/evidence_receipt"],
        )
    )
    event = Event(
        topic="evidence/observation",
        payload={
            "schema": "evidence.observation.v1",
            "observation_id": "evidence_obs:test",
            "ok": True,
            "artifact_ref": "evidence/touch/thing.jsonl",
            "artifact_kind": "touch",
            "mode": "directed",
            "query": "soft pressure",
            "item_count": 2,
            "items_sample": [{"pressure_series": list(range(64))}],
            "summary": "loaded 2 touch evidence items",
            "trigger_topic": "safety/uncertain_action",
            "route_reason": "safety_uncertainty",
            "priority": 0.91,
            "confidence_hint": 0.7,
        },
        source="evidence_loaded_binder_neuron",
        correlation_id="corr-1",
    )

    outputs = list(await neuron.process(event, FakeCtx(tmp_path / "memory")))
    assert len(outputs) == 1
    out = outputs[0]
    assert out.topic == "memory/evidence_receipt"
    assert out.payload["staged"] is True
    assert out.payload["tier"] == "short"
    assert "items_sample" not in out.payload

    pending = list((tmp_path / "memory" / "mem_cell" / "_pending" / "short").glob("*.jsonl"))
    assert len(pending) == 1
    rows = [json.loads(line) for line in pending[0].read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema"] == "mem_cell.pending_upsert.v1"
    staged_row = rows[0]["row"]
    assert staged_row["schema"] == "mem_cell.evidence_observation_receipt.v1"
    assert staged_row["artifact_ref"] == "evidence/touch/thing.jsonl"
    assert "items_sample" not in json.dumps(staged_row)


@pytest.mark.asyncio
async def test_neuron_reports_unstaged_low_priority_scatter(tmp_path):
    neuron = EvidenceObservationReceiptNeuron(
        NeuronConfig(
            name="evidence_observation_receipt_neuron",
            subscribed_topics=["evidence/observation"],
            output_topics=["memory/evidence_receipt"],
        )
    )
    event = Event(
        topic="evidence/observation",
        payload={
            "schema": "evidence.observation.v1",
            "ok": True,
            "artifact_ref": "evidence/touch/thing.jsonl",
            "mode": "scatter",
            "item_count": 1,
            "items_sample": [{"note": "idle sample"}],
            "trigger_topic": "thought/probe",
            "route_reason": "",
            "priority": 0.2,
            "summary": "sampled one idle evidence item",
        },
        correlation_id="corr-2",
    )
    outputs = list(await neuron.process(event, FakeCtx(tmp_path / "memory")))
    assert outputs[0].payload["staged"] is False
    assert not (tmp_path / "memory" / "mem_cell" / "_pending").exists()
