from __future__ import annotations

"""
Evidence Convergence Organ neuron.

This neuron merges comparable evidence packets into workspace candidates and
accepted working beliefs.  Contradictory feedback against a recent high-confidence
working belief becomes an anomaly event, not silent relabeling.
"""

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.evidence_convergence import (
    DEFAULT_ACCEPTED_BELIEF_THRESHOLD,
    DEFAULT_ANOMALY_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_CONVERGENCE_WINDOW_S,
    WorkingBelief,
    contradiction_anomaly,
    converge_evidence_packets,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sensory_fossils import EvidencePacket, clamp01

NEURON_NAME = Path(__file__).stem
PACKET_TOPIC = "evidence/packet"
FEEDBACK_TOPIC = "workspace/belief_feedback"
PENDING_KEY = "evidence:pending_packets:v1"
LAST_CANDIDATES_KEY = "evidence:convergence:last_candidates"
BELIEF_INDEX_KEY = "workspace:working_beliefs:v1"
LAST_ANOMALY_KEY = "workspace:anomaly:last"


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class EvidenceConvergenceNeuron(BaseNeuron):
    async def _read_pending(self, ctx) -> list[dict[str, Any]]:
        data = await ctx.get_kv(PENDING_KEY, [])
        if isinstance(data, list):
            return [dict(x) for x in data if isinstance(x, Mapping)]
        return []

    async def _write_pending(self, ctx, rows: list[dict[str, Any]]) -> None:
        await ctx.set_kv(PENDING_KEY, rows)

    async def _read_beliefs(self, ctx) -> dict[str, dict[str, Any]]:
        data = await ctx.get_kv(BELIEF_INDEX_KEY, {})
        if isinstance(data, Mapping):
            return {str(k): dict(v) for k, v in data.items() if isinstance(v, Mapping)}
        return {}

    async def _write_beliefs(self, ctx, rows: dict[str, dict[str, Any]]) -> None:
        await ctx.set_kv(BELIEF_INDEX_KEY, rows)

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now_ts = time.time()

        if event.topic == PACKET_TOPIC:
            payload = _payload_dict(event.payload)
            packet = EvidencePacket.from_dict(payload)
            window_s = max(0.1, _safe_float(await ctx.get_kv("evidence:convergence_window_s", DEFAULT_CONVERGENCE_WINDOW_S), DEFAULT_CONVERGENCE_WINDOW_S))
            pending = await self._read_pending(ctx)
            pending.append(packet.to_dict())
            pending = [row for row in pending if (now_ts - _safe_float(row.get("timestamp"), 0.0)) <= max(0.1, window_s)]
            await self._write_pending(ctx, pending)

            candidate_threshold = clamp01(
                await ctx.get_kv("evidence:candidate_threshold", DEFAULT_CANDIDATE_THRESHOLD),
                DEFAULT_CANDIDATE_THRESHOLD,
            )
            accepted_threshold = clamp01(
                await ctx.get_kv("evidence:accepted_belief_threshold", DEFAULT_ACCEPTED_BELIEF_THRESHOLD),
                DEFAULT_ACCEPTED_BELIEF_THRESHOLD,
            )
            candidates = converge_evidence_packets(
                pending,
                now_ts=now_ts,
                window_s=window_s,
                candidate_threshold=candidate_threshold,
                accepted_threshold=accepted_threshold,
            )
            await ctx.set_kv(LAST_CANDIDATES_KEY, [c.to_dict() for c in candidates])

            outputs: list[Event] = []
            beliefs = await self._read_beliefs(ctx)
            for candidate in candidates:
                outputs.append(
                    Event(
                        topic="workspace/candidate",
                        payload=candidate.to_dict(),
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={
                            "event_class": "workspace",
                            "store_in_memory": False,
                            "semantic_input": True,
                            "meaning_boundary": "candidate_is_revisable",
                        },
                    )
                )
                if candidate.accepted_working_belief and candidate.target_refs:
                    belief = WorkingBelief.from_candidate(candidate)
                    beliefs[belief.subject_ref] = belief.to_dict()
                    outputs.append(
                        Event(
                            topic="workspace/working_belief",
                            payload=belief.to_dict(),
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={
                                "event_class": "workspace",
                                "store_in_memory": False,
                                "semantic_input": True,
                                "meaning_boundary": "working_belief_not_truth",
                            },
                        )
                    )
            if outputs:
                await self._write_beliefs(ctx, beliefs)
            return outputs

        if event.topic == FEEDBACK_TOPIC:
            payload = _payload_dict(event.payload)
            subject_ref = str(payload.get("subject_ref") or payload.get("source_ref") or "")
            beliefs = await self._read_beliefs(ctx)
            belief_payload = beliefs.get(subject_ref) if subject_ref else None
            if not belief_payload:
                return []
            threshold = clamp01(await ctx.get_kv("evidence:anomaly_threshold", DEFAULT_ANOMALY_THRESHOLD), DEFAULT_ANOMALY_THRESHOLD)
            anomaly = contradiction_anomaly(belief_payload, payload, now_ts=now_ts, threshold=threshold)
            if anomaly is None:
                return []
            await ctx.set_kv(LAST_ANOMALY_KEY, anomaly.to_dict())
            return [
                Event(
                    topic="workspace/anomaly",
                    payload=anomaly.to_dict(),
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "event_class": "workspace",
                        "store_in_memory": False,
                        "semantic_input": True,
                        "meaning_boundary": "contradiction_demands_investigation",
                    },
                )
            ]

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="evidence_convergence",
        subscribed_topics=[PACKET_TOPIC, FEEDBACK_TOPIC],
        output_topics=["workspace/candidate", "workspace/working_belief", "workspace/anomaly"],
        cooldown_sec=0.0,
    )
    yield EvidenceConvergenceNeuron(cfg)
