from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.evidence.evidence_loaded_binder import build_evidence_observation, route_topic_for_observation
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
INPUT_TOPIC = "evidence/loaded"
GENERIC_OUTPUT_TOPIC = "evidence/observation"
ROUTED_OUTPUT_TOPICS = [
    GENERIC_OUTPUT_TOPIC,
    "hypothesis/evidence_observation",
    "review/evidence_observation",
    "trainer/evidence_observation",
    "scene/evidence_observation",
    "recognition/evidence_observation",
    "safety/evidence_observation",
    "object/evidence_observation",
    "memory/evidence_observation",
    "thought/evidence_sample",
]


class EvidenceLoadedBinderNeuron(BaseNeuron):
    """Turn loaded proof samples into compact observations for deliberation."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        observation = build_evidence_observation(event.payload, event_meta=event.meta)
        route_topic = route_topic_for_observation(observation)

        meta = {
            "kind": "evidence_observation",
            "trigger_topic": observation.get("trigger_topic", ""),
            "route_reason": observation.get("route_reason", ""),
            "mode": observation.get("mode", "summary"),
            "ok": bool(observation.get("ok", False)),
            "store_in_memory": False,
            "cognitive_visible": True,
            "raw_payload_policy": "bounded_observation_only",
        }

        outputs: list[Event] = [
            Event(
                topic=GENERIC_OUTPUT_TOPIC,
                payload=observation,
                source=self.name,
                correlation_id=event.correlation_id,
                meta=dict(meta),
            )
        ]
        if route_topic and route_topic != GENERIC_OUTPUT_TOPIC:
            routed_meta = dict(meta)
            routed_meta["routed_from"] = GENERIC_OUTPUT_TOPIC
            outputs.append(
                Event(
                    topic=route_topic,
                    payload=observation,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta=routed_meta,
                )
            )
        return outputs


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[INPUT_TOPIC],
        output_topics=ROUTED_OUTPUT_TOPICS,
        priority=1,
    )
    return [EvidenceLoadedBinderNeuron(cfg)]
