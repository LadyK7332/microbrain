from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.evidence.evidence_request_router import TRIGGER_RULES, route_evidence_requests
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
OUTPUT_TOPIC = "memory/evidence_request"


class EvidenceRequestRouterNeuron(BaseNeuron):
    """Turn proof-demand events into bounded evidence-loader requests."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        route = route_evidence_requests(
            event.topic,
            event.payload,
            source=event.source,
            event_meta=event.meta,
            correlation_id=event.correlation_id,
        )
        if not route.get("routed"):
            return []

        outputs: list[Event] = []
        for request in route.get("requests", []) or []:
            outputs.append(
                Event(
                    topic=OUTPUT_TOPIC,
                    payload=request,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "evidence_request",
                        "trigger_topic": event.topic,
                        "route_reason": request.get("route_reason", ""),
                        "mode": request.get("mode", "summary"),
                        "priority": request.get("priority", 0.5),
                        "store_in_memory": False,
                        "cognitive_visible": False,
                        "raw_payload_policy": "reference_only_request",
                    },
                )
            )
        return outputs


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=list(TRIGGER_RULES.keys()),
        output_topics=[OUTPUT_TOPIC],
        priority=1,
    )
    return [EvidenceRequestRouterNeuron(cfg)]
