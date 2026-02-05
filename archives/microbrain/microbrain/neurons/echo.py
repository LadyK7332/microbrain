from __future__ import annotations

from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class EchoNeuron(BaseNeuron):
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )
        
        reply = Event(
            topic="act/speech",
            payload=f"[echo:{self.name}] {event.payload}",
            source=self.name,
            correlation_id=event.correlation_id,
        )
        return [reply]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="echo_neuron",
        subscribed_topics=["percept/text"],
        output_topics=["act/speech"],
        priority=0,
    )
    yield EchoNeuron(cfg)
