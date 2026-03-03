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

        # Default OFF: echo is for debug/calibration only.
        # Enable by setting KV: echo:enabled = True
        echo_enabled = bool(await ctx.get_kv("echo:enabled", False))
        if not echo_enabled:
            return []

        # Echo only human-meaningful text (prevents "JSON toxin" / meta leakage)
        if isinstance(event.payload, dict):
            safe_text = event.payload.get("text") or event.payload.get("message") or ""
        else:
            safe_text = event.payload

        if not isinstance(safe_text, str):
            safe_text = str(safe_text)

        safe_text = safe_text.strip()
        if not safe_text:
            return []

        reply = Event(
            topic="act/speech",
            payload=safe_text,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"kind": "echo"},
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
