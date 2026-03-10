from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class HazardGateNeuron(BaseNeuron):
    """Arms ER when hazard/report level crosses threshold."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "hazard/report":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        level = int(payload.get("level", 0) or 0)
        tag = str(payload.get("tag", "") or "")
        reason = str(payload.get("reason", "") or "")
        src = str(payload.get("source", event.source) or event.source)

        await ctx.set_kv("hazard:last_level", level)
        await ctx.set_kv("hazard:last_ts", time.time())
        await ctx.set_kv("hazard:last_tag", tag)
        await ctx.set_kv("hazard:last_reason", reason)

        threshold = int(await ctx.get_kv("er:hazard_threshold", 3) or 3)
        enabled = bool(await ctx.get_kv("er:enabled", True))
        if enabled and level >= threshold:
            await ctx.set_kv("er:armed", True)
            await ctx.set_kv("er:last_trigger_ts", time.time())
            await ctx.set_kv("er:last_reason", reason or tag or "hazard")
            await ctx.set_kv("er:last_level", level)
            await ctx.set_kv("er:last_source", src)

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["hazard/report"],
        output_topics=[],
        priority=40,
        cooldown_sec=0.0,
    )
    yield HazardGateNeuron(cfg)
