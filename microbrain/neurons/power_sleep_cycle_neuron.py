from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class PowerSleepCycleNeuron(BaseNeuron):
    """
    Emits power/sleep_cycle when power:sleep is enabled AND the unit is idle.

    Idle is measured by:
      - power:last_external_ts (set by router_text on percept/text)
    """
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "clock/tick":
            return []

        sleep_on = bool(await ctx.get_kv("power:sleep", False))
        if not sleep_on:
            return []

        idle_s = float(await ctx.get_kv("power:sleep_idle_s", 20.0) or 20.0)
        period_s = float(await ctx.get_kv("power:sleep_period_s", 30.0) or 30.0)
        kick = bool(await ctx.get_kv("power:sleep_kick", False))

        now = time.time()
        last_ext = float(await ctx.get_kv("power:last_external_ts", 0.0) or 0.0)
        if last_ext <= 0.0:
            interaction = await ctx.get_kv("interaction:last_input", {}) or {}
            if isinstance(interaction, dict):
                last_ext = float(interaction.get("ts", 0.0) or 0.0)
                if last_ext > 0.0:
                    await ctx.set_kv("power:last_external_ts", last_ext)
        if last_ext and (now - last_ext) < idle_s and not kick:
            return []

        last_cycle = float(await ctx.get_kv("power:sleep_last_cycle_ts", 0.0) or 0.0)
        if not kick and last_cycle and (now - last_cycle) < period_s:
            return []

        await ctx.set_kv("power:sleep_last_cycle_ts", now)
        if kick:
            await ctx.set_kv("power:sleep_kick", False)

        return [
            Event(
                topic="power/sleep_cycle",
                payload={"ts": now, "idle_s": idle_s, "period_s": period_s, "kick": kick},
                source=NEURON_NAME,
                correlation_id=event.correlation_id,
                meta={"kind": "sleep_cycle"},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick"],
        output_topics=["power/sleep_cycle"],
        priority=60,
        cooldown_sec=0.0,
    )
    yield PowerSleepCycleNeuron(cfg)
