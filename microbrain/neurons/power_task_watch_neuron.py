from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class PowerTaskWatchNeuron(BaseNeuron):
    """
    Maintains a simple busy counter to prevent idle mode while tasks are running.

    Increments busy_count on:
      - reason/request
      - task/start  (future subsystem tasks)
    Decrements busy_count on:
      - reason/response
      - task/end

    Sets:
      - power:busy_count (int >= 0)
      - power:busy (bool)
    Also interrupts idle by forcing power:state to 'active' when a task starts.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in ("reason/request", "reason/response", "task/start", "task/end"):
            return []

        busy_count = int(await ctx.get_kv("power:busy_count", 0) or 0)

        if event.topic in ("reason/request", "task/start"):
            busy_count += 1
            await ctx.set_kv("power:busy_count", busy_count)
            await ctx.set_kv("power:busy", True)
            await ctx.set_kv("power:last_task_start_ts", time.time())

            # Interrupt idle immediately
            cur_state = str(await ctx.get_kv("power:state", "active") or "active")
            if cur_state == "idle":
                await ctx.set_kv("power:state", "active")
                await ctx.set_kv("power:state_last_change_ts", time.time())

            return []

        # end events
        busy_count = max(0, busy_count - 1)
        await ctx.set_kv("power:busy_count", busy_count)
        await ctx.set_kv("power:busy", bool(busy_count > 0))
        await ctx.set_kv("power:last_task_end_ts", time.time())
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["reason/request", "reason/response", "task/start", "task/end"],
        output_topics=[],
        priority=10,
        cooldown_sec=0.0,
    )
    yield PowerTaskWatchNeuron(cfg)
