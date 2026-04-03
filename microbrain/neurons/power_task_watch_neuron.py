from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _coerce_power_state(raw):
    if isinstance(raw, dict):
        state = dict(raw)
    else:
        state = {"mode": str(raw or "active")}
    state["mode"] = str(state.get("mode", "active") or "active").lower()
    if "pct" not in state:
        state["pct"] = 100.0
    state["charging"] = bool(state.get("charging", False))
    state["sleep"] = bool(state.get("sleep", False))
    return state


class PowerTaskWatchNeuron(BaseNeuron):
    """
    Maintains a simple busy counter to prevent idle mode while outward tasks are running.

    Increments busy_count on:
      - reason/request for outward channels
      - task/start  (future subsystem tasks)
    Decrements busy_count on:
      - reason/response
      - act/speech for outward channels
      - task/end

    Sets:
      - power:busy_count (int >= 0)
      - power:busy (bool)
    Also interrupts idle by forcing power:state to 'active' when an outward task starts.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in ("reason/request", "reason/response", "act/speech", "task/start", "task/end"):
            return []

        busy_count = int(await ctx.get_kv("power:busy_count", 0) or 0)
        payload = event.payload if isinstance(event.payload, dict) else {}
        channel = str(payload.get("channel", "default") or "default").strip().lower()
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        autonomous = bool(raw_meta.get("autonomous", False))
        outward = channel not in ("thought", "internal") and not autonomous

        if event.topic == "reason/request":
            if not outward:
                return []
            busy_count += 1
            await ctx.set_kv("power:busy_count", busy_count)
            await ctx.set_kv("power:busy", True)
            await ctx.set_kv("power:last_task_start_ts", time.time())

            # Interrupt idle immediately
            state = _coerce_power_state(await ctx.get_kv("power:state", None))
            cur_state = str(state.get("mode", "active") or "active").lower()
            if cur_state == "idle":
                state["mode"] = "active"
                await ctx.set_kv("power:state", state)
                await ctx.set_kv("power:mode", "active")
                await ctx.set_kv("power:state_last_change_ts", time.time())

            return []

        if event.topic == "task/start":
            busy_count += 1
            await ctx.set_kv("power:busy_count", busy_count)
            await ctx.set_kv("power:busy", True)
            await ctx.set_kv("power:last_task_start_ts", time.time())

            state = _coerce_power_state(await ctx.get_kv("power:state", None))
            cur_state = str(state.get("mode", "active") or "active").lower()
            if cur_state == "idle":
                state["mode"] = "active"
                await ctx.set_kv("power:state", state)
                await ctx.set_kv("power:mode", "active")
                await ctx.set_kv("power:state_last_change_ts", time.time())
            return []

        if event.topic == "act/speech" and not outward:
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
        subscribed_topics=["reason/request", "reason/response", "act/speech", "task/start", "task/end"],
        output_topics=[],
        priority=10,
        cooldown_sec=0.0,
    )
    yield PowerTaskWatchNeuron(cfg)
