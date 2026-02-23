from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
SECONDS_PER_DAY = 86400.0


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "on")
    return default


class BatteryClockNeuron(BaseNeuron):
    """
    Simulated power state + sleep gate.

    KV keys written:
      - power:state {pct, charging, sleep, last_ts}
      - power:battery_pct (float 0..100)
      - entropy:allowed (bool)  -> True only if charging AND sleep

    Controls accepted via events (optional for now):
      topic: control/power
        payload examples:
          {"set_pct": 75}
          {"charging": true}
          {"sleep": true}
          {"sleep": false}
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic == "control/power":
            payload = event.payload if isinstance(event.payload, dict) else {}
            state = await ctx.get_kv("power:state", {}) or {}
            if not isinstance(state, dict):
                state = {}

            if "set_pct" in payload:
                state["pct"] = max(0.0, min(100.0, _safe_float(payload.get("set_pct"), 100.0)))
            if "charging" in payload:
                state["charging"] = _safe_bool(payload.get("charging"), False)
            if "sleep" in payload:
                state["sleep"] = _safe_bool(payload.get("sleep"), False)

            state["last_ts"] = time.time()

            await ctx.set_kv("power:state", state)
            await ctx.set_kv("power:battery_pct", float(state.get("pct", 100.0)))
            sleeping = bool(state.get("sleep", False))
            await ctx.set_kv("entropy:allowed", bool(state.get("charging", False) and sleeping))

            if sleeping:
                await ctx.set_kv("attention:allow_babble", False)

            return []
        
        if event.topic != "clock/tick":
            return []

        now = time.time()

        state = await ctx.get_kv("power:state", None)
        if not isinstance(state, dict):
            state = {"pct": 100.0, "charging": False, "sleep": False, "last_ts": now}

        last_ts = _safe_float(state.get("last_ts"), now)
        dt = max(0.0, now - last_ts)

        pct = max(0.0, min(100.0, _safe_float(state.get("pct"), 100.0)))
        charging = _safe_bool(state.get("charging"), False)
        sleep = _safe_bool(state.get("sleep"), False)

        # Tunable rates (KV-overridable)
        drain_per_sec = _safe_float(await ctx.get_kv("power:drain_per_sec", 100.0 / SECONDS_PER_DAY), 100.0 / SECONDS_PER_DAY)
        charge_per_sec = _safe_float(await ctx.get_kv("power:charge_per_sec", 100.0 / 3600.0), 100.0 / 3600.0)  # 1h full charge default

        if charging:
            pct = min(100.0, pct + charge_per_sec * dt)
        else:
            pct = max(0.0, pct - drain_per_sec * dt)

        state["pct"] = pct
        state["charging"] = charging
        state["sleep"] = sleep
        state["last_ts"] = now

        await ctx.set_kv("power:state", state)
        await ctx.set_kv("power:battery_pct", pct)

        # This is the gate you asked for: entropy runs only in sleep+charging.
        await ctx.set_kv("entropy:allowed", bool(charging and sleep))

        # Sleep should be quiet: clamp babble while sleeping.
        if sleep:
            await ctx.set_kv("attention:allow_babble", False)

        return []

def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "control/power"],
        output_topics=[],
        priority=3,
        cooldown_sec=0.25,
    )
    yield BatteryClockNeuron(cfg)