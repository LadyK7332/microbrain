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


def _clamp_pct(x: Any) -> float:
    return max(0.0, min(100.0, _safe_float(x, 100.0)))


def _normalize_mode(x: Any) -> str:
    mode = str(x or "active").strip().lower()
    if mode in ("charging", "charge"):
        return "charge"
    if mode in ("sleep", "sleeping"):
        return "sleep"
    if mode == "idle":
        return "idle"
    return "active"


def _state_dict(raw: Any, now: float) -> Dict[str, Any]:
    if isinstance(raw, dict):
        state = dict(raw)
    else:
        state = {}
        if raw not in (None, ""):
            state["mode"] = raw

    state["pct"] = _clamp_pct(state.get("pct", 100.0))
    state["charging"] = _safe_bool(state.get("charging", False), False)
    state["sleep"] = _safe_bool(state.get("sleep", False), False)
    state["mode"] = _normalize_mode(state.get("mode", state.get("state", "active")))
    state["last_ts"] = _safe_float(state.get("last_ts"), now)
    return state


def _is_outward(payload: Dict[str, Any]) -> bool:
    channel = str(payload.get("channel", "default") or "default").strip().lower()
    raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
    autonomous = bool(raw_meta.get("autonomous", False))
    return channel not in ("thought", "internal") and not autonomous


class BatteryClockNeuron(BaseNeuron):
    """
    Simulated power state + sleep gate.

    KV keys written:
      - power:state {pct, charging, sleep, mode, last_ts}
      - power:battery_pct (float 0..100)
      - power:mode (active|idle|charge|sleep)
      - entropy:allowed (bool)  -> True only if charging AND sleep

    Controls accepted via events:
      topic: control/power
        payload examples:
          {"set_pct": 75}
          {"add_pct": 5}
          {"charging": true}
          {"sleep": true}
          {"mode": "idle"}

    Passive power model:
      - active drain over time
      - lower drain in idle
      - lowest drain in sleep
      - charging adds over time

    Action costs:
      - reason/request
      - act/speech
      - task/start
      - act/motor (future-friendly)
    """

    async def _write_state(self, ctx, state: Dict[str, Any]) -> None:
        state["pct"] = _clamp_pct(state.get("pct", 100.0))
        state["charging"] = _safe_bool(state.get("charging", False), False)
        state["sleep"] = _safe_bool(state.get("sleep", False), False)
        state["mode"] = _normalize_mode(state.get("mode", "active"))

        await ctx.set_kv("power:state", state)
        await ctx.set_kv("power:battery_pct", float(state.get("pct", 100.0)))
        await ctx.set_kv("power:charging", bool(state.get("charging", False)))
        await ctx.set_kv("power:sleep", bool(state.get("sleep", False)))
        await ctx.set_kv("power:mode", str(state.get("mode", "active")))

        sleeping = bool(state.get("sleep", False))
        charging = bool(state.get("charging", False))
        await ctx.set_kv("entropy:allowed", bool(charging and sleeping))

        if sleeping:
            await ctx.set_kv("attention:allow_babble", False)

    async def _apply_action_cost(self, ctx, event: Event, now: float) -> None:
        raw_state = await ctx.get_kv("power:state", None)
        state = _state_dict(raw_state, now)
        payload = event.payload if isinstance(event.payload, dict) else {}

        cost = 0.0
        if event.topic == "reason/request":
            if _is_outward(payload):
                cost = _safe_float(await ctx.get_kv("power:reason_request_cost", 0.10), 0.10)
            else:
                cost = _safe_float(await ctx.get_kv("power:reason_internal_cost", 0.03), 0.03)
        elif event.topic == "act/speech":
            if _is_outward(payload):
                cost = _safe_float(await ctx.get_kv("power:act_speech_cost", 0.12), 0.12)
            else:
                cost = _safe_float(await ctx.get_kv("power:act_speech_thought_cost", 0.02), 0.02)
        elif event.topic == "task/start":
            cost = _safe_float(await ctx.get_kv("power:task_start_cost", 0.15), 0.15)
        elif event.topic == "act/motor":
            cost = _safe_float(await ctx.get_kv("power:act_motor_cost", 0.50), 0.50)

        if cost <= 0.0:
            return

        state["pct"] = _clamp_pct(float(state.get("pct", 100.0)) - cost)
        await self._write_state(ctx, state)
        await ctx.set_kv("power:last_action_cost", {
            "topic": event.topic,
            "cost": round(cost, 4),
            "ts": now,
        })

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.time()

        if event.topic == "control/power":
            payload = event.payload if isinstance(event.payload, dict) else {}
            state = _state_dict(await ctx.get_kv("power:state", None), now)

            if "set_pct" in payload:
                state["pct"] = _clamp_pct(payload.get("set_pct"))
            if "add_pct" in payload:
                state["pct"] = _clamp_pct(float(state.get("pct", 100.0)) + _safe_float(payload.get("add_pct"), 0.0))
            if "charging" in payload:
                state["charging"] = _safe_bool(payload.get("charging"), False)
                await ctx.set_kv("power:charging_last_set_ts", now)
            if "sleep" in payload:
                state["sleep"] = _safe_bool(payload.get("sleep"), False)
                await ctx.set_kv("power:sleep_last_set_ts", now)
            if "mode" in payload:
                state["mode"] = _normalize_mode(payload.get("mode"))

            state["last_ts"] = now
            await self._write_state(ctx, state)
            return []

        if event.topic in ("reason/request", "act/speech", "task/start", "act/motor"):
            await self._apply_action_cost(ctx, event, now)
            return []

        if event.topic != "clock/tick":
            return []

        state = _state_dict(await ctx.get_kv("power:state", None), now)
        last_ts = _safe_float(state.get("last_ts"), now)
        dt = max(0.0, now - last_ts)

        pct = _clamp_pct(state.get("pct", 100.0))
        charging = _safe_bool(state.get("charging", False), False)
        sleep = _safe_bool(state.get("sleep", False), False)
        mode = _normalize_mode(state.get("mode", await ctx.get_kv("power:mode", "active")))
        busy = bool(await ctx.get_kv("power:busy", False))

        active_drain = _safe_float(await ctx.get_kv("power:active_drain_per_sec", 100.0 / SECONDS_PER_DAY), 100.0 / SECONDS_PER_DAY)
        idle_drain = _safe_float(await ctx.get_kv("power:idle_drain_per_sec", 40.0 / SECONDS_PER_DAY), 40.0 / SECONDS_PER_DAY)
        sleep_drain = _safe_float(await ctx.get_kv("power:sleep_drain_per_sec", 10.0 / SECONDS_PER_DAY), 10.0 / SECONDS_PER_DAY)
        busy_extra = _safe_float(await ctx.get_kv("power:busy_drain_per_sec", 15.0 / SECONDS_PER_DAY), 15.0 / SECONDS_PER_DAY)
        charge_per_sec = _safe_float(await ctx.get_kv("power:charge_per_sec", 100.0 / 3600.0), 100.0 / 3600.0)

        if charging:
            pct = min(100.0, pct + charge_per_sec * dt)
        else:
            if sleep:
                drain_per_sec = sleep_drain
            elif mode == "idle":
                drain_per_sec = idle_drain
            else:
                drain_per_sec = active_drain

            if busy and not sleep:
                drain_per_sec += busy_extra

            pct = max(0.0, pct - (drain_per_sec * dt))

        state["pct"] = pct
        state["charging"] = charging
        state["sleep"] = sleep
        state["mode"] = mode
        state["last_ts"] = now

        await self._write_state(ctx, state)
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "control/power", "reason/request", "act/speech", "task/start", "act/motor"],
        output_topics=[],
        priority=3,
        cooldown_sec=0.0,
    )
    yield BatteryClockNeuron(cfg)
