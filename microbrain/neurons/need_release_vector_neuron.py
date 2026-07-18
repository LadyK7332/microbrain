from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


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


def _clamp01(x: Any) -> float:
    return max(0.0, min(1.0, _safe_float(x, 0.0)))


def _clamp_pct(x: Any) -> float:
    return max(0.0, min(100.0, _safe_float(x, 100.0)))


def _normalize_state(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        state = dict(raw)
    else:
        state = {"mode": raw} if raw not in (None, "") else {}
    state["pct"] = _clamp_pct(state.get("pct", 100.0))
    state["charging"] = _safe_bool(state.get("charging", False), False)
    state["sleep"] = _safe_bool(state.get("sleep", False), False)
    state["mode"] = str(state.get("mode", "active") or "active").strip().lower()
    return state


class NeedReleaseVectorNeuron(BaseNeuron):
    """
    Build a simple interaction-need-release vector for power/hunger.

    This is the first organism loop:
      pressure rises -> choose outlet -> ask -> relief may occur -> learner reinforces path.

    For now the outward valve is textual by default, but the vector keeps track of
    outlet availability and learned preferences so audio/motion can join later.
    """

    async def _channel_availability(self, ctx) -> Dict[str, bool]:
        textual = bool(await ctx.get_kv("outlet:textual_available", True))
        audio_pref = str(await ctx.get_kv("speech:audio_preferred_transport", "none") or "none").strip().lower()
        audio = bool(await ctx.get_kv("outlet:audio_available", audio_pref != "none"))
        motion = bool(await ctx.get_kv("outlet:motion_available", False))
        return {"textual": textual, "audio": audio, "motion": motion}

    async def _compute_pressure(self, ctx, state: Dict[str, Any], now: float) -> Dict[str, Any]:
        pct = _clamp_pct(state.get("pct", 100.0))
        low_mark = _safe_float(await ctx.get_kv("drive:power:low_mark_pct", 70.0), 70.0)
        critical_mark = _safe_float(await ctx.get_kv("drive:power:critical_mark_pct", 25.0), 25.0)
        low_mark = max(critical_mark + 1.0, low_mark)

        base = 0.0
        if pct <= low_mark:
            base = (low_mark - pct) / max(1.0, (low_mark - critical_mark))
        base = _clamp01(base)

        started_ts = _safe_float(await ctx.get_kv("drive:power:pressure_started_ts", 0.0), 0.0)
        if base > 0.0:
            if started_ts <= 0.0:
                started_ts = now
                await ctx.set_kv("drive:power:pressure_started_ts", started_ts)
            persistence_s = max(0.0, now - started_ts)
        else:
            persistence_s = 0.0
            if started_ts > 0.0:
                await ctx.set_kv("drive:power:pressure_started_ts", 0.0)

        persistence_window = _safe_float(await ctx.get_kv("drive:power:persistence_window_s", 300.0), 300.0)
        persistence = _clamp01(persistence_s / max(1.0, persistence_window))
        urgency = _clamp01((base * 0.75) + (persistence * 0.25))

        return {
            "need": "power",
            "pct": pct,
            "base_pressure": round(base, 4),
            "persistence": round(persistence, 4),
            "persistence_s": round(persistence_s, 3),
            "urgency": round(urgency, 4),
            "ts": now,
        }

    async def _choose_vector(self, ctx, pressure: Dict[str, Any], channels: Dict[str, bool]) -> Dict[str, Any]:
        stats = await ctx.get_kv("route:power_relief_stats", {})
        if not isinstance(stats, dict):
            stats = {}

        options: list[Dict[str, Any]] = []
        for outlet in ("textual", "audio", "motion"):
            if not channels.get(outlet, False):
                continue
            entry = stats.get(outlet, {}) if isinstance(stats.get(outlet, {}), dict) else {}
            success_rate = _clamp01(entry.get("success_rate", 0.0))
            avg_relief = max(0.0, _safe_float(entry.get("avg_relief", 0.0), 0.0))
            avg_latency = max(0.0, _safe_float(entry.get("avg_latency_s", 999.0), 999.0))
            latency_score = 0.0 if avg_latency >= 999.0 else _clamp01(1.0 - (avg_latency / 60.0))
            base_bias = {
                "textual": _safe_float(await ctx.get_kv("drive:power:textual_bias", 0.20), 0.20),
                "audio": _safe_float(await ctx.get_kv("drive:power:audio_bias", 0.10), 0.10),
                "motion": _safe_float(await ctx.get_kv("drive:power:motion_bias", 0.05), 0.05),
            }.get(outlet, 0.0)
            score = base_bias + (success_rate * 0.45) + (min(avg_relief, 5.0) / 5.0 * 0.20) + (latency_score * 0.15)
            options.append({
                "outlet": outlet,
                "score": round(score, 4),
                "success_rate": round(success_rate, 4),
                "avg_relief": round(avg_relief, 4),
                "avg_latency_s": round(avg_latency, 4) if avg_latency < 999.0 else None,
            })

        if not options:
            return {
                "need": "power",
                "outlet": None,
                "style": None,
                "message": None,
                "score": 0.0,
                "channels": channels,
                "options": options,
            }

        options.sort(key=lambda item: (item["score"], item["outlet"] == "textual"), reverse=True)
        chosen = dict(options[0])
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        if urgency >= 0.85:
            style = "urgent_direct"
            message = "I need to charge soon."
        elif urgency >= 0.55:
            style = "direct_simple"
            message = "My power is getting low."
        else:
            style = "gentle_notice"
            message = "Power is dipping a bit."

        chosen.update({
            "need": "power",
            "style": style,
            "message": message,
            "channels": channels,
            "options": options,
        })
        return chosen

    def _status_text(self, pressure: Dict[str, Any], state: Dict[str, Any], vector: Dict[str, Any]) -> str:
        pct = _safe_float(pressure.get("pct", state.get("pct", 100.0)), 100.0)
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        band = self._threshold_band(urgency)
        mode = str(state.get("mode", "active") or "active")
        if band == "critical":
            return f"power: critical at {pct:.0f}% | charge soon | mode={mode}"
        if band == "active":
            return f"power: low at {pct:.0f}% | charge soon | mode={mode}"
        return f"power: dipping at {pct:.0f}% | watch charge | mode={mode}"

    async def _emit_status(self, ctx, event: Event, pressure: Dict[str, Any], state: Dict[str, Any], vector: Dict[str, Any], now: float) -> list[Event]:
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        band = self._threshold_band(urgency)
        status_payload = {
            "text": self._status_text(pressure, state, vector),
            "kind": "power_need_status",
            "need": "power",
            "band": band,
            "urgency": round(urgency, 4),
            "pct": round(_safe_float(pressure.get("pct", state.get("pct", 100.0)), 100.0), 2),
            "state": state,
            "pressure": pressure,
            "vector": vector,
            "speech_allowed": False,
            "ts": now,
        }
        await ctx.set_kv("drive:power:last_status_ts", now)
        await ctx.set_kv("drive:power:last_status", status_payload)
        return [
            Event(
                topic="ui/status",
                payload=status_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "power_need_status",
                    "need": "power",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                },
            )
        ]

    async def _speech_allowed(self, ctx, pressure: Dict[str, Any]) -> bool:
        # Internal pressure is body/status first. It may seize the mouth only when
        # explicitly enabled or genuinely critical. User-requested self-report is
        # handled by the normal responder from the stored internal status.
        gate_enabled = _safe_bool(await ctx.get_kv("drive:power:speech_gate_enabled", True), True)
        if not gate_enabled:
            return True
        if _safe_bool(await ctx.get_kv("drive:power:allow_unsolicited_speech", False), False):
            return True
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        critical_threshold = _safe_float(await ctx.get_kv("drive:power:critical_speech_threshold", 0.90), 0.90)
        return urgency >= critical_threshold

    async def _emit_request(self, ctx, event: Event, pressure: Dict[str, Any], vector: Dict[str, Any], now: float) -> list[Event]:
        outlet = vector.get("outlet")
        style = str(vector.get("style", "direct_simple") or "direct_simple")
        message = str(vector.get("message", "I need to charge.") or "I need to charge.")
        thought_text = self._need_thought_text(pressure, style)

        pending = {
            "need": "power",
            "ts": now,
            "pressure": pressure,
            "vector": vector,
            "outlet": outlet,
            "style": style,
            "message": message,
            "thought_text": thought_text,
            "correlation_id": event.correlation_id,
        }
        await ctx.set_kv("drive:power_pending_request", pending)
        await ctx.set_kv("drive:power:last_signal_ts", now)
        await ctx.set_kv("drive:power:last_signal_style", style)

        outputs = [
            Event(
                topic="thought/internal",
                payload={
                    "text": thought_text,
                    "kind": "need_state",
                    "source_need": "power",
                    "urgency": _safe_float(pressure.get("urgency", 0.0), 0.0),
                    "pressure": pressure,
                    "style": style,
                    "threshold_band": self._threshold_band(_safe_float(pressure.get("urgency", 0.0), 0.0)),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "channel": "thought",
                    "kind": "need_state_thought",
                    "need": "power",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                    "cognitive_visible": False,
                },
            ),
            Event(
                topic="drive/power_request",
                payload=pending,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"need": "power", "outlet": outlet, "style": style},
            ),
            Event(
                topic="speech/reason",
                payload={
                    **pending,
                    "channel": "default",
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "need": "power",
                    "outlet": outlet,
                    "style": style,
                    "kind": "need_signal_reason",
                },
            ),
        ]
        return outputs

    def _threshold_band(self, urgency: float) -> str:
        if urgency >= 0.85:
            return "critical"
        if urgency >= 0.55:
            return "active"
        if urgency > 0.0:
            return "rising"
        return "inactive"

    def _need_thought_text(self, pressure: Dict[str, Any], style: str) -> str:
        pct = _safe_float(pressure.get("pct", 100.0), 100.0)
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        if style == "urgent_direct" or urgency >= 0.85:
            return f"Power is critical at {pct:.0f}%. I need to charge."
        if style == "direct_simple" or urgency >= 0.55:
            return f"Power is low at {pct:.0f}%. I need to charge."
        return f"Power is dipping at {pct:.0f}%. I should top up soon."

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.time()
        state = _normalize_state(await ctx.get_kv("power:state", None))
        pressure = await self._compute_pressure(ctx, state, now)
        channels = await self._channel_availability(ctx)
        vector = await self._choose_vector(ctx, pressure, channels)

        snapshot = {
            "need": "power",
            "state": state,
            "pressure": pressure,
            "vector": vector,
            "channels": channels,
            "ts": now,
        }
        await ctx.set_kv("drive:power_vector", snapshot)

        if state.get("charging", False) or state.get("sleep", False):
            return []
        if event.topic == "event/relief/power":
            return []
        if vector.get("outlet") is None:
            return []

        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        threshold = _safe_float(await ctx.get_kv("drive:power:signal_threshold", 0.58), 0.58)
        if urgency < threshold:
            return []

        last_relief_ts = _safe_float(await ctx.get_kv("drive:power:last_relief_ts", 0.0), 0.0)
        quiet_after_relief_s = _safe_float(await ctx.get_kv("drive:power:quiet_after_relief_s", 30.0), 30.0)
        if last_relief_ts > 0.0 and (now - last_relief_ts) < quiet_after_relief_s:
            return []

        if not await self._speech_allowed(ctx, pressure):
            last_status_ts = _safe_float(await ctx.get_kv("drive:power:last_status_ts", 0.0), 0.0)
            status_cooldown_s = _safe_float(await ctx.get_kv("drive:power:status_cooldown_s", 120.0), 120.0)
            if last_status_ts > 0.0 and (now - last_status_ts) < status_cooldown_s:
                return []
            return await self._emit_status(ctx, event, pressure, state, vector, now)

        last_signal_ts = _safe_float(await ctx.get_kv("drive:power:last_signal_ts", 0.0), 0.0)
        cooldown_s = _safe_float(await ctx.get_kv("drive:power:signal_cooldown_s", 90.0), 90.0)
        if last_signal_ts > 0.0 and (now - last_signal_ts) < cooldown_s:
            return []

        return await self._emit_request(ctx, event, pressure, vector, now)


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "control/power", "event/relief/power"],
        output_topics=["thought/internal", "drive/power_request", "speech/reason", "ui/status"],
        priority=8,
        cooldown_sec=0.0,
    )
    yield NeedReleaseVectorNeuron(cfg)
