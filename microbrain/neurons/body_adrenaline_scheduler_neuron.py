from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import (
    PRIMARY_HEARTBEAT_TOPIC,
    service_tick_meta,
    service_tick_payload,
    service_topic,
)

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Number of canonical 20-TPS body heartbeats between service opportunities.
# These are engineering defaults, not biological claims.  The heartbeat never
# changes frequency; arousal reallocates cadence among selected organs.
CADENCE_PROFILES: dict[str, dict[str, int]] = {
    "normal": {
        "cognition": 1,          # full 20 Hz cognitive coordination frame
        "affect": 10,            # 2 Hz body/drive housekeeping
        "curiosity": 10,         # 2 Hz autonomous/background curiosity
        "vision": 4,             # 5 Hz capture/report opportunity
        "gaze": 2,               # 10 Hz gaze coordination
        "touch": 2,
        "proprioception": 2,
        "motor_watch": 2,
        "hazard": 4,
        "capability": 4,
        "outcome": 10,
        "ipc": 4,
        "evidence": 4,
        "power": 20,             # 1 Hz power/sleep physiology
        "memory": 20,            # 1 Hz memory housekeeping trigger
        "maintenance": 20,
    },
    "alert": {
        "cognition": 1,
        "affect": 4,
        "curiosity": 20,         # background exploration backs off
        "vision": 2,
        "gaze": 1,
        "touch": 1,
        "proprioception": 1,
        "motor_watch": 1,
        "hazard": 2,
        "capability": 2,
        "outcome": 4,
        "ipc": 4,
        "evidence": 2,
        "power": 10,
        "memory": 40,            # defer noncritical digestion
        "maintenance": 40,
    },
    "emergency": {
        "cognition": 1,
        "affect": 2,
        "curiosity": 100,        # suppress idle novelty seeking
        "vision": 1,
        "gaze": 1,
        "touch": 1,
        "proprioception": 1,
        "motor_watch": 1,
        "hazard": 1,
        "capability": 1,
        "outcome": 2,
        "ipc": 2,
        "evidence": 1,
        "power": 4,
        "memory": 100,
        "maintenance": 100,
    },
}

# Synthetic-adrenaline hysteresis. Emergency decays to alert before normal.
EMERGENCY_HOLD_S = 3.0
ALERT_RECOVERY_S = 5.0
ALERT_HAZARD_LEVEL = 2
EMERGENCY_HAZARD_LEVEL = 3

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

AROUSAL_STATE_TOPIC = "body/arousal_state"
SCENE_TOPIC = "object/scene"
VISION_DELTA_TOPIC = "vision/object_delta"
HAZARD_TOPIC = "hazard/report"
VALID_MODES = {"normal", "alert", "emergency"}
EMERGENCY_CLASSIFIERS = {
    "emergency",
    "hazard",
    "danger",
    "dangerous",
    "safety_hazard",
    "state.emergency",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class BodyAdrenalineSchedulerNeuron(BaseNeuron):
    """Body-side cadence allocator / synthetic sympathetic response.

    Only this scheduler consumes the raw canonical heartbeat.  It derives
    target-specific ``body/service/<target>`` opportunities on the isolated body
    bus.  Relevant danger changes cadence; the 20-TPS pacemaker itself remains
    fixed.  Meaningful arousal transitions are emitted onto the normal bus.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.monotonic()

        if event.topic == HAZARD_TOPIC:
            return await self._ingest_hazard(event, ctx, now)
        if event.topic == VISION_DELTA_TOPIC:
            return await self._ingest_vision_delta(event, ctx, now)
        if event.topic == SCENE_TOPIC:
            return await self._ingest_scene(event, ctx, now)
        if event.topic != PRIMARY_HEARTBEAT_TOPIC:
            return []

        heartbeat = event.payload if isinstance(event.payload, Mapping) else {}
        tick = _safe_int(heartbeat.get("tick", 0), 0)
        if tick <= 0:
            return []

        mode_before = str(await ctx.get_kv("body:arousal_mode", "normal") or "normal")
        mode = await self._resolve_mode(ctx, now)
        if mode not in VALID_MODES:
            mode = "normal"

        profile = await self._profile(ctx, mode)
        await ctx.set_kv("body:arousal_mode", mode)
        await ctx.set_kv("body:organ_cadence", profile)

        state = await self._state_payload(ctx, mode, profile, heartbeat, now)
        await ctx.set_kv("body:adrenaline", state)

        outputs: list[Event] = []
        if mode != mode_before:
            transition_state = dict(state)
            transition_state["transition_from"] = mode_before
            outputs.append(self._arousal_event(transition_state, reason="recovery_transition"))

        # Emit only service targets with live subscribers. This keeps the body
        # stream compact while allowing future organs to opt in simply by
        # subscribing to body/service/<target>.
        active_targets = await ctx.get_kv("body:service_targets", [])
        if not isinstance(active_targets, (list, tuple, set)):
            active_targets = []
        active = {str(target) for target in active_targets}

        for target, divisor in profile.items():
            divisor = max(1, int(divisor))
            if target not in active or tick % divisor != 0:
                continue
            outputs.append(
                Event(
                    topic=service_topic(target),
                    payload=service_tick_payload(
                        heartbeat,
                        target=target,
                        mode=mode,
                        divisor=divisor,
                    ),
                    source=self.name,
                    # Intentionally no heartbeat correlation propagation.
                    meta=service_tick_meta(target),
                )
            )

        return outputs

    async def _ingest_hazard(self, event: Event, ctx, now: float) -> list[Event]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        level = _safe_int(payload.get("level", 0), 0)
        reason = str(payload.get("reason") or payload.get("tag") or "hazard_report")
        if level >= EMERGENCY_HAZARD_LEVEL:
            return await self._trigger(ctx, now, "emergency", reason, level=level, source_event=event)
        if level >= ALERT_HAZARD_LEVEL:
            return await self._trigger(ctx, now, "alert", reason, level=level, source_event=event)
        return []

    async def _ingest_vision_delta(self, event: Event, ctx, now: float) -> list[Event]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        deltas = payload.get("deltas", []) if isinstance(payload.get("deltas"), list) else []
        for delta in deltas:
            if not isinstance(delta, Mapping):
                continue
            quorum = delta.get("quorum", {}) if isinstance(delta.get("quorum"), Mapping) else {}
            if bool(quorum.get("emergency_override", False)):
                return await self._trigger(
                    ctx,
                    now,
                    "emergency",
                    "vision_emergency_override",
                    level=EMERGENCY_HAZARD_LEVEL,
                    source_event=event,
                )
        return []

    async def _ingest_scene(self, event: Event, ctx, now: float) -> list[Event]:
        scene = event.payload if isinstance(event.payload, Mapping) else {}
        state = scene.get("state", {}) if isinstance(scene.get("state"), Mapping) else {}
        classifiers = {
            str(value).strip().lower()
            for value in scene.get("classifiers", []) or []
            if str(value).strip()
        }
        emergency = bool(state.get("emergency", False)) or bool(classifiers & EMERGENCY_CLASSIFIERS)
        if emergency:
            return await self._trigger(
                ctx,
                now,
                "emergency",
                "scene_emergency",
                level=_safe_int(state.get("hazard_level", EMERGENCY_HAZARD_LEVEL), EMERGENCY_HAZARD_LEVEL),
                source_event=event,
            )
        if bool(state.get("alert", False)):
            return await self._trigger(
                ctx,
                now,
                "alert",
                "scene_alert",
                level=ALERT_HAZARD_LEVEL,
                source_event=event,
            )
        return []

    async def _trigger(
        self,
        ctx,
        now: float,
        requested_mode: str,
        reason: str,
        *,
        level: int,
        source_event: Event,
    ) -> list[Event]:
        previous = str(await ctx.get_kv("body:arousal_mode", "normal") or "normal")
        emergency_until = _safe_float(await ctx.get_kv("body:adrenaline:emergency_until", 0.0), 0.0)
        alert_until = _safe_float(await ctx.get_kv("body:adrenaline:alert_until", 0.0), 0.0)

        if requested_mode == "emergency":
            emergency_until = max(emergency_until, now + EMERGENCY_HOLD_S)
            alert_until = max(alert_until, emergency_until + ALERT_RECOVERY_S)
        elif requested_mode == "alert":
            alert_until = max(alert_until, now + ALERT_RECOVERY_S)

        await ctx.set_kv("body:adrenaline:emergency_until", emergency_until)
        await ctx.set_kv("body:adrenaline:alert_until", alert_until)
        await ctx.set_kv(
            "body:adrenaline:last_trigger",
            {
                "mode": requested_mode,
                "reason": reason,
                "level": int(level),
                "source_topic": source_event.topic,
                "source": source_event.source,
                "ts_monotonic": now,
                "ts_epoch": time.time(),
            },
        )

        resolved = await self._resolve_mode(ctx, now)
        await ctx.set_kv("body:arousal_mode", resolved)
        profile = await self._profile(ctx, resolved)
        await ctx.set_kv("body:organ_cadence", profile)
        state = await self._state_payload(ctx, resolved, profile, {}, now)
        await ctx.set_kv("body:adrenaline", state)

        # Repeated hazard reports extend the hold silently.  The original hazard
        # event already exists on the meaningful bus; only an actual body-mode
        # transition deserves another cognitive/engineering event.
        if resolved == previous:
            return []

        state["transition_from"] = previous
        return [self._arousal_event(state, reason=reason)]

    async def _resolve_mode(self, ctx, now: float) -> str:
        emergency_until = _safe_float(await ctx.get_kv("body:adrenaline:emergency_until", 0.0), 0.0)
        alert_until = _safe_float(await ctx.get_kv("body:adrenaline:alert_until", 0.0), 0.0)
        if now < emergency_until:
            return "emergency"
        if now < alert_until:
            return "alert"
        return "normal"

    async def _profile(self, ctx, mode: str) -> dict[str, int]:
        default = dict(CADENCE_PROFILES.get(mode, CADENCE_PROFILES["normal"]))
        override = await ctx.get_kv(f"body:cadence_profile:{mode}", {})
        if isinstance(override, Mapping):
            for target, value in override.items():
                divisor = _safe_int(value, default.get(str(target), 1))
                if divisor > 0:
                    default[str(target)] = divisor
        return default

    async def _state_payload(
        self,
        ctx,
        mode: str,
        profile: Mapping[str, int],
        heartbeat: Mapping[str, Any],
        now: float,
    ) -> dict[str, Any]:
        trigger = await ctx.get_kv("body:adrenaline:last_trigger", {})
        if not isinstance(trigger, Mapping):
            trigger = {}
        return {
            "schema": "body.adrenaline.v2",
            "mode": mode,
            "tick": _safe_int(heartbeat.get("tick", 0), 0),
            "ts_monotonic": now,
            "ts_epoch": time.time(),
            "cadence_divisors": dict(profile),
            "emergency_until": _safe_float(await ctx.get_kv("body:adrenaline:emergency_until", 0.0), 0.0),
            "alert_until": _safe_float(await ctx.get_kv("body:adrenaline:alert_until", 0.0), 0.0),
            "last_trigger": dict(trigger),
            "policy": "fixed_20tps_selective_organ_surge",
        }

    def _arousal_event(self, state: Mapping[str, Any], *, reason: str) -> Event:
        payload = dict(state)
        payload["reason"] = reason
        return Event(
            topic=AROUSAL_STATE_TOPIC,
            payload=payload,
            source=self.name,
            meta={
                "kind": "body_arousal_state",
                "channel": "body",
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "self_output_track": False,
                "cognitive_visible": True,
            },
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            PRIMARY_HEARTBEAT_TOPIC,
            HAZARD_TOPIC,
            VISION_DELTA_TOPIC,
            SCENE_TOPIC,
        ],
        output_topics=[AROUSAL_STATE_TOPIC],
        priority=35,
        cooldown_sec=0.0,
    )
    yield BodyAdrenalineSchedulerNeuron(cfg)
