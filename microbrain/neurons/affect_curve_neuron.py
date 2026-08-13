from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.affect_curves import decay_curve_map, signed_feedback_curve, safe_float
from microbrain.hormone import derive_ddna_modulators
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("affect_curve")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_from_event(event: Event) -> str:
    payload = event.payload
    if isinstance(payload, dict):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return str(payload or "").strip()


def _raw_meta(event: Event) -> dict[str, Any]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    raw = payload.get("raw_meta") if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return raw
    return dict(event.meta or {})


def _fingerprint(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))[:160]


def _signed_acc_from_event(event: Event) -> float:
    raw = _raw_meta(event)
    meta = dict(event.meta or {})
    positive = max(0.0, safe_float(raw.get("accent_positive", meta.get("accent_positive", 0.0)), 0.0))
    negative = max(0.0, safe_float(raw.get("accent_negative_severity", meta.get("accent_negative_severity", 0.0)), 0.0))
    if positive > 0.0:
        return min(10.0, positive)
    if negative > 0.0:
        return -min(10.0, negative)
    value = raw.get("accent_value", meta.get("accent_value", 0.0))
    return max(-10.0, min(10.0, safe_float(value, 0.0)))


class AffectCurveNeuron(BaseNeuron):
    """
    Maintains time-decaying affect curves with flow/capacity saturation.

    This is the field-theory substrate for /acc feedback and later edge arousal:
    emotions bend outcome probability and telemetry, but do not select actions or
    bypass the governor. It runs beside the existing reward/novelty pulse organ
    so older consumers keep working while newer organs can read affect:curve_state.
    """

    STATE_KEY = "affect_curve_state"

    async def _profile_math(self, ctx) -> dict[str, float]:
        mods = await ctx.get_kv("drive:ddna_modulators", None)
        if isinstance(mods, dict) and mods:
            return dict(mods)
        pdna = await ctx.get_kv("pdna:profile", None)
        mods = derive_ddna_modulators(pdna)
        await ctx.set_kv("drive:ddna_modulators", mods)
        return dict(mods or {})

    async def _load_state(self, ctx, now: float) -> dict[str, Any]:
        state = await self.load_state(
            ctx,
            self.STATE_KEY,
            default={"curves": {}, "last_ts": now, "last_reason": "idle", "last_effect": {}},
        )
        state = dict(state or {})
        state["curves"] = decay_curve_map(_as_dict(state.get("curves")), now=now)
        state["last_ts"] = now
        return state

    async def _persist(self, ctx, state: dict[str, Any], *, reason: str, effect: Mapping[str, Any], source_topic: str) -> None:
        now = safe_float(state.get("last_ts"), time.time())
        state["last_reason"] = reason
        state["last_effect"] = dict(effect or {})
        state["source_topic"] = source_topic
        await self.save_state(ctx, self.STATE_KEY, state)
        await ctx.set_kv(
            "affect:curve_state",
            {
                "curves": dict(state.get("curves") or {}),
                "last_reason": reason,
                "last_effect": dict(effect or {}),
                "source_topic": source_topic,
                "ts": now,
            },
        )

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in {
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "control/reinforce",
            "control/trainer_correction",
        }:
            return []

        now = time.time()
        state = await self._load_state(ctx, now)
        mods = await self._profile_math(ctx)
        reason = "decay"
        effect: dict[str, Any] = {}

        if event.topic == SERVICE_TOPIC:
            await self._persist(ctx, state, reason=reason, effect=effect, source_topic=event.topic)
            return []

        if event.topic == "percept/text":
            signed = _signed_acc_from_event(event)
            if signed != 0.0:
                target = _fingerprint(_text_from_event(event)) or str(event.correlation_id or "")
                next_curves, effect = signed_feedback_curve(
                    _as_dict(state.get("curves")),
                    signed_strength=signed,
                    now=now,
                    ddna=mods,
                    target_key=target,
                    target_confidence=1.0,
                    novelty=1.0,
                    source="percept/text:/acc",
                )
                state["curves"] = next_curves
                reason = "positive_accent_curve" if signed > 0 else "negative_accent_curve"

        elif event.topic == "control/reinforce":
            payload = _as_dict(event.payload)
            signed = safe_float(payload.get("weight", payload.get("score", payload.get("acc"))), 0.0)
            if signed != 0.0:
                target = str(payload.get("target_key", payload.get("target", "")) or "")[:180]
                next_curves, effect = signed_feedback_curve(
                    _as_dict(state.get("curves")),
                    signed_strength=signed,
                    now=now,
                    ddna=mods,
                    target_key=target,
                    target_confidence=max(0.0, min(1.0, safe_float(payload.get("target_confidence"), 1.0))),
                    novelty=max(0.0, min(1.0, safe_float(payload.get("novelty"), 1.0))),
                    source="control/reinforce",
                )
                state["curves"] = next_curves
                reason = "positive_reinforcement_curve" if signed > 0 else "negative_reinforcement_curve"

        elif event.topic == "control/trainer_correction":
            next_curves, effect = signed_feedback_curve(
                _as_dict(state.get("curves")),
                signed_strength=4.0,
                now=now,
                ddna=mods,
                target_key=str(event.correlation_id or "trainer_correction"),
                target_confidence=0.85,
                novelty=0.8,
                source="control/trainer_correction",
            )
            state["curves"] = next_curves
            reason = "trainer_alignment_curve"

        elif event.topic == "percept/vision":
            # Vision edges are cheap arousal: motion/salience can raise the curve
            # without forcing any behavior. Later outcome-field organs can use it.
            payload = _as_dict(event.payload)
            motion = max(0.0, safe_float(payload.get("motion", payload.get("motion_score", 0.0)), 0.0))
            salience = max(0.0, safe_float(payload.get("salience", payload.get("score", 0.0)), 0.0))
            edge_amount = max(motion, salience)
            if edge_amount > 0.0:
                from microbrain.affect_curves import apply_curve_pulse

                next_curves, pulse = apply_curve_pulse(
                    _as_dict(state.get("curves")),
                    name="arousal",
                    signed_amount=max(0.0, min(1.0, edge_amount)),
                    now=now,
                    ddna_gain=max(0.2, min(2.5, safe_float(mods.get("arousal_gain"), 1.0))),
                    target_confidence=max(0.0, min(1.0, safe_float(payload.get("confidence"), 0.65))),
                    novelty=1.0,
                    target_key=str(payload.get("object_id", payload.get("vobj_id", "")) or "")[:180],
                    source="percept/vision:edge",
                )
                state["curves"] = next_curves
                effect = {
                    "curve_name": "arousal",
                    "effective_reward": 0.0,
                    "arousal_delta": round(max(0.0, pulse.effective), 4),
                    "saturation": round(pulse.saturation, 4),
                    "overload": round(pulse.overload, 4),
                    "flow_available": round(pulse.flow_available, 4),
                    "capacity_remaining": round(pulse.capacity_remaining, 4),
                    "pulse": pulse.to_dict(),
                }
                reason = "vision_edge_arousal_curve"

        await self._persist(ctx, state, reason=reason, effect=effect, source_topic=event.topic)
        if not effect and reason == "decay":
            return []

        meta = {"ui_visible": False, "store_in_memory": False, "cognitive_visible": False}
        return [
            Event(
                topic="affect/curve_state",
                payload={
                    "curves": dict(state.get("curves") or {}),
                    "reason": reason,
                    "effect": dict(effect or {}),
                    "ts": now,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta=meta,
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "control/reinforce",
            "control/trainer_correction",
        ],
        output_topics=["affect/curve_state"],
        priority=4,
    )
    yield AffectCurveNeuron(cfg)
