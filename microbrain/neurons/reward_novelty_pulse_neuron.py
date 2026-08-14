from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.affect_curves import (
    AffectCurveConfig,
    apply_affect_pulse,
    decay_curve_bucket,
    summarize_curve_bucket,
)
from microbrain.hormone import derive_ddna_modulators
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.pdna.access import profile_path

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("affect")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp_signed(value: float) -> float:
    return _clamp(value, -1.0, 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def _is_internal_or_control(event: Event) -> bool:
    meta = dict(event.meta or {})
    payload = event.payload if isinstance(event.payload, dict) else {}
    raw = _raw_meta(event)
    channel = str(payload.get("channel", "") or meta.get("channel", "") or raw.get("channel", "") or "")
    source = str(payload.get("source", "") or meta.get("source", "") or raw.get("source", "") or "")
    text = _text_from_event(event)
    return (
        bool(meta.get("control", False))
        or bool(raw.get("control", False))
        or bool(meta.get("cognitive_visible") is False)
        or bool(raw.get("cognitive_visible") is False)
        or text.lstrip().startswith("/")
        or event.topic.startswith("ui/")
        or event.topic.startswith("control/")
        or channel in {"internal", "thought"}
        or source in {"internal", "assistant", "system"}
    )


def _fingerprint(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))[:160]


def _event_payload_dict(event: Event) -> dict[str, Any]:
    return event.payload if isinstance(event.payload, dict) else {}


def _motion_strength_from_event(event: Event, raw: Mapping[str, Any]) -> float:
    payload = _event_payload_dict(event)
    candidates = (
        payload.get("motion_salience"),
        payload.get("motion_score"),
        payload.get("motion_intensity"),
        payload.get("motion"),
        payload.get("salience"),
        raw.get("motion_salience"),
        raw.get("motion_score"),
        raw.get("motion_intensity"),
        raw.get("salience"),
    )
    return _clamp01(max((_safe_float(v, 0.0) for v in candidates), default=0.0))


def _target_key_from_event(
    event: Event,
    raw: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    fallback_text: str = "",
) -> str:
    payload = _event_payload_dict(event)
    for key in (
        "target_id",
        "target",
        "trace_id",
        "selected_vobj_id",
        "vobj_id",
        "object_id",
        "source_id",
    ):
        value = payload.get(key, raw.get(key))
        if value:
            return str(value)[:160]

    # /acc feedback should usually point at what MB just did, not the literal
    # words "great job".  Until the rolling-index target binder is richer, use
    # recent MB output/vision fingerprints before falling back to user text.
    for key in ("last_output_fp", "last_vision_fp"):
        value = state.get(key)
        if value:
            return str(value)[:160]

    fp = _fingerprint(fallback_text or event.topic)
    return fp or event.topic


class RewardNoveltyPulseNeuron(BaseNeuron):
    """
    Maintains fast reward/novelty/valence pulses for the UI pressure band and drives.

    DDNA v2 wiring: the active PDNA/DDNA profile mutates gain, decay,
    salience, boredom relief, and trainer reward sensitivity. This stays an
    affect organ only; it does not write responses or command actions.

    Hormone Curve Field v1.1: curve math augments this existing organ instead
    of replacing it.  The curve layer shapes incoming pulses through flow,
    capacity, saturation, overload, and repeat dampening, then this neuron keeps
    publishing the authoritative affect/reward and affect/salience events.
    """

    REWARD_DECAY_PER_S = 0.20
    NOVELTY_DECAY_PER_S = 0.055
    RELIEF_DECAY_PER_S = 0.040
    SALIENCE_DECAY_PER_S = 0.045
    VALENCE_DECAY_PER_S = 0.035
    SATISFACTION_DECAY_PER_S = 0.010

    async def _profile_math(self, ctx) -> tuple[Any, dict[str, float]]:
        pdna = await ctx.get_kv("pdna:profile", None)
        reinforcement_model = await ctx.get_kv("pdna:reinforcement_model", None)
        # If the boot publisher exposed pdna:reinforcement_model, prefer that
        # live KV section so hot-swapped profiles can affect reward math before
        # the next full profile reload.
        if isinstance(reinforcement_model, Mapping):
            try:
                pdna_view = dict(pdna.to_dict()) if hasattr(pdna, "to_dict") else dict(pdna or {})
                pdna_view["reinforcement_model"] = dict(reinforcement_model)
                pdna = pdna_view
            except Exception:
                pass
        mods = await ctx.get_kv("drive:ddna_modulators", None)
        if not isinstance(mods, dict) or not mods:
            mods = derive_ddna_modulators(pdna)
            await ctx.set_kv("drive:ddna_modulators", mods)
        return pdna, dict(mods or {})

    def _decays(self, pdna: Any, mods: Mapping[str, Any]) -> dict[str, float]:
        resistance = _clamp(_safe_float(mods.get("decay_resistance"), 1.0), 0.35, 2.00)
        salience_resistance = _clamp(_safe_float(mods.get("salience_decay_resistance"), resistance), 0.35, 2.00)
        return {
            "reward": max(0.001, _safe_float(profile_path(pdna, "affect_model", "decay.dopamine_decay_per_second", self.REWARD_DECAY_PER_S), self.REWARD_DECAY_PER_S) / resistance),
            "novelty": max(0.001, _safe_float(profile_path(pdna, "affect_model", "decay.curiosity_decay_per_second", self.NOVELTY_DECAY_PER_S), self.NOVELTY_DECAY_PER_S) / resistance),
            "relief": max(0.001, self.RELIEF_DECAY_PER_S / resistance),
            "salience": max(0.001, _safe_float(profile_path(pdna, "affect_model", "decay.salience_decay_per_second", self.SALIENCE_DECAY_PER_S), self.SALIENCE_DECAY_PER_S) / salience_resistance),
            "valence": max(0.001, _safe_float(profile_path(pdna, "affect_model", "decay.valence_decay_per_second", self.VALENCE_DECAY_PER_S), self.VALENCE_DECAY_PER_S) / resistance),
            "satisfaction": max(0.001, _safe_float(profile_path(pdna, "affect_model", "decay.satisfaction_decay_per_second", self.SATISFACTION_DECAY_PER_S), self.SATISFACTION_DECAY_PER_S) / resistance),
        }

    def _curve_configs(self, pdna: Any, mods: Mapping[str, Any]) -> dict[str, AffectCurveConfig]:
        reward_gain = _clamp(_safe_float(mods.get("reward_gain"), 1.0), 0.25, 2.00)
        salience_gain = _clamp(_safe_float(mods.get("salience_gain"), 1.0), 0.25, 2.20)
        trainer_gain = _clamp(_safe_float(mods.get("trainer_alignment_gain"), 1.0), 0.25, 2.00)
        decay_resistance = _clamp(_safe_float(mods.get("decay_resistance"), 1.0), 0.35, 2.00)

        return {
            "user_approval": AffectCurveConfig(
                name="user_approval",
                flow_threshold=_safe_float(profile_path(pdna, "affect_model", "curves.user_approval.flow_threshold", 1.15), 1.15) * trainer_gain,
                flow_window_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_approval.flow_window_s", 10.0), 10.0),
                curve_capacity=_safe_float(profile_path(pdna, "affect_model", "curves.user_approval.curve_capacity", 1.0), 1.0) * reward_gain,
                decay_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_approval.decay_half_life_s", 20.0), 20.0) * decay_resistance,
                overload_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_approval.overload_half_life_s", 12.0), 12.0) * decay_resistance,
                repeat_window_s=6.0,
                repeat_dampening=0.18,
            ),
            "user_correction": AffectCurveConfig(
                name="user_correction",
                flow_threshold=_safe_float(profile_path(pdna, "affect_model", "curves.user_correction.flow_threshold", 1.05), 1.05),
                flow_window_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_correction.flow_window_s", 12.0), 12.0),
                curve_capacity=_safe_float(profile_path(pdna, "affect_model", "curves.user_correction.curve_capacity", 0.95), 0.95),
                decay_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_correction.decay_half_life_s", 24.0), 24.0) * decay_resistance,
                overload_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.user_correction.overload_half_life_s", 14.0), 14.0) * decay_resistance,
                repeat_window_s=7.0,
                repeat_dampening=0.14,
            ),
            "arousal": AffectCurveConfig(
                name="arousal",
                flow_threshold=_safe_float(profile_path(pdna, "affect_model", "curves.arousal.flow_threshold", 1.35), 1.35) * salience_gain,
                flow_window_s=_safe_float(profile_path(pdna, "affect_model", "curves.arousal.flow_window_s", 6.0), 6.0),
                curve_capacity=_safe_float(profile_path(pdna, "affect_model", "curves.arousal.curve_capacity", 1.15), 1.15),
                decay_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.arousal.decay_half_life_s", 8.0), 8.0) * decay_resistance,
                overload_half_life_s=_safe_float(profile_path(pdna, "affect_model", "curves.arousal.overload_half_life_s", 6.0), 6.0) * decay_resistance,
                repeat_window_s=3.0,
                repeat_dampening=0.10,
            ),
        }

    def _shape_curve(
        self,
        state: dict[str, Any],
        pdna: Any,
        mods: Mapping[str, Any],
        curve_name: str,
        *,
        now: float,
        incoming_strength: float,
        target_key: str,
        target_confidence: float = 1.0,
        novelty: float = 1.0,
        ddna_gain: float = 1.0,
    ):
        configs = self._curve_configs(pdna, mods)
        cfg = configs[curve_name]
        curves = _as_dict(state.get("curves"))
        result = apply_affect_pulse(
            curves.get(curve_name),
            cfg,
            now=now,
            incoming_strength=incoming_strength,
            target_key=target_key,
            target_confidence=target_confidence,
            novelty=novelty,
            ddna_gain=ddna_gain,
        )
        curves[curve_name] = result.state
        state["curves"] = curves
        state["last_curve_result"] = result.to_dict()
        return result

    async def _load_state(self, ctx, now: float) -> dict[str, Any]:
        state = await self.load_state(
            ctx,
            "reward_novelty_state",
            default={
                "reward": 0.0,
                "novelty": 0.0,
                "boredom_relief": 0.0,
                "salience": 0.0,
                "valence": 0.0,
                "satisfaction": 0.0,
                "curves": {},
                "last_curve_result": {},
                "last_ts": now,
                "last_user_fp": "",
                "last_vision_fp": "",
                "last_output_fp": "",
                "last_reason": "idle",
                "last_delta": 0.0,
                "last_reward_ts": 0.0,
                "same_reward_count": 0,
            },
        )
        return dict(state or {})

    def _decay_state(self, state: dict[str, Any], now: float, decays: Mapping[str, float], curve_configs: Mapping[str, AffectCurveConfig]) -> dict[str, Any]:
        last_ts = _safe_float(state.get("last_ts"), now)
        dt = max(0.0, now - last_ts)
        state["reward"] = _clamp01(_safe_float(state.get("reward"), 0.0) - (_safe_float(decays.get("reward"), self.REWARD_DECAY_PER_S) * dt))
        state["novelty"] = _clamp01(_safe_float(state.get("novelty"), 0.0) - (_safe_float(decays.get("novelty"), self.NOVELTY_DECAY_PER_S) * dt))
        state["boredom_relief"] = _clamp01(_safe_float(state.get("boredom_relief"), 0.0) - (_safe_float(decays.get("relief"), self.RELIEF_DECAY_PER_S) * dt))
        state["salience"] = _clamp01(_safe_float(state.get("salience"), 0.0) - (_safe_float(decays.get("salience"), self.SALIENCE_DECAY_PER_S) * dt))
        val = _safe_float(state.get("valence"), 0.0)
        val_decay = _safe_float(decays.get("valence"), self.VALENCE_DECAY_PER_S) * dt
        if val > 0.0:
            val = max(0.0, val - val_decay)
        elif val < 0.0:
            val = min(0.0, val + val_decay)
        state["valence"] = _clamp_signed(val)
        state["satisfaction"] = _clamp01(_safe_float(state.get("satisfaction"), 0.0) - (_safe_float(decays.get("satisfaction"), self.SATISFACTION_DECAY_PER_S) * dt))
        state["curves"] = decay_curve_bucket(_as_dict(state.get("curves")), curve_configs, now=now)
        state["last_ts"] = now
        return state

    def _positive_reinforcement_gains(self, pdna: Any, mods: Mapping[str, Any], weight: float, now: float, state: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        # /acc is a 1..10 teaching pulse. Older code often stored 1..5; both
        # work, but 5 and 10 are no longer flattened into the same spike.
        acc = _clamp(weight, 0.0, 10.0)
        reward_gain = _clamp(_safe_float(mods.get("reward_gain"), 1.0), 0.25, 2.00)
        salience_gain = _clamp(_safe_float(mods.get("salience_gain"), 1.0), 0.25, 2.20)
        relief_gain = _clamp(_safe_float(mods.get("boredom_relief_gain"), 1.0), 0.25, 2.00)
        trainer_gain = _clamp(_safe_float(mods.get("trainer_alignment_gain"), 1.0), 0.25, 2.00)

        dopamine = min(0.72, 0.05 + (acc * 0.055)) * reward_gain
        salience = min(0.55, 0.04 + (acc * 0.045)) * salience_gain
        relief = min(0.42, 0.04 + (acc * 0.035)) * relief_gain
        satisfaction = min(0.28, 0.02 + (acc * 0.025)) * trainer_gain
        valence = min(0.62, 0.04 + (acc * 0.045)) * reward_gain

        # Legacy anti-button-mashing remains as a secondary guard.  The curve
        # layer does the richer flow/capacity/repeat math before this method is
        # called, but this prevents old direct reinforce paths from spiking too hard.
        last_reward_ts = _safe_float(state.get("last_reward_ts"), 0.0)
        same_reward_count = int(state.get("same_reward_count", 0) or 0)
        if last_reward_ts > 0.0 and (now - last_reward_ts) < 4.0:
            damp = max(0.35, 1.0 - (0.14 * min(4, same_reward_count + 1)))
            dopamine *= damp
            salience *= damp
            relief *= damp
            satisfaction *= damp
            valence *= damp
        return dopamine, salience, relief, satisfaction, valence

    async def _persist(self, ctx, state: dict[str, Any], *, reason: str, delta: float, source_topic: str) -> None:
        now = _safe_float(state.get("last_ts"), time.time())
        state["last_reason"] = reason
        state["last_delta"] = round(delta, 4)
        await self.save_state(ctx, "reward_novelty_state", state)
        reward_level = _clamp01(_safe_float(state.get("reward"), 0.0))
        novelty_level = _clamp01(_safe_float(state.get("novelty"), 0.0))
        relief_level = _clamp01(_safe_float(state.get("boredom_relief"), 0.0))
        salience_level = _clamp01(_safe_float(state.get("salience"), 0.0))
        valence_level = _clamp_signed(_safe_float(state.get("valence"), 0.0))
        satisfaction_level = _clamp01(_safe_float(state.get("satisfaction"), 0.0))
        curve_summary = summarize_curve_bucket(_as_dict(state.get("curves")))
        curve_last = _as_dict(state.get("last_curve_result"))

        await ctx.set_kv(
            "affect:reward_state",
            {
                "level": round(reward_level, 4),
                "dopamine": round(reward_level, 4),
                "valence": round(valence_level, 4),
                "satisfaction": round(satisfaction_level, 4),
                "last_delta": round(delta, 4),
                "reason": reason,
                "source_topic": source_topic,
                "curve_last": curve_last,
                "ts": now,
            },
        )
        await ctx.set_kv(
            "affect:novelty_state",
            {
                "level": round(novelty_level, 4),
                "boredom_relief": round(relief_level, 4),
                "reason": reason,
                "source_topic": source_topic,
                "ts": now,
            },
        )
        await ctx.set_kv(
            "affect:salience_state",
            {
                "level": round(salience_level, 4),
                "reason": reason,
                "source_topic": source_topic,
                "curve_last": curve_last,
                "ts": now,
            },
        )
        await ctx.set_kv(
            "affect:curve_state",
            {
                "curves": curve_summary,
                "last": curve_last,
                "reason": reason,
                "source_topic": source_topic,
                "ts": now,
            },
        )
        await ctx.set_kv(
            "drive:boredom_relief",
            {
                "level": round(relief_level, 4),
                "reason": reason,
                "source_topic": source_topic,
                "ts": now,
            },
        )

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in {
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "act/speech",
            "control/reinforce",
            "control/trainer_correction",
        }:
            return []

        now = time.time()
        pdna, mods = await self._profile_math(ctx)
        curve_configs = self._curve_configs(pdna, mods)
        state = self._decay_state(await self._load_state(ctx, now), now, self._decays(pdna, mods), curve_configs)

        reward_delta = 0.0
        novelty_delta = 0.0
        relief_delta = 0.0
        salience_delta = 0.0
        valence_delta = 0.0
        satisfaction_delta = 0.0
        reason = "decay"

        novelty_gain = _clamp(_safe_float(mods.get("novelty_gain"), 1.0), 0.25, 2.00)
        reward_gain = _clamp(_safe_float(mods.get("reward_gain"), 1.0), 0.25, 2.00)
        salience_gain = _clamp(_safe_float(mods.get("salience_gain"), 1.0), 0.25, 2.20)
        relief_gain = _clamp(_safe_float(mods.get("boredom_relief_gain"), 1.0), 0.25, 2.00)
        trainer_gain = _clamp(_safe_float(mods.get("trainer_alignment_gain"), 1.0), 0.25, 2.00)

        if event.topic == SERVICE_TOPIC:
            await self._persist(ctx, state, reason=reason, delta=0.0, source_topic=event.topic)
            return []

        if event.topic == "control/reinforce":
            payload = _as_dict(event.payload)
            weight = _safe_float(payload.get("weight", payload.get("score", payload.get("acc"))), 0.0)
            target_key = _target_key_from_event(event, payload, state, fallback_text=payload.get("text", event.topic))
            if weight > 0.0:
                raw_strength = _clamp01(weight / 10.0)
                curve = self._shape_curve(
                    state,
                    pdna,
                    mods,
                    "user_approval",
                    now=now,
                    incoming_strength=raw_strength,
                    target_key=target_key,
                    target_confidence=_clamp01(_safe_float(payload.get("target_confidence"), 1.0)),
                    novelty=1.0,
                    ddna_gain=reward_gain * trainer_gain,
                )
                shaped_weight = min(10.0, weight * curve.multiplier)
                reward_delta, salience_delta, relief_delta, satisfaction_delta, valence_delta = self._positive_reinforcement_gains(pdna, mods, shaped_weight, now, state)
                novelty_delta += (0.06 + (0.018 * min(shaped_weight, 10.0))) * novelty_gain
                state["last_reward_ts"] = now
                state["same_reward_count"] = int(state.get("same_reward_count", 0) or 0) + 1
                reason = f"positive_reinforcement:{curve.reason}"
            elif weight < 0.0:
                raw_strength = _clamp01(abs(weight) / 10.0)
                curve = self._shape_curve(
                    state,
                    pdna,
                    mods,
                    "user_correction",
                    now=now,
                    incoming_strength=raw_strength,
                    target_key=target_key,
                    target_confidence=_clamp01(_safe_float(payload.get("target_confidence"), 1.0)),
                    novelty=1.0,
                    ddna_gain=trainer_gain,
                )
                scaled = -curve.effective_strength
                reward_delta += 0.45 * scaled
                valence_delta += 0.52 * scaled
                salience_delta += (0.10 + (0.12 * curve.effective_strength)) * salience_gain
                novelty_delta += 0.06 * novelty_gain
                state["same_reward_count"] = 0
                reason = f"negative_reinforcement:{curve.reason}"

        elif event.topic == "control/trainer_correction":
            target_key = _target_key_from_event(event, _as_dict(event.payload), state, fallback_text="trainer_correction")
            curve = self._shape_curve(
                state,
                pdna,
                mods,
                "user_correction",
                now=now,
                incoming_strength=0.55,
                target_key=target_key,
                target_confidence=1.0,
                novelty=1.0,
                ddna_gain=trainer_gain,
            )
            strength = curve.effective_strength
            reward_delta += (0.12 + (0.18 * strength)) * reward_gain * trainer_gain
            salience_delta += (0.10 + (0.16 * strength)) * salience_gain
            novelty_delta += (0.08 + (0.10 * strength)) * novelty_gain
            relief_delta += (0.10 + (0.12 * strength)) * relief_gain
            satisfaction_delta += 0.08 * strength * trainer_gain
            valence_delta += 0.18 * strength * reward_gain
            state["same_reward_count"] = 0
            reason = f"trainer_correction:{curve.reason}"

        elif event.topic in {"percept/text", "percept/vision"} and not _is_internal_or_control(event):
            raw = _raw_meta(event)
            payload = _event_payload_dict(event)
            accent_positive = max(0.0, _safe_float(raw.get("accent_positive", event.meta.get("accent_positive") if event.meta else 0.0), 0.0))
            accent_negative = max(0.0, _safe_float(raw.get("accent_negative_severity", event.meta.get("accent_negative_severity") if event.meta else 0.0), 0.0))
            text = _text_from_event(event)
            fp_key = "last_user_fp" if event.topic == "percept/text" else "last_vision_fp"
            fp = _fingerprint(text or event.topic)
            was_new = bool(fp and fp != str(state.get(fp_key, "") or ""))
            if fp:
                state[fp_key] = fp
            if was_new:
                reward_delta += 0.035 * reward_gain
                salience_delta += 0.12 * salience_gain
                novelty_delta += 0.16 * novelty_gain
                relief_delta += 0.18 * relief_gain
                valence_delta += 0.035 * reward_gain
                state["same_reward_count"] = 0
                reason = "novel_interaction"
            else:
                novelty_delta += 0.02 * novelty_gain
                salience_delta += 0.03 * salience_gain
                reason = "repeated_interaction"

            if event.topic == "percept/vision":
                motion_strength = _motion_strength_from_event(event, raw)
                if motion_strength > 0.0:
                    target_key = _target_key_from_event(event, raw, state, fallback_text=payload.get("label", "vision_motion"))
                    curve = self._shape_curve(
                        state,
                        pdna,
                        mods,
                        "arousal",
                        now=now,
                        incoming_strength=motion_strength,
                        target_key=target_key,
                        target_confidence=_clamp01(_safe_float(payload.get("confidence", raw.get("confidence", 0.75)), 0.75)),
                        novelty=1.0 if was_new else 0.70,
                        ddna_gain=salience_gain,
                    )
                    salience_delta += (0.06 + (0.20 * curve.effective_strength)) * salience_gain
                    novelty_delta += (0.02 + (0.08 * curve.effective_strength)) * novelty_gain
                    valence_delta += 0.01 * curve.effective_strength
                    reason = f"motion_arousal:{curve.reason}"

            if accent_positive > 0.0:
                raw_strength = _clamp01(accent_positive / 10.0)
                target_key = _target_key_from_event(event, raw, state, fallback_text=text)
                curve = self._shape_curve(
                    state,
                    pdna,
                    mods,
                    "user_approval",
                    now=now,
                    incoming_strength=raw_strength,
                    target_key=target_key,
                    target_confidence=_clamp01(_safe_float(raw.get("target_confidence", 1.0), 1.0)),
                    novelty=1.0 if was_new else 0.65,
                    ddna_gain=reward_gain * trainer_gain,
                )
                strength = curve.effective_strength
                reward_delta += (0.18 + (0.62 * strength)) * reward_gain
                salience_delta += (0.10 + (0.24 * strength)) * salience_gain
                novelty_delta += (0.08 + (0.10 * strength)) * novelty_gain
                relief_delta += (0.12 + (0.18 * strength)) * relief_gain
                valence_delta += (0.12 + (0.40 * strength)) * reward_gain
                satisfaction_delta += 0.08 * strength * trainer_gain
                reason = f"positive_accent:{curve.reason}"
            elif accent_negative > 0.0:
                raw_strength = _clamp01(accent_negative / 10.0)
                target_key = _target_key_from_event(event, raw, state, fallback_text=text)
                curve = self._shape_curve(
                    state,
                    pdna,
                    mods,
                    "user_correction",
                    now=now,
                    incoming_strength=raw_strength,
                    target_key=target_key,
                    target_confidence=_clamp01(_safe_float(raw.get("target_confidence", 1.0), 1.0)),
                    novelty=1.0 if was_new else 0.70,
                    ddna_gain=trainer_gain,
                )
                strength = curve.effective_strength
                reward_delta -= (0.10 + (0.28 * strength)) * reward_gain
                valence_delta -= (0.14 + (0.36 * strength))
                salience_delta += (0.12 + (0.16 * strength)) * salience_gain
                novelty_delta += 0.05 * novelty_gain
                reason = f"negative_accent:{curve.reason}"

        elif event.topic == "act/speech" and not _is_internal_or_control(event):
            fp = _fingerprint(_text_from_event(event))
            if fp and fp != str(state.get("last_output_fp", "") or ""):
                state["last_output_fp"] = fp
                reward_delta += 0.025 * reward_gain
                salience_delta += 0.05 * salience_gain
                novelty_delta += 0.07 * novelty_gain
                relief_delta += 0.05 * relief_gain
                valence_delta += 0.015 * reward_gain
                reason = "new_output_attempt"
            elif fp:
                reward_delta -= 0.025 * reward_gain
                novelty_delta -= 0.04
                valence_delta -= 0.02
                reason = "repeated_output"

        state["reward"] = _clamp01(_safe_float(state.get("reward"), 0.0) + reward_delta)
        state["novelty"] = _clamp01(_safe_float(state.get("novelty"), 0.0) + novelty_delta)
        state["boredom_relief"] = _clamp01(_safe_float(state.get("boredom_relief"), 0.0) + relief_delta)
        state["salience"] = _clamp01(_safe_float(state.get("salience"), 0.0) + salience_delta)
        state["valence"] = _clamp_signed(_safe_float(state.get("valence"), 0.0) + valence_delta)
        state["satisfaction"] = _clamp01(_safe_float(state.get("satisfaction"), 0.0) + satisfaction_delta)

        await self._persist(ctx, state, reason=reason, delta=reward_delta, source_topic=event.topic)

        meta = {"ui_visible": False, "store_in_memory": False, "cognitive_visible": False}
        curve_summary = summarize_curve_bucket(_as_dict(state.get("curves")))
        curve_last = _as_dict(state.get("last_curve_result"))
        events = [
            Event(
                topic="affect/reward",
                payload={
                    "reward": round(_clamp01(_safe_float(state.get("reward"), 0.0)), 4),
                    "dopamine": round(_clamp01(_safe_float(state.get("reward"), 0.0)), 4),
                    "novelty": round(_clamp01(_safe_float(state.get("novelty"), 0.0)), 4),
                    "boredom_relief": round(_clamp01(_safe_float(state.get("boredom_relief"), 0.0)), 4),
                    "valence": round(_clamp_signed(_safe_float(state.get("valence"), 0.0)), 4),
                    "satisfaction": round(_clamp01(_safe_float(state.get("satisfaction"), 0.0)), 4),
                    "reward_delta": round(reward_delta, 4),
                    "novelty_delta": round(novelty_delta, 4),
                    "salience_delta": round(salience_delta, 4),
                    "reason": reason,
                    "curves": curve_summary,
                    "curve_last": curve_last,
                    "ts": now,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta=meta,
            )
        ]
        if salience_delta != 0.0 or _safe_float(state.get("salience"), 0.0) > 0.0:
            events.append(
                Event(
                    topic="affect/salience",
                    payload={
                        "score": round(_clamp01(_safe_float(state.get("salience"), 0.0)), 4),
                        "salience": round(_clamp01(_safe_float(state.get("salience"), 0.0)), 4),
                        "source_topic": event.topic,
                        "reason": reason,
                        "curve_last": curve_last,
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta=meta,
                )
            )
        return events


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "act/speech",
            "control/reinforce",
            "control/trainer_correction",
        ],
        output_topics=["affect/reward", "affect/salience"],
        priority=4,
    )
    yield RewardNoveltyPulseNeuron(cfg)
