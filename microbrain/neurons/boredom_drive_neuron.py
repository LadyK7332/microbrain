from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Iterable, Dict, Any

from microbrain.hormone import derive_ddna_modulators
from microbrain.pdna.access import profile_path
from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

# Neuron name = this file's basename without .py
from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("affect")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _text_from_event(event: Event) -> str:
    payload = event.payload
    if isinstance(payload, dict):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return str(payload or "").strip()


def _is_control_or_internal(event: Event) -> bool:
    meta = dict(event.meta or {})
    payload = event.payload if isinstance(event.payload, dict) else {}
    channel = str(payload.get("channel", "") or meta.get("channel", "") or "")
    text = _text_from_event(event)
    return (
        bool(meta.get("control", False))
        or bool(meta.get("cognitive_visible") is False)
        or text.lstrip().startswith("/")
        or event.topic.startswith("ui/")
        or event.topic.startswith("control/")
        or channel in {"internal", "thought"}
    )


def _fingerprint(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))[:160]


class BoredomDriveNeuron(BaseNeuron):
    """
    Tracks a simple "boredom" drive based on repetition and low novelty.

    Heuristic v2:
      - Time without external stimulation slowly raises boredom.
      - New external percepts lower boredom.
      - Repeated HRM location raises boredom.
      - Repeating the same/near-same output without novelty raises boredom.
      - Different output or different user response creates a small novelty relief.
      - PDNA traits (energy, playfulness) modulate how quickly boredom rises.
      - Current state is exposed via KV as "drive:boredom".

    Design rule: novelty is not output itself; novelty is the delta between an
    attempt and the result. Same output + same result decays novelty.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic not in (
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "act/speech",
            "control/reinforce",
            "control/trainer_correction",
            "affect/reward",
        ):
            return []

        now = time.time()
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)
        pdna_profile = await ctx.get_kv("pdna:profile", None)
        ddna_mods = await ctx.get_kv("drive:ddna_modulators", None)
        if not isinstance(ddna_mods, dict) or not ddna_mods:
            ddna_mods = derive_ddna_modulators(pdna_profile)
            await ctx.set_kv("drive:ddna_modulators", ddna_mods)
        novelty_gain = max(0.20, min(2.0, float((ddna_mods or {}).get("novelty_gain", 1.0) or 1.0)))
        boredom_growth_gain = max(0.20, min(2.0, float((ddna_mods or {}).get("boredom_growth_gain", 1.0) or 1.0)))
        boredom_relief_gain = max(0.20, min(2.0, float((ddna_mods or {}).get("boredom_relief_gain", 1.0) or 1.0)))
        boredom_relief_state = await ctx.get_kv("drive:boredom_relief", {})
        base_idle_growth = float(profile_path(pdna_profile, "affect_model", "decay.boredom_growth_per_second_when_understimulated", 0.07) or 0.07)
        # The profile decay value is conservative; keep the old live UI twitch
        # if the profile is lower, but let DDNA mutate it.
        idle_growth_per_s = max(0.012, base_idle_growth) * boredom_growth_gain

        state = await self.load_state(
            ctx,
            "boredom_state",
            default={
                "prev_idx": None,
                "repetitions": 0,
                "level": 0.0,
                "last_tick_ts": now,
                "last_external_ts": now,
                "last_output_fp": "",
                "last_user_fp": "",
                "same_output_repetitions": 0,
                "same_user_repetitions": 0,
                "novelty_delta": 0.0,
            },
        )

        prev_idx = state.get("prev_idx", None)
        repetitions = int(state.get("repetitions", 0) or 0)
        level = float(state.get("level", 0.0) or 0.0)
        last_tick_ts = float(state.get("last_tick_ts", now) or now)
        last_external_ts = float(state.get("last_external_ts", now) or now)
        last_output_fp = str(state.get("last_output_fp", "") or "")
        last_user_fp = str(state.get("last_user_fp", "") or "")
        same_output_repetitions = int(state.get("same_output_repetitions", 0) or 0)
        same_user_repetitions = int(state.get("same_user_repetitions", 0) or 0)
        novelty_delta = 0.0

        dt = max(0.0, now - last_tick_ts)

        # External stimulation: lower boredom, but repeated identical input gives less relief.
        if event.topic in ("percept/text", "percept/vision") and not _is_control_or_internal(event):
            last_external_ts = now
            if event.topic == "percept/text":
                user_fp = _fingerprint(_text_from_event(event))
                if user_fp and user_fp == last_user_fp:
                    same_user_repetitions += 1
                    novelty_delta -= min(0.08, 0.02 * same_user_repetitions) * novelty_gain
                    level += min(0.04, 0.01 * same_user_repetitions) * boredom_growth_gain
                else:
                    same_user_repetitions = 0
                    if user_fp:
                        novelty_delta += 0.16 * novelty_gain
                    # Novel interaction should be a visible relief pulse, not a tiny nudge.
                    level = max(0.0, level - (0.18 * boredom_relief_gain))
                raw_meta = payload.get("raw_meta", {}) if isinstance(payload, dict) and isinstance(payload.get("raw_meta"), dict) else {}
                accent_positive = _clamp01(float(raw_meta.get("accent_positive", (event.meta or {}).get("accent_positive", 0.0)) or 0.0) / 10.0)
                accent_negative = _clamp01(float(raw_meta.get("accent_negative_severity", (event.meta or {}).get("accent_negative_severity", 0.0)) or 0.0) / 10.0)
                if accent_positive > 0.0:
                    novelty_delta += 0.12 * accent_positive * novelty_gain
                    level = max(0.0, level - ((0.10 + 0.22 * accent_positive) * boredom_relief_gain))
                elif accent_negative > 0.0:
                    # Correction is still activity/novelty, but it is not reward.
                    novelty_delta += 0.04 * accent_negative * novelty_gain
                    level = max(0.0, level - (0.04 * accent_negative * boredom_relief_gain))
                if user_fp:
                    last_user_fp = user_fp
            else:
                novelty_delta += 0.12 * novelty_gain
                level = max(0.0, level - (0.14 * boredom_relief_gain))

        # Self-output novelty: repeating the same output is stale unless it changes the result later.
        if event.topic == "act/speech" and not _is_control_or_internal(event):
            out_fp = _fingerprint(_text_from_event(event))
            if out_fp:
                if out_fp == last_output_fp:
                    same_output_repetitions += 1
                    # Same output should not satisfy novelty; it becomes stale pressure.
                    penalty = min(0.16, 0.035 * same_output_repetitions)
                    novelty_delta -= penalty * novelty_gain
                    level += penalty * boredom_growth_gain
                else:
                    same_output_repetitions = 0
                    novelty_delta += 0.08 * novelty_gain
                    level = max(0.0, level - (0.06 * boredom_relief_gain))
                last_output_fp = out_fp

        # Explicit reward/training events should act like successful activity and relieve boredom.
        if event.topic == "control/reinforce":
            payload = event.payload if isinstance(event.payload, dict) else {}
            weight = float(payload.get("weight", 0.0) or 0.0)
            if weight > 0.0:
                strength = _clamp01(weight / 5.0)
                novelty_delta += (0.10 + 0.10 * strength) * novelty_gain
                level = max(0.0, level - ((0.16 + 0.22 * strength) * boredom_relief_gain))
            elif weight < 0.0:
                novelty_delta += 0.04 * novelty_gain

        if event.topic == "control/trainer_correction":
            novelty_delta += 0.18 * novelty_gain
            level = max(0.0, level - (0.24 * boredom_relief_gain))

        if event.topic == "affect/reward":
            payload = event.payload if isinstance(event.payload, dict) else {}
            reward_delta = float(payload.get("reward_delta", 0.0) or 0.0)
            relief = float(payload.get("boredom_relief", 0.0) or 0.0)
            if reward_delta > 0.0 or relief > 0.0:
                strength = _clamp01(max(reward_delta, relief))
                novelty_delta += 0.08 * strength * novelty_gain
                level = max(0.0, level - (0.16 * strength * boredom_relief_gain))

        if isinstance(boredom_relief_state, dict):
            relief_level = _clamp01(float(boredom_relief_state.get("level", 0.0) or 0.0))
            if relief_level > 0.0 and event.topic == SERVICE_TOPIC:
                novelty_delta += 0.025 * relief_level * novelty_gain
                level = max(0.0, level - (0.035 * relief_level * boredom_relief_gain))

        idle = (now - last_external_ts) > 2.0
        if event.topic == SERVICE_TOPIC:
            if idle:
                level += dt * idle_growth_per_s
            else:
                level -= dt * 0.05 * boredom_relief_gain

        level = _clamp01(level)

        # HRM repetition boredom: only on affect-service opportunities so it reflects "stuck over time".
        if prev_idx is None and hrm_last_idx is not None:
            prev_idx = hrm_last_idx
        elif event.topic == SERVICE_TOPIC and hrm_last_idx is not None:
            if hrm_last_idx == prev_idx:
                repetitions += 1
                base_inc = 0.02
                energy = getattr(pdna_profile, "energy", 0.5) if pdna_profile is not None else 0.5
                playfulness = getattr(pdna_profile, "playfulness", 0.5) if pdna_profile is not None else 0.5
                pdna_factor = 0.5 + energy * 0.3 + playfulness * 0.3
                repetition_factor = 1.0 + min(repetitions, 20) / 20.0
                level += base_inc * pdna_factor * repetition_factor * boredom_growth_gain
                novelty_delta -= min(0.06, 0.004 * repetitions) * novelty_gain
            else:
                repetitions = max(0, repetitions - 1)
                prev_idx = hrm_last_idx
                level *= max(0.0, 0.8 - (0.05 * min(1.0, boredom_relief_gain - 1.0)))
                novelty_delta += 0.10 * novelty_gain

        level = _clamp01(level)
        novelty_delta = max(-1.0, min(1.0, novelty_delta))

        state.update(
            {
                "prev_idx": prev_idx,
                "repetitions": repetitions,
                "level": level,
                "last_tick_ts": now,
                "last_external_ts": last_external_ts,
                "last_output_fp": last_output_fp,
                "last_user_fp": last_user_fp,
                "same_output_repetitions": same_output_repetitions,
                "same_user_repetitions": same_user_repetitions,
                "novelty_delta": novelty_delta,
            }
        )
        await self.save_state(ctx, "boredom_state", state)

        boredom_payload = {
            "active": level >= 0.6,
            "high": level >= 0.85,
            "level": round(level, 4),
            "repetitions": repetitions,
            "same_output_repetitions": same_output_repetitions,
            "same_user_repetitions": same_user_repetitions,
            "novelty_delta": round(novelty_delta, 4),
            "stale_output": same_output_repetitions > 0,
        }
        await ctx.set_kv("drive:boredom", boredom_payload)

        # Social experimentation is a derived drive: boredom + social need.
        social = await ctx.get_kv("drive:social_interaction", {})
        social_level = 0.0
        if isinstance(social, dict):
            try:
                social_level = float(social.get("level", 0.0) or 0.0)
            except Exception:
                social_level = 0.0
        social_experiment = {
            "active": (level >= 0.55 and social_level >= 0.45),
            "high": (level >= 0.75 and social_level >= 0.65),
            "boredom": round(level, 4),
            "social": round(max(0.0, min(1.0, social_level)), 4),
            "pressure": round(max(0.0, min(1.0, (level + social_level) / 2.0)), 4),
        }
        await ctx.set_kv("drive:social_experimentation", social_experiment)

        self.debug("boredom_updated", **boredom_payload, social_experiment=social_experiment)
        return []


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
            "affect/reward",
        ],
        output_topics=[],
        priority=-10,
    )
    yield BoredomDriveNeuron(cfg)
