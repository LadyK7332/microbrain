from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


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
    style = str(payload.get("style", "") or "")
    return (
        bool(meta.get("control", False))
        or bool(meta.get("cognitive_visible") is False)
        or event.topic.startswith("ui/")
        or event.topic.startswith("control/")
        or channel in {"internal", "thought"}
        or style == "system"
    )


class SocialInteractionDriveNeuron(BaseNeuron):
    """
    Tracks a base social/interaction drive.

    This is distinct from boredom:
      - boredom = low novelty / stale attempts
      - social interaction = pressure toward response/contact/coupling

    The drive rises during social silence and when MB has made an outward
    attempt that has not received a response. It drops when user/external text
    arrives. It does not speak by itself; other organs consume the KV state.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        if event.topic not in ("clock/tick", "percept/text", "act/speech", "control/reinforce"):
            return []

        now = time.time()
        state = await self.load_state(
            ctx,
            "social_interaction_state",
            default={
                "level": 0.0,
                "last_tick_ts": now,
                "last_user_ts": now,
                "last_mb_output_ts": 0.0,
                "awaiting_response": False,
                "unanswered_attempts": 0,
                "last_result": "init",
            },
        )

        level = float(state.get("level", 0.0) or 0.0)
        last_tick_ts = float(state.get("last_tick_ts", now) or now)
        last_user_ts = float(state.get("last_user_ts", now) or now)
        last_mb_output_ts = float(state.get("last_mb_output_ts", 0.0) or 0.0)
        awaiting_response = bool(state.get("awaiting_response", False))
        unanswered_attempts = int(state.get("unanswered_attempts", 0) or 0)
        last_result = str(state.get("last_result", "") or "")

        dt = max(0.0, now - last_tick_ts)
        result = "idle"

        if event.topic == "percept/text" and not _is_control_or_internal(event):
            text = _text_from_event(event)
            if text:
                last_user_ts = now
                # External response/contact satisfies social pressure.
                level = max(0.0, level - 0.28)
                if awaiting_response:
                    result = "response_received"
                    unanswered_attempts = 0
                else:
                    result = "external_contact"
                awaiting_response = False
                await ctx.set_kv(
                    "interaction:last_user_response",
                    {"ts": now, "text": text[:160], "after_mb_output_ts": last_mb_output_ts},
                )

        elif event.topic == "act/speech" and not _is_control_or_internal(event):
            text = _text_from_event(event)
            if text:
                last_mb_output_ts = now
                awaiting_response = True
                result = "awaiting_user_response"
                # Speaking relieves a little pressure, but if ignored the clock will raise it again.
                level = max(0.0, level - 0.06)

        elif event.topic == "control/reinforce":
            # User reinforcement is social feedback. It should calm interaction pressure.
            level = max(0.0, level - 0.20)
            awaiting_response = False
            unanswered_attempts = 0
            result = "reinforced_feedback"

        if event.topic == "clock/tick":
            silence_s = max(0.0, now - last_user_ts)
            if silence_s > 30.0:
                # Slow rise during social silence.
                level += dt * 0.004
            if awaiting_response and last_mb_output_ts > 0.0 and (now - last_mb_output_ts) > 15.0:
                # MB tried to interact and did not get a response: social need rises faster.
                level += dt * 0.012
                if last_result != "unanswered_attempt":
                    unanswered_attempts += 1
                result = "unanswered_attempt"

        boredom = await ctx.get_kv("drive:boredom", {})
        boredom_level = 0.0
        if isinstance(boredom, dict):
            try:
                boredom_level = float(boredom.get("level", 0.0) or 0.0)
            except Exception:
                boredom_level = 0.0

        power_state = await ctx.get_kv("power:state", {})
        sleeping = False
        if isinstance(power_state, dict):
            sleeping = bool(power_state.get("sleep", False))
        sleeping = sleeping or bool(await ctx.get_kv("power:sleep", False))

        level = _clamp01(level)
        if sleeping:
            # Sleep/maintenance posture suppresses social expression while preserving a low trace.
            level = min(level, 0.25)
            awaiting_response = False

        social_experiment_pressure = _clamp01((0.55 * level) + (0.45 * boredom_level))
        payload: Dict[str, Any] = {
            "active": level >= 0.45,
            "high": level >= 0.75,
            "level": round(level, 4),
            "awaiting_response": awaiting_response,
            "unanswered_attempts": unanswered_attempts,
            "last_user_age_s": round(max(0.0, now - last_user_ts), 2),
            "last_mb_output_age_s": round(max(0.0, now - last_mb_output_ts), 2) if last_mb_output_ts else None,
            "last_result": result if result != "idle" else last_result,
            "social_experiment_pressure": round(social_experiment_pressure, 4),
        }
        await ctx.set_kv("drive:social_interaction", payload)

        # Also publish a combined social experimentation hint for modules that do not read boredom separately.
        await ctx.set_kv(
            "drive:social_experimentation",
            {
                "active": (level >= 0.45 and boredom_level >= 0.55),
                "high": (level >= 0.65 and boredom_level >= 0.75),
                "social": round(level, 4),
                "boredom": round(boredom_level, 4),
                "pressure": round(social_experiment_pressure, 4),
                "awaiting_response": awaiting_response,
            },
        )

        state.update(
            {
                "level": level,
                "last_tick_ts": now,
                "last_user_ts": last_user_ts,
                "last_mb_output_ts": last_mb_output_ts,
                "awaiting_response": awaiting_response,
                "unanswered_attempts": unanswered_attempts,
                "last_result": result if result != "idle" else last_result,
            }
        )
        await self.save_state(ctx, "social_interaction_state", state)
        self.debug("social_interaction_updated", **payload)
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "percept/text", "act/speech", "control/reinforce"],
        output_topics=[],
        priority=-9,
    )
    yield SocialInteractionDriveNeuron(cfg)
