from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
_GREETING_STARTS = (
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "guten tag",
    "gutenberg",
    "moin",
)
_DIRECT_ADDRESS_MARKERS = (
    " demi",
    " demi ",
    " mb",
    " mb ",
    "mi-",
    " mi ",
)


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


class InteractionReleaseVectorNeuron(BaseNeuron):
    """
    Interaction pressure sibling to power pressure.

    Purpose:
      unresolved interaction -> pressure rises -> choose outlet -> express ->
      vent partially through outward release.

    It leans on the initiative state so it can mirror MB's existing sense of
    pending user input, continuity pressure, and clarification readiness instead
    of inventing a second disconnected social model.
    """

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _fingerprint(self, text: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", self._clean_text(text).lower()).strip()[:160]

    def _external_text_from_event(self, event: Event) -> tuple[str, str, bool]:
        """Return (text, channel, eligible) for user-facing text stimuli.

        Interaction pressure is a reflex arc: external input should create
        short-lived response pressure. Control-plane/status/thought traffic must
        not do that, or commands and diagnostics start acting like social needs.
        """
        if event.topic != "percept/text":
            return "", "default", False
        payload = event.payload if isinstance(event.payload, dict) else {}
        meta = event.meta or {}
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        text = self._clean_text(str(payload.get("text", "") or ""))
        channel = str(raw_meta.get("channel", payload.get("channel", "default")) or "default")
        source = str(raw_meta.get("source", payload.get("source", "user")) or "user").lower()
        if not text:
            return "", channel, False
        if source in {"assistant", "system", "mb"}:
            return text, channel, False
        if bool(meta.get("control", False)) or bool(raw_meta.get("control", False)):
            return text, channel, False
        if meta.get("cognitive_visible") is False or raw_meta.get("cognitive_visible") is False:
            return text, channel, False
        if text.lstrip().startswith("/"):
            return text, channel, False
        return text, channel, True

    async def _record_input_stimulus(self, ctx, event: Event, now: float) -> None:
        text, channel, eligible = self._external_text_from_event(event)
        if not eligible:
            return
        fp = self._fingerprint(text)
        await ctx.set_kv(
            "drive:interaction:last_input_stimulus",
            {
                "text": text[:280],
                "fingerprint": fp,
                "channel": channel,
                "ts": now,
                "correlation_id": event.correlation_id,
                "source": self.name,
            },
        )

    def _looks_like_greeting(self, text: str) -> bool:
        lowered = self._clean_text(text).lower()
        return any(lowered.startswith(item) for item in _GREETING_STARTS)

    def _direct_address(self, text: str) -> bool:
        lowered = f" {self._clean_text(text).lower()} "
        return any(marker in lowered for marker in _DIRECT_ADDRESS_MARKERS)

    async def _channel_availability(self, ctx) -> Dict[str, bool]:
        textual = bool(await ctx.get_kv("outlet:textual_available", True))
        audio_pref = str(await ctx.get_kv("speech:audio_preferred_transport", "none") or "none").strip().lower()
        audio = bool(await ctx.get_kv("outlet:audio_available", audio_pref != "none"))
        motion = bool(await ctx.get_kv("outlet:motion_available", False))
        return {"textual": textual, "audio": audio, "motion": motion}

    async def _compute_pressure(self, ctx, now: float) -> Dict[str, Any]:
        initiative = await ctx.get_kv("initiative:last", {}) or {}
        state = await ctx.get_kv("neuron:InitiativeThresholdNeuron:initiative_threshold_neuron:initiative_state", {}) or {}
        if not isinstance(initiative, dict):
            initiative = {}
        if not isinstance(state, dict):
            state = {}

        pending_text = self._clean_text(str(initiative.get("pending_text", state.get("pending_text", "")) or ""))
        pending_flags = state.get("pending_flags", {}) if isinstance(state.get("pending_flags", {}), dict) else {}
        talk_pressure = _clamp01(initiative.get("talk_pressure", 0.0))
        think_pressure = _clamp01(initiative.get("think_pressure", 0.0))
        pending_age = max(0.0, _safe_float(initiative.get("pending_age_s", 0.0), 0.0))

        # InitiativeThresholdNeuron runs later than this neuron on the same
        # percept/text event, so a brand-new user input could otherwise be
        # invisible until the next clock tick. Use a short-lived input stimulus
        # as the immediate reflex source for interaction pressure.
        stimulus = await ctx.get_kv("drive:interaction:last_input_stimulus", {}) or {}
        if isinstance(stimulus, dict):
            stim_text = self._clean_text(str(stimulus.get("text", "") or ""))
            stim_ts = _safe_float(stimulus.get("ts", 0.0), 0.0)
            stim_age = max(0.0, now - stim_ts) if stim_ts > 0.0 else 9999.0
            stimulus_window = _safe_float(await ctx.get_kv("drive:interaction:input_stimulus_window_s", 18.0), 18.0)
            if stim_text and stim_age <= max(1.0, stimulus_window) and not pending_text:
                pending_text = stim_text
                pending_age = stim_age
                pending_flags = {
                    "has_question": "?" in stim_text,
                    "has_response_request": any(tok in stim_text.lower() for tok in ("respond", "reply", "speak up", "can you hear me")),
                    "has_error_language": any(tok in stim_text.lower() for tok in ("error", "issue", "problem", "stuck", "not working")),
                    "short_fragment": len(stim_text.split()) <= 3,
                    "clarify_ready": False,
                    "coherence_score": 0.20 if "?" in stim_text else 0.08,
                    "from_input_stimulus": True,
                }
                talk_pressure = max(talk_pressure, _safe_float(await ctx.get_kv("drive:interaction:input_talk_floor", 0.58), 0.58))
                think_pressure = max(think_pressure, _safe_float(await ctx.get_kv("drive:interaction:input_think_floor", 0.34), 0.34))
        clarify_ready = bool(initiative.get("clarify_ready", pending_flags.get("clarify_ready", False)))
        interruption_cost = _clamp01(initiative.get("interruption_cost", 0.0))
        answered = _safe_bool(state.get("pending_answered", False), False)
        clarify_said = _safe_bool(state.get("clarify_said", False), False)

        question = bool(pending_flags.get("has_question", False))
        response_request = bool(pending_flags.get("has_response_request", False))
        error_language = bool(pending_flags.get("has_error_language", False))
        short_fragment = bool(pending_flags.get("short_fragment", False))
        greeting = self._looks_like_greeting(pending_text)
        direct_address = self._direct_address(pending_text)
        social_bid = greeting or question or response_request or direct_address

        base = 0.0
        if pending_text:
            base += 0.14
            if bool(pending_flags.get("from_input_stimulus", False)):
                base += _safe_float(await ctx.get_kv("drive:interaction:input_base_boost", 0.34), 0.34)
        if question:
            base += 0.30
        if response_request:
            base += 0.28
        if greeting:
            base += 0.18
        if direct_address:
            base += 0.14
        if short_fragment:
            base += 0.08
        if error_language:
            base += 0.06
        if clarify_ready and pending_text:
            base += 0.10

        if answered:
            base *= 0.38
        if clarify_said:
            base *= 0.72

        persistence_window = _safe_float(await ctx.get_kv("drive:interaction:persistence_window_s", 120.0), 120.0)
        persistence = _clamp01(pending_age / max(1.0, persistence_window)) if pending_text else 0.0

        urgency = _clamp01(
            (base * 0.45)
            + (talk_pressure * 0.28)
            + (think_pressure * 0.10)
            + (persistence * 0.17)
            + (0.08 if social_bid else 0.0)
            + (0.08 if question or response_request else 0.0)
            - (0.18 * interruption_cost)
        )

        return {
            "need": "interaction",
            "pending_text": pending_text,
            "pending_fingerprint": self._fingerprint(pending_text),
            "stimulus_input": bool(pending_flags.get("from_input_stimulus", False)),
            "pending_age_s": round(pending_age, 3),
            "talk_pressure": round(talk_pressure, 4),
            "think_pressure": round(think_pressure, 4),
            "base_pressure": round(_clamp01(base), 4),
            "persistence": round(persistence, 4),
            "urgency": round(urgency, 4),
            "clarify_ready": clarify_ready,
            "question": question,
            "response_request": response_request,
            "greeting": greeting,
            "direct_address": direct_address,
            "social_bid": social_bid,
            "answered": answered,
            "clarify_said": clarify_said,
            "ts": now,
        }

    async def _choose_vector(self, ctx, pressure: Dict[str, Any], channels: Dict[str, bool]) -> Dict[str, Any]:
        stats = await ctx.get_kv("route:interaction_relief_stats", {})
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
            latency_score = 0.0 if avg_latency >= 999.0 else _clamp01(1.0 - (avg_latency / 20.0))
            base_bias = {
                "textual": _safe_float(await ctx.get_kv("drive:interaction:textual_bias", 0.28), 0.28),
                "audio": _safe_float(await ctx.get_kv("drive:interaction:audio_bias", 0.10), 0.10),
                "motion": _safe_float(await ctx.get_kv("drive:interaction:motion_bias", 0.03), 0.03),
            }.get(outlet, 0.0)
            score = base_bias + (success_rate * 0.40) + (min(avg_relief, 1.0) * 0.22) + (latency_score * 0.16)
            options.append({
                "outlet": outlet,
                "score": round(score, 4),
                "success_rate": round(success_rate, 4),
                "avg_relief": round(avg_relief, 4),
                "avg_latency_s": round(avg_latency, 4) if avg_latency < 999.0 else None,
            })

        if not options:
            return {
                "need": "interaction",
                "outlet": None,
                "style": None,
                "message": None,
                "score": 0.0,
                "channels": channels,
                "options": options,
            }

        options.sort(key=lambda item: (item["score"], item["outlet"] == "textual"), reverse=True)
        chosen = dict(options[0])

        pending_text = str(pressure.get("pending_text", "") or "")
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        question = bool(pressure.get("question", False))
        response_request = bool(pressure.get("response_request", False))
        greeting = bool(pressure.get("greeting", False))
        clarify_ready = bool(pressure.get("clarify_ready", False))

        stimulus_input = bool(pressure.get("stimulus_input", False))
        if greeting and urgency < 0.65:
            style = "direct_simple"
            message = "greeting_pressure"
        elif question or response_request:
            if urgency >= 0.80:
                style = "urgent_direct"
                message = "question_pressure_urgent"
            else:
                style = "direct_simple"
                message = "question_pressure"
        elif stimulus_input:
            style = "direct_simple" if urgency >= 0.50 else "gentle_notice"
            message = "stimulus_response_pressure"
        elif clarify_ready:
            style = "gentle_notice" if urgency < 0.60 else "direct_simple"
            message = f"missing_variable:{pending_text}" if pending_text else "missing_variable"
        else:
            style = "gentle_notice"
            message = "interaction_pressure_open"

        chosen.update({
            "need": "interaction",
            "style": style,
            "message": message,
            "channels": channels,
            "options": options,
        })
        return chosen

    async def _emit_request(self, ctx, event: Event, pressure: Dict[str, Any], vector: Dict[str, Any], now: float) -> list[Event]:
        outlet = vector.get("outlet")
        style = str(vector.get("style", "direct_simple") or "direct_simple")
        message = str(vector.get("message", "interaction_pressure_open") or "interaction_pressure_open")

        pending = {
            "need": "interaction",
            "ts": now,
            "pressure": pressure,
            "vector": vector,
            "outlet": outlet,
            "style": style,
            "message": message,
            "pending_text": str(pressure.get("pending_text", "") or ""),
            "correlation_id": event.correlation_id,
        }
        await ctx.set_kv("drive:interaction_pending_request", pending)
        await ctx.set_kv("drive:interaction:last_signal_ts", now)
        await ctx.set_kv("drive:interaction:last_signal_style", style)
        await ctx.set_kv("drive:interaction:last_signal_fingerprint", str(pressure.get("pending_fingerprint", "") or ""))

        thought_text = "input_response_pressure"
        return [
            Event(
                topic="thought/internal",
                payload={
                    "text": thought_text,
                    "kind": "interaction_pressure",
                    "source_need": "interaction",
                    "urgency": pressure.get("urgency", 0.0),
                    "pending_text": str(pressure.get("pending_text", "") or "")[:160],
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "channel": "thought",
                    "kind": "interaction_pressure",
                    "need": "interaction",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                    "cognitive_visible": False,
                },
            ),
            Event(
                topic="drive/interaction_request",
                payload=pending,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"need": "interaction", "outlet": outlet, "style": style},
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
                    "need": "interaction",
                    "outlet": outlet,
                    "style": style,
                    "kind": "need_signal_reason",
                },
            ),
        ]

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.time()
        if event.topic == "percept/text":
            await self._record_input_stimulus(ctx, event, now)
        elif event.topic == "event/relief/interaction":
            await ctx.set_kv("drive:interaction:last_input_stimulus", {})

        power_state = await ctx.get_kv("power:state", {}) or {}
        sleeping = bool((power_state or {}).get("sleep", False))
        charging = bool((power_state or {}).get("charging", False))

        pressure = await self._compute_pressure(ctx, now)
        channels = await self._channel_availability(ctx)
        vector = await self._choose_vector(ctx, pressure, channels)

        snapshot = {
            "need": "interaction",
            "pressure": pressure,
            "vector": vector,
            "channels": channels,
            "ts": now,
        }
        await ctx.set_kv("drive:interaction_vector", snapshot)

        if sleeping:
            return []
        if event.topic == "event/relief/interaction":
            return []
        if vector.get("outlet") is None:
            return []
        if not str(pressure.get("pending_text", "") or "").strip():
            return []
        if bool(pressure.get("answered", False)):
            return []

        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        threshold = _safe_float(await ctx.get_kv("drive:interaction:signal_threshold", 0.50), 0.50)
        if bool(pressure.get("stimulus_input", False)):
            threshold = min(threshold, _safe_float(await ctx.get_kv("drive:interaction:input_signal_threshold", 0.36), 0.36))
        if urgency < threshold:
            return []

        this_fp = str(pressure.get("pending_fingerprint", "") or "")
        last_signal_fp = str(await ctx.get_kv("drive:interaction:last_signal_fingerprint", "") or "")
        same_signal = bool(this_fp and last_signal_fp and this_fp == last_signal_fp)
        last_signal_ts = _safe_float(await ctx.get_kv("drive:interaction:last_signal_ts", 0.0), 0.0)
        cooldown_s = _safe_float(await ctx.get_kv("drive:interaction:signal_cooldown_s", 45.0), 45.0)
        if same_signal and last_signal_ts > 0.0 and (now - last_signal_ts) < cooldown_s:
            return []

        last_relief_ts = _safe_float(await ctx.get_kv("drive:interaction:last_relief_ts", 0.0), 0.0)
        quiet_after_relief_s = _safe_float(await ctx.get_kv("drive:interaction:quiet_after_relief_s", 12.0), 12.0)
        if last_relief_ts > 0.0 and (now - last_relief_ts) < quiet_after_relief_s:
            return []

        # Keep it quieter while power is being actively handled.
        if charging and urgency < 0.78:
            return []

        return await self._emit_request(ctx, event, pressure, vector, now)


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "percept/text", "act/speech", "event/relief/interaction"],
        output_topics=["drive/interaction_request", "speech/reason"],
        priority=8,
        cooldown_sec=0.0,
    )
    yield InteractionReleaseVectorNeuron(cfg)
