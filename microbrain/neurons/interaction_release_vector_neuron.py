from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# How long a newly observed participant message can act as the immediate
# interaction-pressure stimulus before InitiativeThresholdNeuron catches up.
# Unit: seconds. Practical range: 2.0-60.0.
INPUT_STIMULUS_WINDOW_S = 18.0

# Minimum immediate pressure supplied by a fresh participant message.
# Range: 0.0-1.0. Higher values make interaction pressure rise faster.
INPUT_TALK_FLOOR = 0.58
INPUT_THINK_FLOOR = 0.34
INPUT_BASE_BOOST = 0.34

# Base pressure contributions. Range: 0.0-1.0 each.
PENDING_TEXT_BASE = 0.14
QUESTION_BASE = 0.30
RESPONSE_REQUEST_BASE = 0.28
GREETING_BASE = 0.18
DIRECT_ADDRESS_BASE = 0.14
SHORT_FRAGMENT_BASE = 0.08
QUESTION_COHERENCE_SCORE = 0.20
STATEMENT_COHERENCE_SCORE = 0.08
ERROR_LANGUAGE_BASE = 0.06
CLARIFY_READY_BASE = 0.10

# Pressure attenuation after an answer or clarification has already occurred.
# Range: 0.0-1.0. Lower values vent more pressure.
ANSWERED_PRESSURE_SCALE = 0.38
CLARIFY_SAID_PRESSURE_SCALE = 0.72

# Time for unresolved interaction pressure to reach full persistence.
# Unit: seconds. Practical range: 15.0-600.0.
PERSISTENCE_WINDOW_S = 120.0

# Urgency fusion weights. These are intentionally visible because they define
# how strongly each bodily signal contributes to interaction pressure.
URGENCY_BASE_WEIGHT = 0.45
URGENCY_TALK_WEIGHT = 0.28
URGENCY_THINK_WEIGHT = 0.10
URGENCY_PERSISTENCE_WEIGHT = 0.17
URGENCY_SOCIAL_BID_BONUS = 0.08
URGENCY_EXPLICIT_RESPONSE_BONUS = 0.08
URGENCY_INTERRUPTION_PENALTY = 0.18

# Outlet scoring defaults. These may still be overridden through KV controls.
TEXTUAL_OUTLET_BIAS = 0.28
AUDIO_OUTLET_BIAS = 0.10
MOTION_OUTLET_BIAS = 0.03
OUTLET_SUCCESS_WEIGHT = 0.40
OUTLET_RELIEF_WEIGHT = 0.22
OUTLET_LATENCY_WEIGHT = 0.16
OUTLET_LATENCY_TARGET_S = 20.0

# Style thresholds. Range: 0.0-1.0.
GREETING_DIRECT_MAX_URGENCY = 0.65
URGENT_QUESTION_MIN_URGENCY = 0.80
STIMULUS_DIRECT_MIN_URGENCY = 0.50
CLARIFY_DIRECT_MIN_URGENCY = 0.60

# Emission gates. These may still be overridden through KV controls.
SIGNAL_THRESHOLD = 0.50
INPUT_SIGNAL_THRESHOLD = 0.36
SIGNAL_COOLDOWN_S = 45.0
QUIET_AFTER_RELIEF_S = 12.0
CHARGING_RELEASE_MIN_URGENCY = 0.78

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

# Fixed bus routes and identity markers. Changing these requires updating every
# producer/subscriber that participates in the interaction-pressure protocol.
NEURON_NAME = Path(__file__).stem

# Unified response-ownership law. External participant turns are interpreted
# and released by the hypothesis path. This neuron may measure and publish
# pressure, but must not create a second outward reply for the same turn.
# Do not change without redesigning the response arbitration protocol.
HYPOTHESIS_OWNS_EXTERNAL_INTERACTION = True

THOUGHT_TOPIC = "thought/internal"
INTERACTION_REQUEST_TOPIC = "drive/interaction_request"
SPEECH_REASON_TOPIC = "speech/reason"
EXTERNAL_RESPONSE_OWNER = "hypothesis"
INTERNAL_RESPONSE_OWNER = "interaction_release_vector"

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

_NON_PARTICIPANT_SOURCES = {"assistant", "system", "mb"}
_RESPONSE_REQUEST_MARKERS = ("respond", "reply", "speak up", "can you hear me")
_ERROR_LANGUAGE_MARKERS = ("error", "issue", "problem", "stuck", "not working")


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
        if source in _NON_PARTICIPANT_SOURCES:
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
            stimulus_window = _safe_float(
                await ctx.get_kv("drive:interaction:input_stimulus_window_s", INPUT_STIMULUS_WINDOW_S),
                INPUT_STIMULUS_WINDOW_S,
            )
            if stim_text and stim_age <= max(1.0, stimulus_window) and not pending_text:
                pending_text = stim_text
                pending_age = stim_age
                pending_flags = {
                    "has_question": "?" in stim_text,
                    "has_response_request": any(tok in stim_text.lower() for tok in _RESPONSE_REQUEST_MARKERS),
                    "has_error_language": any(tok in stim_text.lower() for tok in _ERROR_LANGUAGE_MARKERS),
                    "short_fragment": len(stim_text.split()) <= 3,
                    "clarify_ready": False,
                    "coherence_score": QUESTION_COHERENCE_SCORE if "?" in stim_text else STATEMENT_COHERENCE_SCORE,
                    "from_input_stimulus": True,
                }
                talk_pressure = max(
                    talk_pressure,
                    _safe_float(
                        await ctx.get_kv("drive:interaction:input_talk_floor", INPUT_TALK_FLOOR),
                        INPUT_TALK_FLOOR,
                    ),
                )
                think_pressure = max(
                    think_pressure,
                    _safe_float(
                        await ctx.get_kv("drive:interaction:input_think_floor", INPUT_THINK_FLOOR),
                        INPUT_THINK_FLOOR,
                    ),
                )
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
            base += PENDING_TEXT_BASE
            if bool(pending_flags.get("from_input_stimulus", False)):
                base += _safe_float(
                    await ctx.get_kv("drive:interaction:input_base_boost", INPUT_BASE_BOOST),
                    INPUT_BASE_BOOST,
                )
        if question:
            base += QUESTION_BASE
        if response_request:
            base += RESPONSE_REQUEST_BASE
        if greeting:
            base += GREETING_BASE
        if direct_address:
            base += DIRECT_ADDRESS_BASE
        if short_fragment:
            base += SHORT_FRAGMENT_BASE
        if error_language:
            base += ERROR_LANGUAGE_BASE
        if clarify_ready and pending_text:
            base += CLARIFY_READY_BASE

        if answered:
            base *= ANSWERED_PRESSURE_SCALE
        if clarify_said:
            base *= CLARIFY_SAID_PRESSURE_SCALE

        persistence_window = _safe_float(
            await ctx.get_kv("drive:interaction:persistence_window_s", PERSISTENCE_WINDOW_S),
            PERSISTENCE_WINDOW_S,
        )
        persistence = _clamp01(pending_age / max(1.0, persistence_window)) if pending_text else 0.0

        urgency = _clamp01(
            (base * URGENCY_BASE_WEIGHT)
            + (talk_pressure * URGENCY_TALK_WEIGHT)
            + (think_pressure * URGENCY_THINK_WEIGHT)
            + (persistence * URGENCY_PERSISTENCE_WEIGHT)
            + (URGENCY_SOCIAL_BID_BONUS if social_bid else 0.0)
            + (URGENCY_EXPLICIT_RESPONSE_BONUS if question or response_request else 0.0)
            - (URGENCY_INTERRUPTION_PENALTY * interruption_cost)
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
            latency_score = (
                0.0
                if avg_latency >= 999.0
                else _clamp01(1.0 - (avg_latency / OUTLET_LATENCY_TARGET_S))
            )
            base_bias = {
                "textual": _safe_float(
                    await ctx.get_kv("drive:interaction:textual_bias", TEXTUAL_OUTLET_BIAS),
                    TEXTUAL_OUTLET_BIAS,
                ),
                "audio": _safe_float(
                    await ctx.get_kv("drive:interaction:audio_bias", AUDIO_OUTLET_BIAS),
                    AUDIO_OUTLET_BIAS,
                ),
                "motion": _safe_float(
                    await ctx.get_kv("drive:interaction:motion_bias", MOTION_OUTLET_BIAS),
                    MOTION_OUTLET_BIAS,
                ),
            }.get(outlet, 0.0)
            score = (
                base_bias
                + (success_rate * OUTLET_SUCCESS_WEIGHT)
                + (min(avg_relief, 1.0) * OUTLET_RELIEF_WEIGHT)
                + (latency_score * OUTLET_LATENCY_WEIGHT)
            )
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
        if greeting and urgency < GREETING_DIRECT_MAX_URGENCY:
            style = "direct_simple"
            message = "greeting_pressure"
        elif question or response_request:
            if urgency >= URGENT_QUESTION_MIN_URGENCY:
                style = "urgent_direct"
                message = "question_pressure_urgent"
            else:
                style = "direct_simple"
                message = "question_pressure"
        elif stimulus_input:
            style = "direct_simple" if urgency >= STIMULUS_DIRECT_MIN_URGENCY else "gentle_notice"
            message = "stimulus_response_pressure"
        elif clarify_ready:
            style = "gentle_notice" if urgency < CLARIFY_DIRECT_MIN_URGENCY else "direct_simple"
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

    async def _response_ownership(self, ctx, pressure: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve who owns outward response selection for this pressure event.

        The interaction drive may observe and publish pressure from participant
        input, but the hypothesis path owns reply-or-silence selection for that
        same external turn.  Internal interaction pressure remains eligible for
        the legacy need-speech route.
        """
        pending_fp = str(pressure.get("pending_fingerprint", "") or "")
        stimulus = await ctx.get_kv("drive:interaction:last_input_stimulus", {}) or {}
        stimulus = stimulus if isinstance(stimulus, dict) else {}
        stimulus_fp = str(stimulus.get("fingerprint", "") or "")
        stimulus_corr = str(stimulus.get("correlation_id", "") or "")
        external_match = bool(pending_fp and stimulus_fp and pending_fp == stimulus_fp)
        hypothesis_owned = bool(HYPOTHESIS_OWNS_EXTERNAL_INTERACTION and external_match)
        ownership = {
            "owner": EXTERNAL_RESPONSE_OWNER if hypothesis_owned else INTERNAL_RESPONSE_OWNER,
            "hypothesis_owned": hypothesis_owned,
            "outward_speech_allowed": not hypothesis_owned,
            "pending_fingerprint": pending_fp,
            "stimulus_fingerprint": stimulus_fp,
            "stimulus_correlation_id": stimulus_corr,
        }
        await ctx.set_kv("drive:interaction:last_response_ownership", ownership)
        return ownership

    async def _emit_request(self, ctx, event: Event, pressure: Dict[str, Any], vector: Dict[str, Any], now: float) -> list[Event]:
        outlet = vector.get("outlet")
        style = str(vector.get("style", "direct_simple") or "direct_simple")
        message = str(vector.get("message", "interaction_pressure_open") or "interaction_pressure_open")
        ownership = await self._response_ownership(ctx, pressure)

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
            "response_owner": ownership["owner"],
            "outward_speech_allowed": ownership["outward_speech_allowed"],
        }
        await ctx.set_kv("drive:interaction_pending_request", pending)
        await ctx.set_kv("drive:interaction:last_signal_ts", now)
        await ctx.set_kv("drive:interaction:last_signal_style", style)
        await ctx.set_kv("drive:interaction:last_signal_fingerprint", str(pressure.get("pending_fingerprint", "") or ""))

        thought_text = "input_response_pressure"
        events = [
            Event(
                topic=THOUGHT_TOPIC,
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
                topic=INTERACTION_REQUEST_TOPIC,
                payload=pending,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "need": "interaction",
                    "outlet": outlet,
                    "style": style,
                    "response_owner": ownership["owner"],
                    "outward_speech_allowed": ownership["outward_speech_allowed"],
                },
            ),
        ]

        if ownership["outward_speech_allowed"]:
            events.append(
                Event(
                    topic=SPEECH_REASON_TOPIC,
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
                        "response_owner": ownership["owner"],
                    },
                )
            )
        return events

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
        threshold = _safe_float(
            await ctx.get_kv("drive:interaction:signal_threshold", SIGNAL_THRESHOLD),
            SIGNAL_THRESHOLD,
        )
        if bool(pressure.get("stimulus_input", False)):
            threshold = min(
                threshold,
                _safe_float(
                    await ctx.get_kv("drive:interaction:input_signal_threshold", INPUT_SIGNAL_THRESHOLD),
                    INPUT_SIGNAL_THRESHOLD,
                ),
            )
        if urgency < threshold:
            return []

        this_fp = str(pressure.get("pending_fingerprint", "") or "")
        last_signal_fp = str(await ctx.get_kv("drive:interaction:last_signal_fingerprint", "") or "")
        same_signal = bool(this_fp and last_signal_fp and this_fp == last_signal_fp)
        last_signal_ts = _safe_float(await ctx.get_kv("drive:interaction:last_signal_ts", 0.0), 0.0)
        cooldown_s = _safe_float(
            await ctx.get_kv("drive:interaction:signal_cooldown_s", SIGNAL_COOLDOWN_S),
            SIGNAL_COOLDOWN_S,
        )
        if same_signal and last_signal_ts > 0.0 and (now - last_signal_ts) < cooldown_s:
            return []

        last_relief_ts = _safe_float(await ctx.get_kv("drive:interaction:last_relief_ts", 0.0), 0.0)
        quiet_after_relief_s = _safe_float(
            await ctx.get_kv("drive:interaction:quiet_after_relief_s", QUIET_AFTER_RELIEF_S),
            QUIET_AFTER_RELIEF_S,
        )
        if last_relief_ts > 0.0 and (now - last_relief_ts) < quiet_after_relief_s:
            return []

        # Keep it quieter while power is being actively handled.
        if charging and urgency < CHARGING_RELEASE_MIN_URGENCY:
            return []

        return await self._emit_request(ctx, event, pressure, vector, now)


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "percept/text", "act/speech", "event/relief/interaction"],
        output_topics=[INTERACTION_REQUEST_TOPIC, SPEECH_REASON_TOPIC],
        priority=8,
        cooldown_sec=0.0,
    )
    yield InteractionReleaseVectorNeuron(cfg)
