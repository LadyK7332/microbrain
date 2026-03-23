from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.hormone import derive_ddna_modulators, derive_rosehip_state, merge_need_maps
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

_UNCERTAINTY_PATTERNS = (
    "?",
    "not sure",
    "unsure",
    "unclear",
    "maybe",
    "might",
    "probably",
    "i think",
    "i guess",
    "issue",
    "problem",
    "error",
    "stuck",
    "doesn't work",
    "not working",
    "please respond",
    "respond",
    "reply",
    "speak up",
    "can you hear me",
)

_OPTION_PATTERNS = (
    " or ",
    " either ",
    " vs ",
)

_GOAL_WORDS = (
    "fix",
    "update",
    "change",
    "build",
    "make",
    "implement",
    "add",
    "remove",
    "wire",
    "patch",
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class InitiativeThresholdNeuron(BaseNeuron):
    """
    Tiered initiative / reflection arbiter.

    Purpose:
      - Build a small needs stack from existing MB signals.
      - Blend needs into slow-moving virtual-hormone style state.
      - Use threshold tiers + hysteresis to decide whether MB should:
          tier 0: stay quiet
          tier 1: mark an internal unresolved state
          tier 2: think internally
          tier 3: ask one concise clarification outwardly

    This intentionally prefers constrained, objective behavior.
    It does NOT try to babble or generate open-ended chatter.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in (
            "clock/tick",
            "percept/text",
            "percept/vision",
            "act/speech",
            "affect/state",
            "affect/salience",
        ):
            return []

        now = time.time()
        state = await self.load_state(
            ctx,
            "initiative_state",
            default={
                "last_user_ts": now,
                "last_external_ts": now,
                "last_speech_ts": 0.0,
                "pending_text": "",
                "pending_since": 0.0,
                "pending_flags": {},
                "pending_answered": False,
                "last_user_channel": "repl",
                "tier": 0,
                "last_thought_ts": 0.0,
                "last_clarify_ts": 0.0,
                "clarify_said": False,
                "hormones": {
                    "arousal": 0.15,
                    "inquiry": 0.10,
                    "affiliation": 0.10,
                    "settling": 0.80,
                },
            },
        )

        await ctx.set_kv("initiative:block_babble", True)

        if event.topic == "percept/text":
            payload = event.payload if isinstance(event.payload, dict) else {}
            text = str(payload.get("text", "") or "").strip()
            raw_meta = payload.get("raw_meta", {}) or {}
            src = str(raw_meta.get("source", payload.get("source", "user")) or "user")
            channel = str(raw_meta.get("channel", payload.get("channel", "repl")) or "repl")

            if src not in ("assistant", "system", "mb") and text:
                state["last_user_ts"] = now
                state["last_external_ts"] = now
                state["last_user_channel"] = channel
                state["pending_text"] = text[:280]
                state["pending_since"] = now
                state["pending_flags"] = self._classify_text(text)
                state["pending_answered"] = False
                state["clarify_said"] = False

        elif event.topic == "percept/vision":
            state["last_external_ts"] = now

        elif event.topic == "act/speech":
            payload = event.payload if isinstance(event.payload, dict) else {}
            channel = str(payload.get("channel", "repl") or "repl")
            style = str(payload.get("style", "") or "")
            state["last_speech_ts"] = now

            if channel != "thought":
                state["last_external_ts"] = now

            if style in ("assistant", "system") and state.get("pending_text"):
                state["pending_answered"] = True
                flags = state.get("pending_flags", {}) or {}
                if not bool(flags.get("clarify_ready", False)):
                    state["pending_text"] = ""
                    state["pending_since"] = 0.0
                    state["pending_flags"] = {}
                    state["clarify_said"] = False

        boredom = await ctx.get_kv("drive:boredom", {}) or {}
        stress = await ctx.get_kv("drive:stress", {}) or {}
        affect_state = await ctx.get_kv("affect:state", {}) or {}
        global_salience = await ctx.get_kv("affect:global_salience", None)
        pdna = await ctx.get_kv("pdna:profile", None)
        ddna_mods = await ctx.get_kv("drive:ddna_modulators", None)
        base_needs = await ctx.get_kv("drive:needs_base", {}) or {}
        shared_hormones = await ctx.get_kv("drive:hormones", {}) or {}
        power_state = await ctx.get_kv("power:state", {}) or {}
        r_pending = bool(await ctx.get_kv("control:r_pending", False))

        boredom_level = float((boredom or {}).get("level", 0.0) or 0.0)
        stress_level = float((stress or {}).get("level", 0.0) or 0.0)

        salience = 0.0
        if isinstance(global_salience, (float, int)):
            salience = float(global_salience)
        elif isinstance(affect_state, dict):
            salience = float(affect_state.get("salience", 0.0) or 0.0)

        if not isinstance(ddna_mods, dict) or not ddna_mods:
            ddna_mods = derive_ddna_modulators(pdna)

        introspection = float(getattr(pdna, "introspection", 0.6) if pdna is not None else 0.6)
        focus = float(getattr(pdna, "focus", 0.6) if pdna is not None else 0.6)
        energy = float(getattr(pdna, "energy", 0.5) if pdna is not None else 0.5)
        support_level = float(getattr(pdna, "support_level", 0.7) if pdna is not None else 0.7)

        expression_bias = float((ddna_mods or {}).get("expression_bias", 1.0) or 1.0)
        restraint_bias = float((ddna_mods or {}).get("restraint_bias", 1.0) or 1.0)
        caution_gain = float((ddna_mods or {}).get("caution_gain", 1.0) or 1.0)
        persistence_gain = float((ddna_mods or {}).get("persistence_gain", 1.0) or 1.0)

        time_since_user = max(0.0, now - float(state.get("last_user_ts", now) or now))
        time_since_external = max(0.0, now - float(state.get("last_external_ts", now) or now))
        time_since_speech = max(0.0, now - float(state.get("last_speech_ts", 0.0) or 0.0))
        pending_text = str(state.get("pending_text", "") or "")
        pending_since = float(state.get("pending_since", 0.0) or 0.0)
        pending_age = max(0.0, now - pending_since) if pending_text else 0.0
        pending_flags = state.get("pending_flags", {}) or {}

        social_need = _clamp(max(0.0, (time_since_user - 12.0) / 120.0) * (0.35 + 0.65 * support_level))
        stimulation_need = _clamp(boredom_level)
        coherence_need = _clamp(float(pending_flags.get("coherence_score", 0.0) or 0.0) + (0.20 * stress_level) + (0.10 * salience))

        continuity_need = 0.0
        if pending_text:
            continuity_need = _clamp(0.20 + min(0.55, pending_age / 90.0))
            if bool(state.get("pending_answered", False)):
                continuity_need *= 0.45
            if bool(state.get("clarify_said", False)):
                continuity_need *= 0.50

        needs = merge_need_maps(
            base_needs,
            {
                "stimulation": round(stimulation_need, 4),
                "social": round(social_need, 4),
                "coherence": round(coherence_need, 4),
                "continuity": round(continuity_need, 4),
                "safety": round(stress_level, 4),
                "salience": round(_clamp(salience), 4),
                "novelty": round(max(stimulation_need, min(1.0, time_since_external / 90.0)), 4),
            },
        )

        hormones = dict(shared_hormones or {}) if isinstance(shared_hormones, dict) else {}
        state["hormones"] = hormones

        sleeping = bool((power_state or {}).get("sleep", False))
        overload = 1.0 if sleeping else 0.0
        interruption_cost = 0.0
        if time_since_user < 2.5:
            interruption_cost += 0.45
        if time_since_speech < 4.0:
            interruption_cost += 0.25
        if r_pending:
            interruption_cost += 0.60

        arousal = float(hormones.get("arousal", 0.15) or 0.15)
        inquiry = float(hormones.get("inquiry", 0.10) or 0.10)
        affiliation = float(hormones.get("affiliation", 0.10) or 0.10)
        caution = float(hormones.get("caution", 0.20) or 0.20)
        settling = float(hormones.get("settling", 0.80) or 0.80)
        persistence = float(hormones.get("persistence", 0.45) or 0.45)
        continuity_h = float(hormones.get("continuity", continuity_need) or continuity_need)

        think_pressure = _clamp(
            (0.36 * inquiry)
            + (0.20 * continuity_h)
            + (0.12 * stimulation_need)
            + (0.08 * salience)
            + (0.08 * introspection)
            + (0.08 * persistence * persistence_gain)
            - (0.12 * caution * caution_gain)
            - (0.18 * overload)
        )
        talk_pressure = _clamp(
            expression_bias * (
                (0.28 * inquiry)
                + (0.22 * continuity_h)
                + (0.18 * affiliation)
                + (0.10 * social_need)
                + (0.06 * salience)
                + (0.04 * arousal)
            )
            - (0.14 * caution * caution_gain)
            - (0.12 * interruption_cost)
            - (0.10 * max(0.0, restraint_bias - 1.0))
            - (0.20 * overload)
        )

        if not bool(pending_flags.get("clarify_ready", False)):
            talk_pressure *= 0.60
        if settling > 0.80 and not pending_text:
            talk_pressure *= 0.85

        rosehip_enabled = bool(await ctx.get_kv("rosehip:enabled", True))
        direct_address = 1.0 if (bool(pending_flags.get("has_question", False)) or bool(pending_flags.get("has_response_request", False))) else 0.0
        recent_user = _clamp(1.0 - (time_since_user / max(1.0, float(await ctx.get_kv("rosehip:conversation_hold_s", 12.0) or 12.0))))
        redundancy = 0.0
        repeat_window_s = float(await ctx.get_kv("rosehip:repeat_reply_window_s", 18.0) or 18.0)
        if time_since_speech < repeat_window_s:
            redundancy += _clamp(1.0 - (time_since_speech / max(1.0, repeat_window_s)))
        if bool(state.get("clarify_said", False)):
            redundancy += 0.25
        redundancy = _clamp(redundancy)
        confidence = _clamp(0.55 + (0.25 * float(pending_flags.get("coherence_score", 0.0) or 0.0)) + (0.10 if pending_text else 0.0))
        rosehip = derive_rosehip_state(
            hormones,
            needs=needs,
            ddna=ddna_mods,
            context={
                "interruption_cost": interruption_cost,
                "redundancy": redundancy,
                "confidence": confidence,
                "direct_address": direct_address,
                "recent_user": recent_user,
                "answered": 1.0 if bool(state.get("pending_answered", False)) else 0.0,
                "sleeping": sleeping,
                "charging": bool((power_state or {}).get("charging", False)),
            },
        ) if rosehip_enabled else {}
        await ctx.set_kv("drive:rosehip", rosehip)

        if rosehip:
            think_pressure = _clamp((think_pressure * float(rosehip.get("internal_scale", 1.0) or 1.0)) + (0.08 * float(rosehip.get("internal_bias", 0.0) or 0.0)))
            talk_pressure = _clamp(
                (talk_pressure * float(rosehip.get("outward_scale", 1.0) or 1.0))
                + (0.10 * float(rosehip.get("external_bias", 0.0) or 0.0))
                - (0.18 * float(rosehip.get("expression_brake", 0.0) or 0.0))
                - (0.12 * float(rosehip.get("redundancy_brake", 0.0) or 0.0))
                - (0.16 * float(rosehip.get("interrupt_brake", 0.0) or 0.0))
                - (0.20 * float(rosehip.get("sleep_quiet_brake", 0.0) or 0.0))
                - (0.10 * float(rosehip.get("confidence_brake", 0.0) or 0.0))
            )
            if direct_address > 0.0:
                talk_pressure = max(talk_pressure, min(float(rosehip.get("direct_reply_floor", 0.0) or 0.0), 0.85))

        tier_score = max(think_pressure, talk_pressure)
        prev_tier = int(state.get("tier", 0) or 0)
        new_tier = self._select_tier(
            prev_tier=prev_tier,
            tier_score=tier_score,
            think_pressure=think_pressure,
            talk_pressure=talk_pressure,
            clarify_ready=bool(pending_flags.get("clarify_ready", False)),
            interruption_cost=interruption_cost,
        )
        state["tier"] = new_tier

        initiative_snapshot = {
            "tier": new_tier,
            "think_pressure": round(think_pressure, 4),
            "talk_pressure": round(talk_pressure, 4),
            "interruption_cost": round(interruption_cost, 4),
            "clarify_ready": bool(pending_flags.get("clarify_ready", False)),
            "pending_text": pending_text,
            "pending_age_s": round(pending_age, 3),
            "rosehip": rosehip,
        }

        await ctx.set_kv("initiative:needs_local", needs)
        await ctx.set_kv("drive:need_signal:initiative", needs)
        await ctx.set_kv("initiative:last", initiative_snapshot)
        await ctx.set_kv("initiative:tier", new_tier)

        out: List[Event] = []
        emitted_thought = False

        if (
            new_tier >= 2
            and not sleeping
            and time_since_user >= 1.5
            and (prev_tier < 2 or (now - float(state.get("last_thought_ts", 0.0) or 0.0)) >= float(await ctx.get_kv("rosehip:thought_min_interval_s", 35.0) or 35.0))
        ):
            thought_note = self._build_internal_note(
                needs=needs,
                hormones=hormones,
                pending_text=pending_text,
                pending_flags=pending_flags,
            )
            out.append(
                Event(
                    topic="reason/output",
                    payload={"text": thought_note},
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "channel": "thought",
                        "kind": "initiative_reflection",
                        "mode": "initiative_reflection",
                        "tier": new_tier,
                    },
                )
            )
            state["last_thought_ts"] = now
            emitted_thought = True

        if (
            new_tier >= 3
            and bool(pending_flags.get("clarify_ready", False))
            and not bool(state.get("clarify_said", False))
            and not sleeping
            and time_since_user >= 2.5
            and (now - float(state.get("last_clarify_ts", 0.0) or 0.0)) >= float(await ctx.get_kv("rosehip:clarify_min_interval_s", 30.0) or 30.0)
        ):
            question = self._build_clarify_text(pending_text=pending_text, flags=pending_flags)
            if question:
                out.append(
                    Event(
                        topic="act/speech",
                        payload={
                            "text": question,
                            "channel": str(state.get("last_user_channel", "repl") or "repl"),
                            "style": "assistant",
                        },
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={"kind": "initiative_clarify", "tier": new_tier},
                    )
                )
                state["clarify_said"] = True
                state["last_clarify_ts"] = now

        if new_tier == 0 and pending_text and pending_age > 180.0:
            state["pending_text"] = ""
            state["pending_since"] = 0.0
            state["pending_flags"] = {}
            state["pending_answered"] = False
            state["clarify_said"] = False

        await self.save_state(ctx, "initiative_state", state)
        self.debug(
            "initiative_state",
            tier=new_tier,
            think_pressure=round(think_pressure, 3),
            talk_pressure=round(talk_pressure, 3),
            pending=bool(pending_text),
            emitted_thought=emitted_thought,
        )
        return out

    def _classify_text(self, text: str) -> Dict[str, Any]:
        lowered = text.lower().strip()
        marker_hits = sum(1 for pat in _UNCERTAINTY_PATTERNS if pat in lowered)
        option_hits = sum(1 for pat in _OPTION_PATTERNS if pat in lowered)
        goal_hits = sum(1 for pat in _GOAL_WORDS if pat in lowered)
        has_question = "?" in lowered
        has_error_language = any(pat in lowered for pat in ("error", "issue", "problem", "not working", "stuck"))
        has_response_request = any(pat in lowered for pat in ("please respond", "respond", "reply", "speak up", "can you hear me"))

        coherence_score = 0.0
        coherence_score += 0.30 if has_question else 0.0
        coherence_score += 0.18 if has_response_request else 0.0
        coherence_score += min(0.35, 0.12 * marker_hits)
        coherence_score += 0.20 if option_hits > 0 else 0.0
        coherence_score += 0.10 if has_error_language else 0.0
        coherence_score = _clamp(coherence_score)

        clarify_ready = bool(
            has_question
            or option_hits > 0
            or has_error_language
            or has_response_request
            or (goal_hits > 0 and marker_hits > 0)
            or marker_hits >= 2
        )

        return {
            "has_question": has_question,
            "has_options": option_hits > 0,
            "has_error_language": has_error_language,
            "has_response_request": has_response_request,
            "goal_hits": goal_hits,
            "marker_hits": marker_hits,
            "coherence_score": round(coherence_score, 4),
            "clarify_ready": clarify_ready,
        }

    def _select_tier(
        self,
        *,
        prev_tier: int,
        tier_score: float,
        think_pressure: float,
        talk_pressure: float,
        clarify_ready: bool,
        interruption_cost: float,
    ) -> int:
        enter_t1 = 0.22
        enter_t2 = 0.46
        enter_t3 = 0.68

        leave_t1 = 0.16
        leave_t2 = 0.34
        leave_t3 = 0.54

        if prev_tier >= 3:
            if talk_pressure >= leave_t3 and clarify_ready and interruption_cost < 0.70:
                return 3
            if think_pressure >= enter_t2:
                return 2
            if tier_score >= enter_t1:
                return 1
            return 0

        if prev_tier == 2:
            if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
                return 3
            if think_pressure >= leave_t2:
                return 2
            if tier_score >= enter_t1:
                return 1
            return 0

        if prev_tier == 1:
            if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
                return 3
            if think_pressure >= enter_t2:
                return 2
            if tier_score >= leave_t1:
                return 1
            return 0

        if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
            return 3
        if think_pressure >= enter_t2:
            return 2
        if tier_score >= enter_t1:
            return 1
        return 0

    def _build_internal_note(
        self,
        *,
        needs: Dict[str, Any],
        hormones: Dict[str, Any],
        pending_text: str,
        pending_flags: Dict[str, Any],
    ) -> str:
        compact = re.sub(r"\s+", " ", (pending_text or "").strip())
        if len(compact) > 140:
            compact = compact[:137] + "..."

        inquiry = float(hormones.get("inquiry", 0.0) or 0.0)
        caution = float(hormones.get("caution", 0.0) or 0.0)
        continuity = float(hormones.get("continuity", 0.0) or 0.0)
        coherence = float(needs.get("coherence", 0.0) or 0.0)

        notes: List[str] = []

        if compact:
            notes.append(f"pending: {compact}")
        else:
            notes.append("pending: none")

        if bool(pending_flags.get("has_response_request", False)):
            notes.append("user likely wants acknowledgement or direct outward reply")
        elif bool(pending_flags.get("has_question", False)):
            notes.append("question detected; likely needs a missing-variable check")
        elif bool(pending_flags.get("has_options", False)):
            notes.append("multiple valid paths detected; target may be missing")

        if continuity >= 0.40:
            notes.append("continuity pressure elevated")
        if coherence >= 0.40 or inquiry >= 0.45:
            notes.append("coherence / inquiry pressure suggests clarification or tighter framing")
        if caution >= 0.45:
            notes.append("caution elevated; avoid overcommitting")

        if len(notes) == 1:
            notes.append("next useful move: hold internally and wait for clearer signal")

        return " | ".join(notes[:4])

    def _build_clarify_text(self, *, pending_text: str, flags: Dict[str, Any]) -> str:
        if not pending_text:
            return ""

        compact = re.sub(r"\s+", " ", pending_text).strip()
        if len(compact) > 120:
            compact = compact[:117] + "..."

        if bool(flags.get("has_options", False)):
            return "I can take either path here. Which option should I optimize for?"
        if bool(flags.get("has_error_language", False)):
            return "I can dig in, but I need the target outcome first. What should success look like?"
        if bool(flags.get("has_response_request", False)):
            return "I hear you. What do you want me to respond with: a quick acknowledgement, an explanation, or a concrete action?"
        if bool(flags.get("has_question", False)):
            return "I have one missing variable. Do you want an explanation, a plan, or a concrete patch?"
        return f"I think the missing variable is the target outcome for: {compact} What should I optimize for?"



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "clock/tick",
            "percept/text",
            "percept/vision",
            "act/speech",
            "affect/state",
            "affect/salience",
        ],
        output_topics=["reason/output", "act/speech"],
        priority=-8,
    )
    yield InitiativeThresholdNeuron(cfg)
