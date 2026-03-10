from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        power_state = await ctx.get_kv("power:state", {}) or {}
        r_pending = bool(await ctx.get_kv("control:r_pending", False))

        boredom_level = float((boredom or {}).get("level", 0.0) or 0.0)
        stress_level = float((stress or {}).get("level", 0.0) or 0.0)

        salience = 0.0
        if isinstance(global_salience, (float, int)):
            salience = float(global_salience)
        elif isinstance(affect_state, dict):
            salience = float(affect_state.get("salience", 0.0) or 0.0)

        warmth = float(getattr(pdna, "warmth", 0.6) if pdna is not None else 0.6)
        introspection = float(getattr(pdna, "introspection", 0.6) if pdna is not None else 0.6)
        focus = float(getattr(pdna, "focus", 0.6) if pdna is not None else 0.6)
        energy = float(getattr(pdna, "energy", 0.5) if pdna is not None else 0.5)
        support_level = float(getattr(pdna, "support_level", 0.7) if pdna is not None else 0.7)

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

        needs = {
            "stimulation": round(stimulation_need, 4),
            "social": round(social_need, 4),
            "coherence": round(coherence_need, 4),
            "continuity": round(continuity_need, 4),
            "safety": round(stress_level, 4),
            "salience": round(_clamp(salience), 4),
        }

        hormones = dict(state.get("hormones", {}) or {})
        prev_arousal = float(hormones.get("arousal", 0.15) or 0.15)
        prev_inquiry = float(hormones.get("inquiry", 0.10) or 0.10)
        prev_affiliation = float(hormones.get("affiliation", 0.10) or 0.10)
        prev_settling = float(hormones.get("settling", 0.80) or 0.80)

        arousal = _clamp(
            (0.72 * prev_arousal)
            + (0.18 * salience)
            + (0.20 * stress_level)
            + (0.10 * stimulation_need)
            + (0.06 * energy)
        )
        inquiry = _clamp(
            (0.68 * prev_inquiry)
            + (0.26 * coherence_need)
            + (0.22 * continuity_need)
            + (0.12 * stimulation_need)
            + (0.06 * focus)
        )
        affiliation = _clamp(
            (0.76 * prev_affiliation)
            + (0.22 * social_need)
            + (0.08 * warmth)
        )
        settling = _clamp(
            (0.70 * prev_settling)
            + (0.18 * (1.0 - stress_level))
            + (0.08 * (1.0 - coherence_need))
        )

        hormones = {
            "arousal": round(arousal, 4),
            "inquiry": round(inquiry, 4),
            "affiliation": round(affiliation, 4),
            "settling": round(settling, 4),
        }
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

        think_pressure = _clamp(
            (0.42 * inquiry)
            + (0.22 * continuity_need)
            + (0.14 * stimulation_need)
            + (0.10 * salience)
            + (0.08 * introspection)
            - (0.18 * overload)
        )
        talk_pressure = _clamp(
            (0.34 * inquiry)
            + (0.24 * continuity_need)
            + (0.18 * affiliation)
            + (0.12 * social_need)
            + (0.08 * salience)
            - (0.20 * overload)
            - (0.18 * interruption_cost)
        )

        if not bool(pending_flags.get("clarify_ready", False)):
            talk_pressure *= 0.60

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
        }

        await ctx.set_kv("drive:needs_stack", needs)
        await ctx.set_kv("drive:hormones", hormones)
        await ctx.set_kv("initiative:last", initiative_snapshot)
        await ctx.set_kv("initiative:tier", new_tier)

        out: List[Event] = []
        emitted_thought = False

        if (
            new_tier >= 2
            and not sleeping
            and time_since_user >= 1.5
            and (prev_tier < 2 or (now - float(state.get("last_thought_ts", 0.0) or 0.0)) >= 35.0)
        ):
            thought_prompt = self._build_internal_prompt(
                needs=needs,
                hormones=hormones,
                pending_text=pending_text,
                pending_flags=pending_flags,
            )
            out.append(
                Event(
                    topic="reason/request",
                    payload={
                        "text": thought_prompt,
                        "source": "internal",
                        "channel": "thought",
                        "raw_meta": {
                            "mode": "initiative_reflection",
                            "tier": new_tier,
                            "needs": needs,
                            "hormones": hormones,
                        },
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "initiative_reflection"},
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
            and (now - float(state.get("last_clarify_ts", 0.0) or 0.0)) >= 30.0
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

        coherence_score = 0.0
        coherence_score += 0.30 if has_question else 0.0
        coherence_score += min(0.35, 0.12 * marker_hits)
        coherence_score += 0.20 if option_hits > 0 else 0.0
        coherence_score += 0.10 if has_error_language else 0.0
        coherence_score = _clamp(coherence_score)

        clarify_ready = bool(
            has_question
            or option_hits > 0
            or has_error_language
            or (goal_hits > 0 and marker_hits > 0)
            or marker_hits >= 2
        )

        return {
            "has_question": has_question,
            "has_options": option_hits > 0,
            "has_error_language": has_error_language,
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
        enter_t1 = 0.34
        leave_t1 = 0.24
        enter_t2 = 0.56
        leave_t2 = 0.42
        enter_t3 = 0.78
        leave_t3 = 0.58

        tier = prev_tier
        if prev_tier >= 3:
            tier = 3 if (talk_pressure >= leave_t3 and clarify_ready) else 2
        elif prev_tier == 2:
            if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
                tier = 3
            elif tier_score >= leave_t2:
                tier = 2
            elif tier_score >= leave_t1:
                tier = 1
            else:
                tier = 0
        elif prev_tier == 1:
            if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
                tier = 3
            elif think_pressure >= enter_t2:
                tier = 2
            elif tier_score >= leave_t1:
                tier = 1
            else:
                tier = 0
        else:
            if talk_pressure >= enter_t3 and clarify_ready and interruption_cost < 0.55:
                tier = 3
            elif think_pressure >= enter_t2:
                tier = 2
            elif tier_score >= enter_t1:
                tier = 1
            else:
                tier = 0
        return tier

    def _build_internal_prompt(
        self,
        *,
        needs: Dict[str, Any],
        hormones: Dict[str, Any],
        pending_text: str,
        pending_flags: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append("Internal reflection only. Do not address the user.")
        lines.append("Summarize what matters in 1-3 short sentences.")
        lines.append("Focus on missing variables, unresolved continuity, or the next useful check.")
        lines.append("")
        lines.append(f"Needs: {needs}")
        lines.append(f"Hormones: {hormones}")
        if pending_text:
            lines.append(f"Pending text: {pending_text}")
            lines.append(f"Pending flags: {pending_flags}")
        else:
            lines.append("Pending text: none")
        return "\n".join(lines)

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
        output_topics=["reason/request", "act/speech"],
        priority=-8,
    )
    yield InitiativeThresholdNeuron(cfg)
