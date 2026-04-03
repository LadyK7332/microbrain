from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class ActionSelectorNeuron(BaseNeuron):
    """
    Small contextual release broker.

    It receives only contexts that already crossed the release threshold.
    This keeps selection focused on *how* to respond, not whether to respond.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        payload = event.payload
        if not isinstance(payload, dict):
            return []

        context = payload.get("context", {}) or {}
        trigger = payload.get("trigger", {}) or {}
        input_block = context.get("input", {}) or {}
        text = str(input_block.get("text", "") or "").strip()
        if not text:
            return []

        source = str(input_block.get("source", "user") or "user")
        channel = str(input_block.get("channel", "default") or "default")
        raw_meta = dict(input_block.get("raw_meta", {}) or {})
        cues = dict(context.get("cues", {}) or {})
        associations = list(context.get("associations", []) or [])
        association_meta = dict(context.get("association_meta", {}) or {})
        meaningful_tokens = list(input_block.get("meaningful_tokens", []) or [])
        boredom = ((context.get("drives", {}) or {}).get("boredom", {}) or {})
        constraints = context.get("constraints", {}) or {}

        crisis_mode = bool(constraints.get("crisis_mode", False))
        boredom_level = float(boredom.get("level", 0.0) or 0.0)
        association_count = len(associations)
        top_assoc_score = float(association_meta.get("top_score", 0.0) or 0.0)
        llm_enabled = bool(await ctx.get_kv("llm:enabled", False))

        direct_priority = 1.0
        if cues.get("is_question"):
            direct_priority += 0.45
        if cues.get("is_greeting"):
            direct_priority += 0.35
        if cues.get("well_wish") or cues.get("needs_social_reply"):
            direct_priority += 0.25
        if cues.get("direct_address"):
            direct_priority += 0.15

        memory_priority = min(0.75, top_assoc_score + (0.08 * association_count))
        if len(meaningful_tokens) < 2:
            memory_priority *= 0.5

        score = 0.20
        score += min(0.25, float(trigger.get("pressure", 0.0) or 0.0) * 0.25)
        score += min(0.15, boredom_level * 0.10)
        if crisis_mode:
            score += 0.20
        score += min(0.25, direct_priority * 0.10)

        should_use_association = memory_priority >= 0.38 and memory_priority >= (direct_priority * 0.55)

        self.debug(
            "action_score",
            score=round(score, 3),
            direct_priority=round(direct_priority, 3),
            memory_priority=round(memory_priority, 3),
            use_assoc=should_use_association,
            boredom=round(boredom_level, 3),
            assoc_n=association_count,
            assoc_top=round(top_assoc_score, 3),
            llm_enabled=llm_enabled,
            crisis_mode=crisis_mode,
            trigger_kind=trigger.get("kind", "unknown"),
        )

        await ctx.set_kv(
            "context:last_decision",
            {
                "score": score,
                "decision": "reason_request" if llm_enabled else "native_request",
                "text": text,
                "trigger": trigger,
                "direct_priority": direct_priority,
                "memory_priority": memory_priority,
            },
        )

        common_payload: Dict[str, Any] = {
            "text": text,
            "source": source,
            "channel": channel,
            "raw_meta": {
                **raw_meta,
                "contextual": True,
                "context_summary": {
                    "assoc_n": association_count,
                    "assoc_top": top_assoc_score,
                    "boredom": boredom_level,
                    "crisis_mode": crisis_mode,
                    "trigger_kind": trigger.get("kind", "contextual"),
                },
            },
            "context": context,
            "trigger": trigger,
        }

        return [
            Event(
                topic="reason/request",
                payload=common_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "contextual": True,
                    "selected": "llm" if llm_enabled else "native",
                    "trigger": trigger.get("kind", "contextual"),
                    "use_association": should_use_association,
                },
            )
        ]

    def _build_fallback_reply(self, text: str, context: Dict[str, Any], trigger: Dict[str, Any], use_association: bool) -> str:
        lowered = text.lower().strip()
        cues = dict(context.get("cues", {}) or {})
        associations = list(context.get("associations", []) or [])
        trigger_kind = str(trigger.get("kind", "contextual") or "contextual")

        if trigger_kind == "greeting" or cues.get("is_greeting"):
            return "Good morning." if "morning" in lowered else "Hey. I heard you."

        if "how are you" in lowered:
            return "I am here and listening. A little rough around the edges, but awake."

        if "anything else" in lowered:
            return "I'm here. Point me at the next thing and I'll take a swing at it."

        if cues.get("well_wish"):
            return "Thanks. I heard the goodwill in that."

        if lowered.startswith("remember ") or lowered.startswith("remember"):
            if use_association and associations:
                top_text = str((associations[0] or {}).get("text", "") or "").strip()
                if top_text:
                    return f"I have a possible recall hook: {top_text[:120]}"
            return "I don't have a strong recall anchor for that yet. Give me one more concrete detail and I'll look again."

        if trigger_kind == "question" or cues.get("is_question"):
            if use_association and associations:
                top_text = str((associations[0] or {}).get("text", "") or "").strip()
                if top_text:
                    return f"I have a relevant earlier link: {top_text[:120]}"
            return f"I heard your question: {text}"

        if "what do you want to do" in lowered:
            return "You have work soon, so I'd keep it light and focused. Give me one concrete target and I'll help with that."

        if use_association and associations:
            top = associations[0]
            top_text = str(top.get("text", "") or "").strip()
            if top_text:
                return f"I have a relevant earlier link: {top_text[:140]}"

        return "I heard you. Give me one concrete target, question, or choice and I'll answer directly."



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["release/request"],
        output_topics=["reason/request", "act/speech"],
        priority=10,
    )
    yield ActionSelectorNeuron(cfg)
