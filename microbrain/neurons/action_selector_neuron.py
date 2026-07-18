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
        drives = context.get("drives", {}) or {}
        boredom = (drives.get("boredom", {}) or {})
        social_interaction = (drives.get("social_interaction", {}) or {})
        social_experimentation = (drives.get("social_experimentation", {}) or {})
        thought_momentum = context.get("thought_momentum", {}) or {}
        conversation_summary = context.get("conversation_summary", {}) or {}
        constraints = context.get("constraints", {}) or {}

        crisis_mode = bool(constraints.get("crisis_mode", False))
        boredom_level = float(boredom.get("level", 0.0) or 0.0)
        social_level = float(social_interaction.get("level", 0.0) or 0.0)
        social_experiment_pressure = float(social_experimentation.get("pressure", 0.0) or 0.0)
        momentum_pressure = float(thought_momentum.get("pressure", 0.0) or 0.0) if isinstance(thought_momentum, dict) else 0.0
        momentum_intent = str(thought_momentum.get("dominant_intent", "") or "") if isinstance(thought_momentum, dict) else ""
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
        if social_level >= 0.45 and (cues.get("is_greeting") or cues.get("needs_social_reply") or cues.get("direct_address")):
            direct_priority += min(0.18, social_level * 0.16)

        memory_priority = min(0.75, top_assoc_score + (0.08 * association_count))
        if len(meaningful_tokens) < 2:
            memory_priority *= 0.5

        score = 0.20
        score += min(0.25, float(trigger.get("pressure", 0.0) or 0.0) * 0.25)
        score += min(0.15, boredom_level * 0.10)
        score += min(0.12, social_level * 0.08)
        if social_experiment_pressure >= 0.55:
            score += min(0.10, social_experiment_pressure * 0.08)
        if momentum_pressure >= 0.25:
            score += min(0.10, momentum_pressure * 0.09)
            if momentum_intent in {"understand_user", "resolve_thread", "await_result"}:
                direct_priority += min(0.10, momentum_pressure * 0.08)
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
            social=round(social_level, 3),
            social_experiment=round(social_experiment_pressure, 3),
            momentum=round(momentum_pressure, 3),
            momentum_intent=momentum_intent,
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
                "social_level": social_level,
                "social_experiment_pressure": social_experiment_pressure,
                "thought_momentum_pressure": momentum_pressure,
                "thought_momentum_intent": momentum_intent,
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
                    "social": social_level,
                    "social_experiment": social_experiment_pressure,
                    "thought_momentum": momentum_pressure,
                    "thought_momentum_intent": momentum_intent,
                    "conversation_topic": str(conversation_summary.get("topic", "") or "") if isinstance(conversation_summary, dict) else "",
                    "conversation_threads": list(conversation_summary.get("active_threads", []) or [])[:6] if isinstance(conversation_summary, dict) else [],
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



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["release/request"],
        output_topics=["reason/request"],
        priority=10,
    )
    yield ActionSelectorNeuron(cfg)
