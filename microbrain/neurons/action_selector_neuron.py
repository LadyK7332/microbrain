from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ActionSelectorNeuron(BaseNeuron):
    """Select how a released hypothesis should enter the reasoning path."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        trigger = payload.get("trigger", {}) if isinstance(payload.get("trigger", {}), Mapping) else {}
        hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        text = str(input_block.get("text", "") or "").strip()
        if not text:
            return []

        source = str(input_block.get("source", "user") or "user")
        channel = str(input_block.get("channel", "default") or "default")
        raw_meta = dict(input_block.get("raw_meta", {}) or {})
        cues = context.get("cues", {}) if isinstance(context.get("cues", {}), Mapping) else {}
        associations = [dict(item) for item in list(context.get("associations", []) or []) if isinstance(item, Mapping)]
        association_meta = context.get("association_meta", {}) if isinstance(context.get("association_meta", {}), Mapping) else {}
        meaningful_tokens = list(input_block.get("meaningful_tokens", []) or [])
        drives = context.get("drives", {}) if isinstance(context.get("drives", {}), Mapping) else {}
        boredom = drives.get("boredom", {}) if isinstance(drives.get("boredom", {}), Mapping) else {}
        social_interaction = drives.get("social_interaction", {}) if isinstance(drives.get("social_interaction", {}), Mapping) else {}
        social_experimentation = drives.get("social_experimentation", {}) if isinstance(drives.get("social_experimentation", {}), Mapping) else {}
        thought_momentum = context.get("thought_momentum", {}) if isinstance(context.get("thought_momentum", {}), Mapping) else {}
        conversation_summary = context.get("conversation_summary", {}) if isinstance(context.get("conversation_summary", {}), Mapping) else {}
        constraints = context.get("constraints", {}) if isinstance(context.get("constraints", {}), Mapping) else {}
        pattern = hypothesis.get("pattern_analysis", {}) if isinstance(hypothesis.get("pattern_analysis", {}), Mapping) else {}
        memory_check = hypothesis.get("memory_check", {}) if isinstance(hypothesis.get("memory_check", {}), Mapping) else {}

        crisis_mode = bool(constraints.get("crisis_mode", False))
        boredom_level = _safe_float(boredom.get("level", 0.0))
        social_level = _safe_float(social_interaction.get("level", 0.0))
        social_experiment_pressure = _safe_float(social_experimentation.get("pressure", 0.0))
        momentum_pressure = _safe_float(thought_momentum.get("pressure", 0.0))
        momentum_intent = str(thought_momentum.get("dominant_intent", "") or "")
        association_count = len(associations)
        top_assoc_score = _safe_float(association_meta.get("top_score", 0.0))
        llm_enabled = bool(await ctx.get_kv("llm:enabled", False))

        selected_action = str(
            trigger.get("recommended_action", "")
            or hypothesis.get("recommended_action", "")
            or "respond"
        )
        response_demand = _safe_float(hypothesis.get("response_demand", 0.0))
        action_score = _safe_float(hypothesis.get("recommended_action_score", 0.0))
        statement_kind = str(pattern.get("statement_kind", "statement") or "statement")
        uncertainty = _safe_float(pattern.get("uncertainty", 0.0))
        continuity = _safe_float(pattern.get("continuity", 0.0))

        direct_priority = 1.0 + (0.35 * response_demand) + (0.18 * action_score)
        if cues.get("is_question"):
            direct_priority += 0.45
        if cues.get("is_greeting"):
            direct_priority += 0.35
        if cues.get("well_wish") or cues.get("needs_social_reply"):
            direct_priority += 0.25
        if cues.get("direct_address"):
            direct_priority += 0.15
        if selected_action in {"clarify", "ask_followup"}:
            direct_priority += min(0.16, uncertainty * 0.16)
        if selected_action in {"continue_thread", "reflect", "acknowledge", "acknowledge_revision"}:
            direct_priority += min(0.14, continuity * 0.14)
        if social_level >= 0.45 and (
            cues.get("is_greeting") or cues.get("needs_social_reply") or cues.get("direct_address")
        ):
            direct_priority += min(0.18, social_level * 0.16)

        deep_evidence = [
            dict(item)
            for item in list(memory_check.get("evidence", []) or [])
            if isinstance(item, Mapping)
        ]
        memory_priority = min(
            1.0,
            top_assoc_score
            + (0.08 * association_count)
            + min(0.24, 0.04 * len(deep_evidence)),
        )
        if len(meaningful_tokens) < 2:
            memory_priority *= 0.5
        if selected_action in {"answer", "investigate", "reflect"}:
            memory_priority += min(0.18, response_demand * 0.18)

        score = 0.20
        score += min(0.25, _safe_float(trigger.get("pressure", 0.0)) * 0.25)
        score += min(0.18, response_demand * 0.18)
        score += min(0.12, action_score * 0.12)
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

        should_use_association = memory_priority >= 0.38 and memory_priority >= (direct_priority * 0.45)
        decision_route = "reason_request" if llm_enabled else "native_request"

        decision = {
            "score": round(score, 6),
            "decision": decision_route,
            "text": text,
            "trigger": dict(trigger),
            "hypothesis_id": str(hypothesis.get("hypothesis_id", "") or ""),
            "selected_action": selected_action,
            "statement_kind": statement_kind,
            "response_demand": round(response_demand, 6),
            "action_score": round(action_score, 6),
            "direct_priority": round(direct_priority, 6),
            "memory_priority": round(memory_priority, 6),
            "social_level": round(social_level, 6),
            "social_experiment_pressure": round(social_experiment_pressure, 6),
            "thought_momentum_pressure": round(momentum_pressure, 6),
            "thought_momentum_intent": momentum_intent,
        }
        await ctx.set_kv("context:last_decision", decision)
        await ctx.set_kv(
            "hypothesis:pending_action",
            {
                "hypothesis_id": decision["hypothesis_id"],
                "correlation_id": event.correlation_id,
                "selected_action": selected_action,
                "predicted_outcomes": [
                    dict(item)
                    for item in list(hypothesis.get("action_candidates", []) or [])
                    if isinstance(item, Mapping) and str(item.get("action", "") or "") == selected_action
                ][:1],
                "created_at": float(hypothesis.get("created_at", 0.0) or 0.0),
                "awaiting_observation": True,
            },
        )

        self.debug(
            "action_score",
            score=round(score, 3),
            selected_action=selected_action,
            statement_kind=statement_kind,
            response_demand=round(response_demand, 3),
            direct_priority=round(direct_priority, 3),
            memory_priority=round(memory_priority, 3),
            use_assoc=should_use_association,
            memory_mode=str(memory_check.get("mode", "working") or "working"),
            llm_enabled=llm_enabled,
            crisis_mode=crisis_mode,
        )

        common_payload: Dict[str, Any] = {
            "text": text,
            "source": source,
            "channel": channel,
            "raw_meta": {
                **raw_meta,
                "contextual": True,
                "hypothesis_id": decision["hypothesis_id"],
                "selected_action": selected_action,
                "response_demand": response_demand,
                "context_summary": {
                    "assoc_n": association_count,
                    "assoc_top": top_assoc_score,
                    "boredom": boredom_level,
                    "social": social_level,
                    "social_experiment": social_experiment_pressure,
                    "thought_momentum": momentum_pressure,
                    "thought_momentum_intent": momentum_intent,
                    "conversation_topic": str(conversation_summary.get("topic", "") or ""),
                    "conversation_threads": list(conversation_summary.get("active_threads", []) or [])[:6],
                    "crisis_mode": crisis_mode,
                    "trigger_kind": trigger.get("kind", "contextual"),
                    "statement_kind": statement_kind,
                    "selected_action": selected_action,
                    "response_demand": response_demand,
                    "hypothesis_uncertainty": uncertainty,
                    "hypothesis_continuity": continuity,
                    "memory_mode": str(memory_check.get("mode", "working") or "working"),
                },
            },
            "context": dict(context),
            "hypothesis": dict(hypothesis),
            "selected_action": selected_action,
            "trigger": dict(trigger),
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
                    "selected_action": selected_action,
                    "hypothesis_id": decision["hypothesis_id"],
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
