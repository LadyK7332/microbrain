from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class DesireTriggerNeuron(BaseNeuron):
    """
    Minimal release trigger.

    It answers the question: given the built context, is there enough pressure
    to justify a visible response *now*?

    Emits:
        - release/request
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
        input_block = context.get("input", {}) or {}
        text = str(input_block.get("text", "") or "").strip()
        if not text:
            return []

        source = str(input_block.get("source", "user") or "user")
        channel = str(input_block.get("channel", "default") or "default")
        raw_meta = dict(input_block.get("raw_meta", {}) or {})

        cues = dict(context.get("cues", {}) or {})
        constraints = dict(context.get("constraints", {}) or {})
        boredom = ((context.get("drives", {}) or {}).get("boredom", {}) or {})
        associations = list(context.get("associations", []) or [])
        assoc_meta = dict(context.get("association_meta", {}) or {})

        boredom_level = float(boredom.get("level", 0.0) or 0.0)
        crisis_mode = bool(constraints.get("crisis_mode", False))
        allow_babble = bool(constraints.get("allow_babble", False))
        top_assoc_score = float(assoc_meta.get("top_score", 0.0) or 0.0)

        pressure = 0.0
        reasons = []

        # User input should almost always earn *some* answer path.
        pressure += 0.35
        reasons.append("user_input")

        if cues.get("is_question"):
            pressure += 0.30
            reasons.append("question")
        if cues.get("is_greeting"):
            pressure += 0.22
            reasons.append("greeting")
        if cues.get("direct_address"):
            pressure += 0.18
            reasons.append("direct_address")
        if cues.get("well_wish"):
            pressure += 0.18
            reasons.append("well_wish")
        if cues.get("needs_social_reply"):
            pressure += 0.22
            reasons.append("social_bid")

        if associations and top_assoc_score >= 0.25:
            pressure += min(0.10, 0.03 * len(associations))
            reasons.append("strong_associations")

        if boredom_level > 0.0:
            pressure += min(0.10, boredom_level * 0.10)
            reasons.append("boredom")

        if allow_babble:
            pressure += 0.05
            reasons.append("allow_babble")

        if crisis_mode:
            pressure += 0.25
            reasons.append("crisis_mode")

        # Socially addressed user input should be easier to release than idle babble.
        threshold = 0.55
        if cues.get("is_question") or cues.get("is_greeting") or cues.get("direct_address"):
            threshold = 0.45
        if crisis_mode:
            threshold = 0.35

        trigger = {
            "pressure": pressure,
            "threshold": threshold,
            "reasons": reasons,
            "should_release": pressure >= threshold,
            "kind": self._classify_trigger(cues=cues, crisis_mode=crisis_mode),
        }

        await ctx.set_kv("desire:last_trigger", trigger)

        self.debug(
            "release_check",
            pressure=round(pressure, 3),
            threshold=round(threshold, 3),
            release=trigger["should_release"],
            reasons=reasons,
            kind=trigger["kind"],
            assoc_top=round(top_assoc_score, 3),
        )

        if not trigger["should_release"]:
            return []

        return [
            Event(
                topic="release/request",
                payload={
                    "context": context,
                    "trigger": trigger,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"contextual": True, "stage": "triggered", "kind": trigger["kind"]},
            )
        ]

    def _classify_trigger(self, cues: Dict[str, Any], crisis_mode: bool) -> str:
        if crisis_mode:
            return "crisis"
        if cues.get("is_question"):
            return "question"
        if cues.get("is_greeting"):
            return "greeting"
        if cues.get("well_wish") or cues.get("needs_social_reply"):
            return "social"
        return "contextual"



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["context/built"],
        output_topics=["release/request"],
        priority=12,
    )
    yield DesireTriggerNeuron(cfg)
