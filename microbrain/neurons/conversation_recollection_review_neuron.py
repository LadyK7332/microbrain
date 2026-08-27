from __future__ import annotations

"""Review sidecar for live conversation and random recollection.

The neuron subscribes to user reasoning requests, outgoing speech, and several
low-trust internal thought/recollection topics.  It stores review anchors in KV
and emits compact review events for dashboards or later memory organs.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.conversation_recollection_review import analyze_anchor, review_pair, trim_ring
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

RECOLLECTION_TOPICS = {
    "thought/probe",
    "thought/internal",
    "thought/recollection",
    "memory/recollection",
    "memory/random_recollection",
    "recollection/random",
}


class ConversationRecollectionReviewNeuron(BaseNeuron):
    """Digest conversation/recollection fragments without controlling speech."""

    async def _get_ring(self, ctx, key: str) -> list[dict[str, Any]]:
        value = await ctx.get_kv(key, [])
        if isinstance(value, list):
            return trim_ring(value, limit=64)
        return []

    async def _append_ring(self, ctx, key: str, item: Mapping[str, Any], *, limit: int = 64) -> None:
        ring = await self._get_ring(ctx, key)
        ring.append(dict(item))
        await ctx.set_kv(key, trim_ring(ring, limit=limit))

    def _payload_text(self, event: Event) -> str:
        payload = event.payload
        if isinstance(payload, Mapping):
            for key in ("text", "message", "pending_text", "thought", "content", "reply"):
                value = payload.get(key, "")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        return str(payload or "").strip()

    def _event_source(self, event: Event, fallback: str) -> str:
        if isinstance(event.payload, Mapping):
            src = str(event.payload.get("source", "") or "").strip()
            if src:
                return src
        return str(event.source or fallback)

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        enabled = bool(await ctx.get_kv("conversation_review:enabled", True))
        if not enabled:
            return []

        topic = str(event.topic or "")
        text = self._payload_text(event)
        if not text:
            return []

        outputs: list[Event] = []
        now = float(event.timestamp or time.time())

        if topic == "reason/request":
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            channel = str(payload.get("channel", "") or "")
            source = self._event_source(event, "user")
            if channel in {"internal", "thought"}:
                return []
            anchor = analyze_anchor(
                text=text,
                source=source or "user",
                role="user",
                previous_user_text="",
                event_topic=topic,
                correlation_id=event.correlation_id,
                ts=now,
            )
            await ctx.set_kv("conversation_review:last_user_anchor", anchor)
            await self._append_ring(ctx, "conversation_review:anchors", anchor, limit=96)
            outputs.append(Event(
                topic="review/utterance_anchor",
                payload=anchor,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "conversation_review_anchor",
                    "store_in_memory": False,
                    "semantic_input": False,
                    "reinforcement_eligible": False,
                },
            ))
            return outputs

        if topic == "act/speech":
            source = self._event_source(event, "assistant")
            last_user = await ctx.get_kv("conversation_review:last_user_anchor", {}) or {}
            previous_user_text = str(last_user.get("text", "") or "") if isinstance(last_user, Mapping) else ""
            anchor = analyze_anchor(
                text=text,
                source=source or "assistant",
                role="assistant",
                previous_user_text=previous_user_text,
                event_topic=topic,
                correlation_id=event.correlation_id,
                ts=now,
            )
            await ctx.set_kv("conversation_review:last_assistant_anchor", anchor)
            await self._append_ring(ctx, "conversation_review:anchors", anchor, limit=96)
            outputs.append(Event(
                topic="review/utterance_anchor",
                payload=anchor,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "conversation_review_anchor",
                    "store_in_memory": False,
                    "semantic_input": False,
                    "reinforcement_eligible": False,
                },
            ))
            review = review_pair(last_user if isinstance(last_user, Mapping) else {}, anchor)
            await ctx.set_kv("conversation_review:last_turn_review", review)
            await self._append_ring(ctx, "conversation_review:turn_reviews", review, limit=96)
            outputs.append(Event(
                topic="review/conversation_turn",
                payload=review,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "conversation_turn_review",
                    "store_in_memory": False,
                    "semantic_input": False,
                    "reinforcement_eligible": False,
                },
            ))
            if not review.get("satisfied_turn", False):
                outputs.append(Event(
                    topic="review/repair_candidate",
                    payload=review,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "conversation_repair_candidate",
                        "store_in_memory": False,
                        "semantic_input": False,
                        "reinforcement_eligible": False,
                    },
                ))
            return outputs

        if topic in RECOLLECTION_TOPICS:
            source = self._event_source(event, "recollection")
            last_user = await ctx.get_kv("conversation_review:last_user_anchor", {}) or {}
            previous_user_text = str(last_user.get("text", "") or "") if isinstance(last_user, Mapping) else ""
            anchor = analyze_anchor(
                text=text,
                source=source or "thought_probe",
                role="recollection",
                previous_user_text=previous_user_text,
                event_topic=topic,
                correlation_id=event.correlation_id,
                ts=now,
            )
            anchor["memory_eligible"] = False
            anchor["promotion_requires"] = ["daylight_review", "trainer_confirmation", "repeat_useful_pattern"]
            await ctx.set_kv("conversation_review:last_recollection_anchor", anchor)
            await self._append_ring(ctx, "conversation_review:recollection_reviews", anchor, limit=96)
            outputs.append(Event(
                topic="review/recollection_anchor",
                payload=anchor,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "recollection_review_anchor",
                    "store_in_memory": False,
                    "semantic_input": False,
                    "reinforcement_eligible": False,
                },
            ))
            return outputs

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "reason/request",
            "act/speech",
            "thought/probe",
            "thought/internal",
            "thought/recollection",
            "memory/recollection",
            "memory/random_recollection",
            "recollection/random",
        ],
        output_topics=["review/utterance_anchor", "review/conversation_turn", "review/repair_candidate", "review/recollection_anchor"],
        priority=2,
        cooldown_sec=0.0,
    )
    yield ConversationRecollectionReviewNeuron(cfg)
