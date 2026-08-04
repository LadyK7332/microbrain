from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

VISUAL_ATTENTION_TTL_S = 20.0
VISUAL_ATTENTION_TURNS = 1
VISUAL_ATTENTION_SALIENCE = 0.78

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

VISUAL_ATTENTION_SCHEMA = "vision.attention_ref.v1"
VISUAL_ATTENTION_KV = "vision:attention_anchor"


class VisualAttentionAnchorNeuron(BaseNeuron):
    """Turn UI object selection into a short-lived attentional/deictic anchor.

    Selection is *not* an identity assertion. It only says: "this is the visual
    thing the user is pointing at right now." The next relevant user turn can
    carry this reference into cognition, after which it is normally consumed.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "control/vision_attention":
            return []
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        action = str(payload.get("action") or "select").strip().lower()
        if action in {"clear", "release", "deselect"}:
            await ctx.set_kv(VISUAL_ATTENTION_KV, None)
            return []

        track_id = str(payload.get("track_id") or payload.get("object_id") or "").strip()
        if not track_id:
            return []

        visual = await ctx.get_kv("visual:current", {})
        objects = visual.get("objects", []) if isinstance(visual, Mapping) else []
        selected = None
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, Mapping):
                continue
            aliases = [str(v) for v in list(obj.get("alias_track_ids") or [])]
            if str(obj.get("track_id") or "") == track_id or track_id in aliases:
                selected = dict(obj)
                break
        if selected is None:
            return []

        now = time.time()
        ttl_s = float(await ctx.get_kv("vision:attention:ttl_s", VISUAL_ATTENTION_TTL_S) or VISUAL_ATTENTION_TTL_S)
        turns = int(await ctx.get_kv("vision:attention:turns", VISUAL_ATTENTION_TURNS) or VISUAL_ATTENTION_TURNS)
        salience = float(await ctx.get_kv("vision:attention:salience", VISUAL_ATTENTION_SALIENCE) or VISUAL_ATTENTION_SALIENCE)
        bbox = selected.get("bbox")
        anchor = {
            "schema": VISUAL_ATTENTION_SCHEMA,
            "track_id": str(selected.get("track_id") or track_id),
            "selected_track_id": track_id,
            "label_hint": str(selected.get("label") or "unknown"),
            "status": str(selected.get("status") or "unknown"),
            "confidence": float(selected.get("confidence", 0.0) or 0.0),
            "bbox": bbox,
            "contour": selected.get("contour"),
            "snippet_ref": str(selected.get("snippet_ref") or ""),
            "source_ref": str(selected.get("source_ref") or ""),
            "selected_at": now,
            "expires_at": now + max(2.0, ttl_s),
            "remaining_turns": max(1, turns),
            "salience": max(0.0, min(1.0, salience)),
            "source": "user_ui_selection",
            "semantics": "attention_only_not_identity_assertion",
        }
        await ctx.set_kv(VISUAL_ATTENTION_KV, anchor)
        await ctx.set_kv("attention:visual_ref", anchor)

        return [
            Event(
                topic="vision/attention_anchor",
                payload=anchor,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "visual_attention_anchor",
                    "store_in_memory": False,
                    "cognitive_visible": False,
                    "reinforcement_eligible": False,
                    "user_pointing": True,
                },
            )
        ]


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield VisualAttentionAnchorNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=["control/vision_attention"],
            output_topics=["vision/attention_anchor"],
            priority=3,
        )
    )
