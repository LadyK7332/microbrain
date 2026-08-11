from __future__ import annotations

import hashlib
import json
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
VISUAL_FROZEN_EVIDENCE_SCHEMA = "vision.frozen_evidence_ref.v1"


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

        snapshot = payload.get("object_snapshot") if isinstance(payload.get("object_snapshot"), Mapping) else None
        selected = dict(snapshot) if isinstance(snapshot, Mapping) else None
        visual = await ctx.get_kv("visual:current", {})
        objects = visual.get("objects", []) if isinstance(visual, Mapping) else []
        if selected is None:
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
        ui_snapshot = selected.get("ui_snapshot") if isinstance(selected.get("ui_snapshot"), Mapping) else {}
        source_ref = str(selected.get("source_ref") or selected.get("frame_ref") or "")
        if not source_ref and isinstance(visual, Mapping):
            source_ref = str(visual.get("frame_ref") or visual.get("source_ref") or "")
        frozen = bool(payload.get("frozen", False) or selected.get("ui_frozen", False) or (ui_snapshot or {}).get("frozen", False))
        evidence_basis = {
            "track_id": str(selected.get("track_id") or track_id),
            "selected_track_id": track_id,
            "bbox": bbox,
            "contour": selected.get("contour"),
            "snippet_ref": str(selected.get("snippet_ref") or ""),
            "source_ref": source_ref,
            "frame_label": str((ui_snapshot or {}).get("frame_label") or ""),
            "selected_at": round(now, 3),
        }
        evidence_digest = hashlib.blake2b(
            json.dumps(evidence_basis, sort_keys=True, default=str).encode("utf-8", errors="ignore"),
            digest_size=8,
        ).hexdigest()
        visual_evidence_ref = f"vision:evidence:{evidence_digest}"
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
            "source_ref": source_ref,
            "frame_label": str((ui_snapshot or {}).get("frame_label") or ""),
            "frame_size": {
                "width": int((ui_snapshot or {}).get("source_width", 0) or 0),
                "height": int((ui_snapshot or {}).get("source_height", 0) or 0),
            },
            "frozen": frozen,
            "visual_evidence_ref": visual_evidence_ref,
            "visual_evidence_schema": VISUAL_FROZEN_EVIDENCE_SCHEMA if frozen else VISUAL_ATTENTION_SCHEMA,
            "object_snapshot": selected if frozen else {},
            "selected_at": now,
            "expires_at": now + max(2.0, ttl_s),
            "remaining_turns": max(1, turns),
            "salience": max(0.0, min(1.0, salience)),
            "source": "user_ui_selection",
            "semantics": "attention_only_not_identity_assertion",
            "teaching_use": "selected_frozen_visual_evidence" if frozen else "selected_live_visual_attention",
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
