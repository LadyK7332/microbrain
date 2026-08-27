from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.vision_mask_correction import (
    VISION_MASK_CORRECTION_SCHEMA,
    build_brush_tool_state,
    correction_from_label_map,
)

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

MASK_CORRECTION_STORE_LIMIT = 128

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

BRUSH_INPUT_TOPICS = ("vision/mask_brush_input", "vision/object_mask_brush")
MASK_CORRECTION_TOPIC = "vision/object_mask_correction"
BRUSH_TOOL_TOPIC = "vision/brush_tool_state"


class VisionMaskCorrectionNeuron(BaseNeuron):
    """Turn trainer brush strokes into compact pixel-mask correction events."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in BRUSH_INPUT_TOPICS:
            return []
        if not bool(await ctx.get_kv("vision:mask_correction:enabled", True)):
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        target = str(payload.get("target_track_id") or payload.get("target") or "").strip()
        if not target:
            return [self._rejected(event, "missing_target_track_id")]

        scene = await ctx.get_kv("vision:pixel_ownership:last", {})
        if not isinstance(scene, Mapping) or not scene.get("objects"):
            scene = await ctx.get_kv("scene:vision:pixel_ownership", {})
        if not isinstance(scene, Mapping) or not scene.get("objects"):
            return [self._rejected(event, "missing_pixel_ownership_scene")]

        label_packet = await ctx.get_kv("vision:pixel_ownership:label_map", {})
        if isinstance(label_packet, Mapping):
            label_map = label_packet.get("label_map")
        else:
            label_map = label_packet
        if label_map is None:
            return [self._rejected(event, "missing_label_map")]

        strokes = list(payload.get("strokes") or payload.get("brush_strokes") or [])
        if not strokes:
            return [self._rejected(event, "missing_brush_strokes")]

        operation = str(payload.get("operation") or payload.get("mode") or "subtract")
        reason = str(payload.get("reason") or "blob_too_large")
        trainer_id = str(payload.get("trainer_id") or payload.get("trainer") or "trainer")
        confidence = payload.get("confidence", 0.72)

        try:
            correction = correction_from_label_map(
                scene=scene,
                label_map=label_map,
                target_track_id=target,
                strokes=strokes,
                operation=operation,
                reason=reason,
                trainer_id=trainer_id,
                timestamp=event.timestamp,
                confidence=confidence,
            )
        except Exception as exc:
            return [self._rejected(event, f"correction_failed:{type(exc).__name__}:{exc}")]

        await ctx.set_kv("vision:mask_correction:last", correction)
        await self._append_correction(ctx, correction)

        tool_state = build_brush_tool_state(
            target_track_id=target,
            source_width=int(scene.get("source_width") or 1),
            source_height=int(scene.get("source_height") or 1),
            bbox_xywh=correction.get("target_bbox_xywh") or correction.get("delta", {}).get("after_bbox_xywh"),
            zoom=payload.get("zoom", 2.0),
            mode=operation,
            radius_px=payload.get("radius_px", payload.get("radius", 5)),
        )
        await ctx.set_kv("vision:brush_tool_state:last", tool_state)

        return [
            Event(
                topic=MASK_CORRECTION_TOPIC,
                payload=correction,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "vision_object_mask_correction",
                    "schema": VISION_MASK_CORRECTION_SCHEMA,
                    "store_in_memory": False,
                    "reinforcement_eligible": True,
                    "trainer_corrected": True,
                    "ui_instrument": True,
                },
            ),
            Event(
                topic=BRUSH_TOOL_TOPIC,
                payload=tool_state,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "vision_brush_tool_state",
                    "schema": tool_state.get("schema"),
                    "store_in_memory": False,
                    "ui_instrument": True,
                },
            ),
        ]

    async def _append_correction(self, ctx, correction: Mapping[str, Any]) -> None:
        limit = int(await ctx.get_kv("vision:mask_correction:history_limit", MASK_CORRECTION_STORE_LIMIT) or MASK_CORRECTION_STORE_LIMIT)
        prior = list(await ctx.get_kv("vision:mask_correction:recent", []) or [])
        prior.append(dict(correction))
        if len(prior) > max(1, limit):
            prior = prior[-max(1, limit) :]
        await ctx.set_kv("vision:mask_correction:recent", prior)

    def _rejected(self, event: Event, reason: str) -> Event:
        return Event(
            topic="vision/object_mask_correction_rejected",
            payload={
                "schema": "vision.object_mask_correction_rejected.v1",
                "reason": str(reason),
                "source_topic": event.topic,
                "memory_policy": "diagnostic only; no visual frame stored",
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "vision_mask_correction_rejected",
                "store_in_memory": False,
                "ui_instrument": True,
            },
        )


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield VisionMaskCorrectionNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=list(BRUSH_INPUT_TOPICS),
            output_topics=[MASK_CORRECTION_TOPIC, BRUSH_TOOL_TOPIC, "vision/object_mask_correction_rejected"],
            priority=3,
        )
    )
