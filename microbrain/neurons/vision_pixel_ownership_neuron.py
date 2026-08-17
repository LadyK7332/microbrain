from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.mb_vision.ram_frames import get_ram_frame
from microbrain.vision_pixel_ownership import PIXEL_OWNERSHIP_SCHEMA, build_pixel_ownership_scene

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

PIXEL_OWNERSHIP_MAX_HZ = 6.0
PIXEL_OWNERSHIP_MAX_EXTRACT_PX = 96
PIXEL_OWNERSHIP_MAX_OBJECTS = 24
PIXEL_OWNERSHIP_MAX_ARTIFACTS = 48

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

PIXEL_OWNERSHIP_TOPIC = "vision/pixel_ownership"


class VisionPixelOwnershipNeuron(BaseNeuron):
    """Project isolated monocular objects into a pixel-ownership scene map.

    Motion/object isolation answers "what changed together?".  This organ asks
    a narrower follow-up question for a single camera:

        "Which pixels currently belong to each vobj, and what small extraction
        can represent that object without saving the whole frame?"

    It creates RAM-only label maps and extracted object crops/masks.  Durable
    fossil promotion remains a separate decision by the fossil/memory organs.
    """

    def __init__(self, config: NeuronConfig) -> None:
        super().__init__(config)
        self._last_analysis_ts = 0.0

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "vision/object_isolation":
            return []
        if not bool(await ctx.get_kv("vision:pixel_ownership:enabled", True)):
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        objects = list(payload.get("objects") or [])
        if not objects:
            return []

        frame_ref = str(payload.get("frame_ref") or payload.get("source_ref") or "").strip()
        if not frame_ref:
            return []

        now = float(payload.get("ts", event.timestamp) or time.time())
        max_hz = float(await ctx.get_kv("vision:pixel_ownership:max_hz", PIXEL_OWNERSHIP_MAX_HZ) or PIXEL_OWNERSHIP_MAX_HZ)
        if max_hz > 0 and self._last_analysis_ts > 0 and (now - self._last_analysis_ts) < (1.0 / max_hz):
            return []

        decoded = await self._decode_frame(ctx, frame_ref)
        if decoded is None:
            return []
        frame_bgr, _source_w, _source_h = decoded

        max_extract_px = int(await ctx.get_kv("vision:pixel_ownership:max_extract_px", PIXEL_OWNERSHIP_MAX_EXTRACT_PX) or PIXEL_OWNERSHIP_MAX_EXTRACT_PX)
        max_objects = int(await ctx.get_kv("vision:pixel_ownership:max_objects", PIXEL_OWNERSHIP_MAX_OBJECTS) or PIXEL_OWNERSHIP_MAX_OBJECTS)
        accelerator_requested = str(await ctx.get_kv("vision:pixel_ownership:accelerator", "cpu") or "cpu").strip().lower()

        scene, artifacts, label_map = build_pixel_ownership_scene(
            frame_bgr,
            objects,
            frame_ref=frame_ref,
            timestamp=now,
            max_extract_px=max(24, min(256, max_extract_px)),
            max_objects=max(1, min(64, max_objects)),
        )
        if not scene.get("objects"):
            return []

        scene["accelerator"] = {
            "requested": accelerator_requested,
            "effective": "cpu",
            "note": "v1 keeps CPU default; GPU/OpenCL/CUDA path may be added behind this budget key later",
        }
        scene["source_event_schema"] = str(payload.get("schema") or "")

        self._last_analysis_ts = now
        await ctx.set_kv("vision:pixel_ownership:last", scene)
        await ctx.set_kv("scene:vision:pixel_ownership", scene)
        await ctx.set_kv("vision:pixel_ownership:label_map", {"ref": scene["label_map_ref"], "ts": now, "label_map": label_map})
        await self._merge_artifacts(ctx, artifacts)

        return [
            Event(
                topic=PIXEL_OWNERSHIP_TOPIC,
                payload=scene,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "vision_pixel_ownership",
                    "schema": PIXEL_OWNERSHIP_SCHEMA,
                    "store_in_memory": False,
                    "cognitive_visible": False,
                    "reinforcement_eligible": False,
                    "ui_instrument": True,
                },
            )
        ]

    async def _merge_artifacts(self, ctx, artifacts: Mapping[str, Dict[str, Any]]) -> None:
        if not artifacts:
            return
        limit = int(await ctx.get_kv("vision:pixel_ownership:max_artifacts", PIXEL_OWNERSHIP_MAX_ARTIFACTS) or PIXEL_OWNERSHIP_MAX_ARTIFACTS)
        prior = dict(await ctx.get_kv("vision:pixel_ownership:extracts", {}) or {})
        prior.update(dict(artifacts))
        if len(prior) > max(1, limit):
            rows = sorted(prior.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0.0) or 0.0), reverse=True)
            prior = dict(rows[: max(1, limit)])
        await ctx.set_kv("vision:pixel_ownership:extracts", prior)

    async def _decode_frame(self, ctx, frame_ref: str) -> Optional[Tuple[Any, int, int]]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return None

        if frame_ref.startswith("ram:vision:"):
            packet = await get_ram_frame(ctx, frame_ref)
            if not isinstance(packet, Mapping):
                return None
            data = packet.get("jpeg_bytes")
            if not isinstance(data, (bytes, bytearray)):
                return None
            frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return None
            h, w = frame.shape[:2]
            return frame, int(w), int(h)

        try:
            frame = cv2.imread(frame_ref, cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return frame, int(w), int(h)


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield VisionPixelOwnershipNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=["vision/object_isolation"],
            output_topics=[PIXEL_OWNERSHIP_TOPIC],
            priority=3,
        )
    )
