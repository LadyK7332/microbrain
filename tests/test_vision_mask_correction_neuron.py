from __future__ import annotations

import asyncio

import numpy as np

from microbrain.neurons.vision_mask_correction_neuron import VisionMaskCorrectionNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value


def test_neuron_emits_correction_and_tool_state() -> None:
    async def run():
        label_map = np.zeros((40, 50), dtype=np.int32)
        label_map[8:30, 10:40] = 3
        ctx = FakeCtx()
        ctx.kv["vision:pixel_ownership:last"] = {
            "schema": "vision.pixel_ownership.v1",
            "frame_ref": "frame:test",
            "source_width": 50,
            "source_height": 40,
            "objects": [{"track_id": "vobj:blob", "label_id": 3, "bbox_xywh": [10, 8, 30, 22]}],
        }
        ctx.kv["vision:pixel_ownership:label_map"] = {"label_map": label_map}
        neuron = VisionMaskCorrectionNeuron(
            NeuronConfig(
                name="vision_mask_correction_neuron",
                subscribed_topics=["vision/mask_brush_input"],
                output_topics=["vision/object_mask_correction", "vision/brush_tool_state"],
            )
        )
        out = list(
            await neuron.process(
                Event(
                    topic="vision/mask_brush_input",
                    payload={
                        "target_track_id": "vobj:blob",
                        "mode": "subtract",
                        "reason": "blob_too_large",
                        "strokes": [{"radius_px": 4, "points": [[36, 24], [37, 25]]}],
                    },
                    source="ui",
                    timestamp=123.0,
                ),
                ctx,
            )
        )
        assert [event.topic for event in out] == ["vision/object_mask_correction", "vision/brush_tool_state"]
        assert out[0].payload["delta"]["removed_pixel_count"] > 0
        assert ctx.kv["vision:mask_correction:last"]["target"] == "vobj:blob"
        assert ctx.kv["vision:brush_tool_state:last"]["target_track_id"] == "vobj:blob"

    asyncio.run(run())


def test_neuron_rejects_without_label_map() -> None:
    async def run():
        ctx = FakeCtx()
        ctx.kv["vision:pixel_ownership:last"] = {
            "objects": [{"track_id": "vobj:blob", "label_id": 1, "bbox_xywh": [0, 0, 5, 5]}],
            "source_width": 10,
            "source_height": 10,
        }
        neuron = VisionMaskCorrectionNeuron(
            NeuronConfig(
                name="vision_mask_correction_neuron",
                subscribed_topics=["vision/mask_brush_input"],
                output_topics=["vision/object_mask_correction"],
            )
        )
        out = list(
            await neuron.process(
                Event(
                    topic="vision/mask_brush_input",
                    payload={"target_track_id": "vobj:blob", "strokes": [{"points": [[1, 1]]}]},
                    source="ui",
                ),
                ctx,
            )
        )
        assert len(out) == 1
        assert out[0].topic == "vision/object_mask_correction_rejected"
        assert out[0].payload["reason"] == "missing_label_map"

    asyncio.run(run())
