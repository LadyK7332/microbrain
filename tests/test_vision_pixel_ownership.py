from __future__ import annotations

import asyncio

import cv2
import numpy as np

from microbrain.neurons.vision_pixel_ownership_neuron import VisionPixelOwnershipNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig
from microbrain.vision_pixel_ownership import (
    build_pixel_ownership_scene,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
    object_mask_from_contour_or_bbox,
)


class FakeCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass


def _scene_frame():
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    frame[:, :] = (18, 18, 18)
    cv2.rectangle(frame, (24, 18), (58, 50), (20, 190, 30), thickness=-1)
    cv2.circle(frame, (68, 28), 8, (220, 220, 220), thickness=-1)
    return frame


def test_rle_roundtrip_preserves_binary_mask() -> None:
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 5:11] = True
    packet = encode_binary_mask_rle(mask)
    decoded = decode_binary_mask_rle(packet)
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask)


def test_object_mask_uses_contour_before_bbox() -> None:
    obj = {
        "track_id": "vobj:triangle",
        "bbox": [10, 10, 60, 50],
        "contour": [[20, 20], [42, 20], [31, 45]],
    }
    mask = object_mask_from_contour_or_bbox(obj, source_width=80, source_height=60)
    assert mask.sum() > 0
    assert mask[22, 31]
    assert not mask[12, 12], "contour ownership should not fill the entire bbox when contour exists"


def test_build_pixel_ownership_scene_saves_extraction_not_full_frame() -> None:
    frame = _scene_frame()
    objects = [
        {
            "track_id": "vobj:green_patch",
            "label": "unknown",
            "bbox": [24, 18, 35, 33],
            "contour": [[24, 18], [58, 18], [58, 50], [24, 50]],
            "confidence": 0.72,
            "objecthood_evidence": ["coherent_motion"],
        }
    ]
    scene, artifacts, label_map = build_pixel_ownership_scene(frame, objects, frame_ref="frame:test", timestamp=123.0)
    assert scene["schema"] == "vision.pixel_ownership.v1"
    assert scene["object_count"] == 1
    obj = scene["objects"][0]
    assert obj["track_id"] == "vobj:green_patch"
    assert obj["extraction_ref"] in artifacts
    artifact = artifacts[obj["extraction_ref"]]
    assert artifact["source_frame_pixel_count"] == frame.shape[0] * frame.shape[1]
    assert artifact["crop_pixel_count"] < artifact["source_frame_pixel_count"]
    assert artifact["diskspace_policy"].startswith("save object-owned")
    assert artifact["mask_rle"]["shape"] == [artifact["bbox_xywh"][3], artifact["bbox_xywh"][2]]
    assert int(label_map.max()) == 1


def test_neuron_projects_object_isolation_into_scene_map(tmp_path) -> None:
    async def run():
        frame = _scene_frame()
        path = tmp_path / "frame.png"
        cv2.imwrite(str(path), frame)
        ctx = FakeCtx()
        neuron = VisionPixelOwnershipNeuron(
            NeuronConfig(
                name="vision_pixel_ownership_neuron",
                subscribed_topics=["vision/object_isolation"],
                output_topics=["vision/pixel_ownership"],
            )
        )
        out = list(
            await neuron.process(
                Event(
                    topic="vision/object_isolation",
                    payload={
                        "schema": "vision.object_isolation.v1",
                        "ts": 456.0,
                        "frame_ref": str(path),
                        "objects": [
                            {
                                "track_id": "vobj:one",
                                "bbox": [24, 18, 35, 33],
                                "contour": [[24, 18], [58, 18], [58, 50], [24, 50]],
                                "confidence": 0.8,
                            }
                        ],
                    },
                    timestamp=456.0,
                ),
                ctx,
            )
        )
        assert len(out) == 1
        assert out[0].topic == "vision/pixel_ownership"
        assert ctx.kv["vision:pixel_ownership:last"]["object_count"] == 1
        assert ctx.kv["scene:vision:pixel_ownership"]["objects"][0]["mask_ref"]
        assert ctx.kv["vision:pixel_ownership:extracts"], "object-owned extraction artifact should live in RAM KV"
        assert ctx.kv["vision:pixel_ownership:label_map"]["label_map"].shape == frame.shape[:2]

    asyncio.run(run())
