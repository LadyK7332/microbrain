from __future__ import annotations

import asyncio

import numpy as np

from microbrain.neurons.motion_object_isolation_neuron import MotionObjectIsolationNeuron
from microbrain.neurons.visual_current_scene_neuron import VisualCurrentSceneNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig
from microbrain.utils.mb_vision.ram_frames import encode_jpeg_bytes, store_ram_frame


class FakeCtx:
    def __init__(self):
        self.kv = {"vision:enabled": True}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass


def _frame(x: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(20, 40, size=(180, 240, 1), dtype=np.uint8)
    img = np.repeat(base, 3, axis=2)
    # Textured moving patch: overlap still changes because the texture translates.
    for yy in range(60, 108):
        for xx in range(x, x + 52):
            value = 220 if ((xx - x) // 5 + (yy - 60) // 5) % 2 == 0 else 80
            img[yy, xx] = (value, value, value)
    return img


def test_motion_comparison_promotes_a_persistent_object_track() -> None:
    async def run():
        ctx = FakeCtx()
        ctx.kv["vision:isolation:max_hz"] = 20.0
        ctx.kv["vision:isolation:pixel_threshold"] = 12
        ctx.kv["vision:isolation:min_area_frac"] = 0.0005
        neuron = MotionObjectIsolationNeuron(
            NeuronConfig(
                name="motion_object_isolation_neuron",
                subscribed_topics=["percept/vision"],
                output_topics=["vision/object_isolation", "vision/motion_attention", "curiosity/adjust"],
            )
        )

        ids = []
        for idx, x in enumerate((45, 53, 61, 69), start=1):
            frame = _frame(x)
            ref = await store_ram_frame(
                ctx,
                sensor="camera",
                frame_id=idx,
                timestamp=100.0 + idx * 0.15,
                jpeg_bytes=encode_jpeg_bytes(frame, quality=90),
                width=frame.shape[1],
                height=frame.shape[0],
            )
            out = list(
                await neuron.process(
                    Event(
                        topic="percept/vision",
                        payload={"data_ref": ref, "frame_ref": ref, "ts": 100.0 + idx * 0.15},
                        timestamp=100.0 + idx * 0.15,
                    ),
                    ctx,
                )
            )
            isolation = [evt for evt in out if evt.topic == "vision/object_isolation"]
            if isolation and isolation[-1].payload.get("objects"):
                ids.append(isolation[-1].payload["objects"][0]["track_id"])

        assert ids, "coherent motion should become an isolated visual object"
        assert len(set(ids[-2:])) == 1, "the same moving region should refresh one ID"
        assert len(ctx.kv["vision:isolation:objects"]) == 1, "coherent leading/trailing motion should collapse into one object"
        obj = ctx.kv["vision:isolation:objects"][0]
        assert obj["status"] in {"isolated", "lost"}
        assert obj["snippet_ref"].startswith("ram:vision:object:")
        assert obj["objecthood_evidence"] == ["coherent_motion", "frame_delta", "spatial_persistence"]
        assert obj["contour"], "motion isolation should expose a direct border/polygon when available"

    asyncio.run(run())


def test_current_scene_coalesces_near_identical_proto_boxes() -> None:
    async def run():
        ctx = FakeCtx()
        neuron = VisualCurrentSceneNeuron(
            NeuronConfig(
                name="visual_current_scene_neuron",
                subscribed_topics=["vision/proto_object"],
                output_topics=[],
            )
        )
        for idx in range(4):
            await neuron.process(
                Event(
                    topic="vision/proto_object",
                    payload={
                        "proto_id": f"vobj:duplicate:{idx}",
                        "fallback_ref": "that unknown thing",
                        "stability": 0.18 + idx * 0.01,
                        "crop_box": {
                            "left": 80 + idx * 0.2,
                            "top": 50,
                            "right": 150 + idx * 0.2,
                            "bottom": 120,
                            "width": 70,
                            "height": 70,
                        },
                    },
                    timestamp=200.0 + idx * 0.1,
                ),
                ctx,
            )
        assert ctx.kv["visual:current"]["object_count"] == 1
        assert len(ctx.kv["visual:current"]["objects"][0].get("alias_track_ids", [])) >= 1

    asyncio.run(run())
