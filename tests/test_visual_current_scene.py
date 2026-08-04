from __future__ import annotations

import asyncio
import inspect

from microbrain.neurons.proto_object_tracker_neuron import ProtoObjectTrackerNeuron
from microbrain.neurons.visual_current_scene_neuron import VisualCurrentSceneNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig
from microbrain.vision_state import bbox_xywh, normalize_visual_object


class FakeCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass


def make_neuron() -> VisualCurrentSceneNeuron:
    return VisualCurrentSceneNeuron(
        NeuronConfig(
            name="visual_current_scene_neuron",
            subscribed_topics=["percept/vision/features", "vision/proto_object", "vision/percept_commit", "vision/object_delta"],
            output_topics=["vision/current_objects"],
        )
    )


def test_visual_current_scene_is_ram_state_and_ui_instrument() -> None:
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        event = Event(
            topic="percept/vision/features",
            payload={
                "data_ref": "frame_001.jpg",
                "objects": [
                    {"id": "chair-1", "label": "chair", "confidence": 0.93, "bbox": [10, 20, 100, 160]},
                    {"id": "mug-1", "label": "mug", "confidence": 0.61, "bbox": [200, 80, 40, 60]},
                ],
            },
            timestamp=100.0,
        )
        outputs = list(await neuron.process(event, ctx))
        assert outputs == []
        assert ctx.kv["visual:current"]["object_count"] == 2
        assert "visual:exp" not in ctx.kv

    asyncio.run(run())


def test_visual_current_scene_updates_bbox_without_bus_flood() -> None:
    async def run():
        ctx = FakeCtx()
        ctx.kv["vision:current:publish_interval_s"] = 0.10
        neuron = make_neuron()
        first = Event(
            topic="percept/vision/features",
            payload={"objects": [{"id": "mug-1", "label": "mug", "confidence": 0.8, "bbox": [10, 10, 20, 20]}]},
            timestamp=100.0,
        )
        second = Event(
            topic="percept/vision/features",
            payload={"objects": [{"id": "mug-1", "label": "mug", "confidence": 0.8, "bbox": [12, 10, 20, 20]}]},
            timestamp=100.05,
        )
        assert list(await neuron.process(first, ctx)) == []
        assert list(await neuron.process(second, ctx)) == []
        assert ctx.kv["visual:current"]["objects"][0]["bbox"] == [12, 10, 20, 20]

    asyncio.run(run())


def test_missing_delta_removes_current_object() -> None:
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        await neuron.process(
            Event(
                topic="vision/percept_commit",
                payload={"proto_id": "vobj:1", "resolved_label": "cup", "max_stability": 0.9, "crop_box": [1, 2, 3, 4]},
                timestamp=100.0,
            ),
            ctx,
        )
        assert ctx.kv["visual:current"]["object_count"] == 1
        await neuron.process(
            Event(
                topic="vision/object_delta",
                payload={
                    "deltas": [
                        {
                            "change_type": "object_missing",
                            "object_key": "vobj:1",
                            "previous": {"object_key": "vobj:1", "label": "cup", "confidence": 0.9, "bbox": [1, 2, 3, 4]},
                        }
                    ]
                },
                timestamp=100.2,
            ),
            ctx,
        )
        assert ctx.kv["visual:current"]["object_count"] == 0

    asyncio.run(run())


def test_proto_tracker_no_longer_rewrites_disk_track_snapshot_each_frame() -> None:
    source = inspect.getsource(ProtoObjectTrackerNeuron)
    assert "vision_proto_tracks.json" not in source
    assert "write_text(" not in source
    assert not hasattr(ProtoObjectTrackerNeuron, "_write_state")


def test_bbox_normalizer_supports_crop_box_mapping() -> None:
    bbox = {"left": 10, "top": 20, "right": 110, "bottom": 220}
    assert bbox_xywh(bbox, source_width=640, source_height=480) == (10.0, 20.0, 100.0, 200.0)
    obj = normalize_visual_object({"proto_id": "abc", "resolved_label": "chair", "stability": 0.82, "crop_box": bbox})
    assert obj["track_id"] == "abc"
    assert obj["label"] == "chair"
    assert obj["confidence"] == 0.82


def test_ram_frame_ring_is_bounded_and_non_durable_by_default() -> None:
    from microbrain.utils.mb_vision.ram_frames import get_ram_frame, store_ram_frame

    async def run():
        ctx = FakeCtx()
        ctx.kv["vision:ram_frames_keep"] = 2
        ctx.kv["vision:ram_frame_ttl_s"] = 10.0
        refs = []
        for idx in range(3):
            refs.append(
                await store_ram_frame(
                    ctx,
                    sensor="camera",
                    frame_id=idx + 1,
                    timestamp=100.0 + idx,
                    jpeg_bytes=f"jpeg-{idx}".encode(),
                    width=320,
                    height=240,
                )
            )
        assert len(ctx.kv["vision:frame:ring"]) == 2
        assert await get_ram_frame(ctx, refs[0]) is None
        assert (await get_ram_frame(ctx, refs[-1]))["jpeg_bytes"] == b"jpeg-2"

    asyncio.run(run())


def test_capture_neurons_default_to_ram_storage() -> None:
    from microbrain.neurons import camera_capture_neuron, vision_window_capture_neuron

    camera_source = inspect.getsource(camera_capture_neuron.CameraCaptureNeuron)
    window_source = inspect.getsource(vision_window_capture_neuron.VisionWindowCaptureNeuron)
    assert 'ctx.get_kv("camera:save_mode", "ram")' in camera_source
    assert 'ctx.get_kv("vision:save_mode", "ram")' in window_source
    assert "store_ram_frame(" in camera_source
    assert "store_ram_frame(" in window_source
