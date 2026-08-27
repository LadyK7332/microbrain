from __future__ import annotations

import numpy as np

from microbrain.vision_mask_correction import (
    VISION_MASK_CORRECTION_SCHEMA,
    apply_brush_strokes,
    build_brush_tool_state,
    build_object_mask_correction,
    build_zoom_region,
    correction_from_label_map,
)


def _mask() -> np.ndarray:
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:40, 20:60] = True
    return mask


def test_zoom_region_clamps_to_object_area() -> None:
    region = build_zoom_region(source_width=80, source_height=60, bbox_xywh=[20, 10, 40, 30], zoom=2.0)
    assert region["schema"] == "vision.zoom_region.v1"
    x, y, w, h = region["viewport_xywh"]
    assert 0 <= x < 80
    assert 0 <= y < 60
    assert w <= 80 and h <= 60
    assert region["zoom"] == 2.0


def test_brush_subtract_carves_blob_without_frame_storage() -> None:
    base = _mask()
    result = apply_brush_strokes(
        base,
        [{"mode": "subtract", "radius_px": 4, "points": [[55, 35], [56, 36]]}],
    )
    assert result["changed"] is True
    assert result["after_mask"].sum() < base.sum()
    assert result["removed_mask"].sum() > 0


def test_build_object_mask_correction_keeps_delta_not_whole_painting() -> None:
    packet = build_object_mask_correction(
        target_track_id="vobj:big_blob",
        base_mask=_mask(),
        strokes=[{"mode": "subtract", "radius_px": 5, "points": [[52, 30]]}],
        reason="blob_too_large",
        timestamp=123.0,
    )
    assert packet["schema"] == VISION_MASK_CORRECTION_SCHEMA
    assert packet["trainer_corrected"] is True
    assert packet["delta"]["removed_pixel_count"] > 0
    assert "after_mask_rle" in packet["delta"]
    assert "full_frame" not in repr(packet).lower()
    assert "not the whole painting" in packet["delta"]["law"]


def test_correction_from_label_map_targets_one_object() -> None:
    scene = {
        "frame_ref": "frame:test",
        "source_width": 80,
        "source_height": 60,
        "objects": [{"track_id": "vobj:one", "label_id": 7, "bbox_xywh": [20, 10, 40, 30]}],
    }
    label_map = np.zeros((60, 80), dtype=np.int32)
    label_map[10:40, 20:60] = 7
    correction = correction_from_label_map(
        scene=scene,
        label_map=label_map,
        target_track_id="vobj:one",
        strokes=[{"mode": "subtract", "radius_px": 5, "points": [[55, 30]]}],
        timestamp=222.0,
    )
    assert correction["target_label_id"] == 7
    assert correction["source_frame_ref"] == "frame:test"
    assert correction["delta"]["removed_pixel_count"] > 0
    assert correction["zoom_region"]["schema"] == "vision.zoom_region.v1"


def test_brush_tool_state_has_runtime_memory_policy() -> None:
    state = build_brush_tool_state(
        target_track_id="vobj:one",
        source_width=80,
        source_height=60,
        bbox_xywh=[20, 10, 40, 30],
        zoom=4,
        mode="subtract",
        radius_px=3,
    )
    assert state["schema"] == "vision.brush_tool_state.v1"
    assert state["mode"] == "subtract"
    assert state["radius_px"] == 3
    assert "not durable" in state["memory_policy"]
