"""Trainer brush corrections for pixel ownership masks.

This module is deliberately about *correction deltas*, not durable frame
storage.  MB may guess a blob.  A trainer may carve the object.  Memory keeps
only the correction/evidence delta unless another organ explicitly promotes a
visual artifact.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Pixel-brush defaults are UI/runtime tunables.  They deliberately stay small
# because this is meant to carve mistaken object masks, not repaint a frame.
DEFAULT_BRUSH_RADIUS_PX = 5
MIN_BRUSH_RADIUS_PX = 1
MAX_BRUSH_RADIUS_PX = 48
DEFAULT_ZOOM = 2.0
MIN_ZOOM = 1.0
MAX_ZOOM = 12.0
DEFAULT_VIEWPORT_PADDING_PX = 18
MAX_STROKE_POINTS = 512
MAX_STROKES = 64

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

VISION_BRUSH_TOOL_SCHEMA = "vision.brush_tool_state.v1"
VISION_MASK_CORRECTION_SCHEMA = "vision.object_mask_correction.v1"
VISION_MASK_DELTA_SCHEMA = "vision.mask_delta.v1"
MASK_RLE_SCHEMA = "vision.mask.rle_bool.v1"
BRUSH_INPUT_SCHEMA = "vision.mask_brush_input.v1"

VALID_BRUSH_MODES = {"add", "subtract", "split", "uncertain"}
VALID_BRUSH_REASONS = {
    "blob_too_large",
    "missing_object_pixels",
    "split_object_from_blob",
    "uncertain_boundary",
    "trainer_correction",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_float(value, default)))
    except Exception:
        return int(default)


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    out = _float(value, default)
    return max(float(low), min(float(high), out))


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    return int(round(_clamp(value, low, high, default)))


def _now(ts: Optional[float] = None) -> float:
    if ts is None:
        return float(time.time())
    out = _float(ts, time.time())
    return out if out > 0 else float(time.time())


def _safe_id(value: Any, fallback: str = "object") -> str:
    text = str(value or "").strip() or fallback
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", ":", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120] or fallback


# ---------------------------------------------------------------------------
# RLE helpers kept local so the brush tool works even before old pixel module
# is present; if both exist the packets remain schema-compatible.
# ---------------------------------------------------------------------------


def encode_binary_mask_rle(mask: Any) -> Dict[str, Any]:
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise ValueError("encode_binary_mask_rle expects a 2D mask")
    flat = arr.astype("uint8", copy=False).ravel()
    if flat.size == 0:
        return {"schema": MASK_RLE_SCHEMA, "shape": [int(arr.shape[0]), int(arr.shape[1])], "start": 0, "runs": []}
    start = int(flat[0])
    last = start
    run = 1
    runs: List[int] = []
    for raw in flat[1:]:
        value = int(raw)
        if value == last:
            run += 1
            continue
        runs.append(run)
        last = value
        run = 1
    runs.append(run)
    return {"schema": MASK_RLE_SCHEMA, "shape": [int(arr.shape[0]), int(arr.shape[1])], "start": start, "runs": runs}


def decode_binary_mask_rle(packet: Mapping[str, Any]) -> Any:
    import numpy as np

    shape = list(packet.get("shape") or [])
    if len(shape) != 2:
        raise ValueError("mask RLE packet needs shape [h, w]")
    h, w = int(shape[0]), int(shape[1])
    total = max(0, h * w)
    value = int(packet.get("start", 0) or 0)
    out = np.zeros((total,), dtype=np.uint8)
    pos = 0
    for run in list(packet.get("runs") or []):
        length = max(0, int(run))
        if length and value:
            out[pos : min(total, pos + length)] = 1
        pos += length
        value = 1 - value
        if pos >= total:
            break
    return out.reshape((h, w)).astype(bool)


def _mask_digest(mask: Any) -> str:
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    h = hashlib.blake2b(digest_size=10)
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.astype("uint8", copy=False).tobytes())
    return "blake2:" + h.hexdigest()


def _bbox(mask: Any) -> List[int]:
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2 or not bool(arr.any()):
        return [0, 0, 0, 0]
    ys, xs = np.where(arr)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def _normalize_point(point: Any, *, width: int, height: int) -> Optional[Tuple[int, int]]:
    if isinstance(point, Mapping):
        x = point.get("x", point.get("px", point.get("col")))
        y = point.get("y", point.get("py", point.get("row")))
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        x, y = point[0], point[1]
    else:
        return None
    xi = _clamp_int(x, 0, max(0, width - 1), 0)
    yi = _clamp_int(y, 0, max(0, height - 1), 0)
    return xi, yi


def _normalize_stroke(stroke: Mapping[str, Any], *, width: int, height: int, default_mode: str = "subtract") -> Dict[str, Any]:
    mode = str(stroke.get("mode") or stroke.get("operation") or default_mode or "subtract").strip().lower()
    if mode not in VALID_BRUSH_MODES:
        mode = "subtract"
    radius = _clamp_int(stroke.get("radius_px", stroke.get("radius", DEFAULT_BRUSH_RADIUS_PX)), MIN_BRUSH_RADIUS_PX, MAX_BRUSH_RADIUS_PX, DEFAULT_BRUSH_RADIUS_PX)
    raw_points = stroke.get("points") or stroke.get("path") or []
    points: List[List[int]] = []
    for raw in list(raw_points)[:MAX_STROKE_POINTS]:
        point = _normalize_point(raw, width=width, height=height)
        if point is not None:
            points.append([int(point[0]), int(point[1])])
    return {
        "mode": mode,
        "radius_px": int(radius),
        "points": points,
        "point_count": len(points),
    }


# ---------------------------------------------------------------------------
# Zoom/brush UI state helpers
# ---------------------------------------------------------------------------


def build_zoom_region(
    *,
    source_width: int,
    source_height: int,
    bbox_xywh: Sequence[Any] | None = None,
    center_xy: Sequence[Any] | None = None,
    zoom: Any = DEFAULT_ZOOM,
    padding_px: Any = DEFAULT_VIEWPORT_PADDING_PX,
) -> Dict[str, Any]:
    """Return a bounded zoom viewport for a target object or point.

    The UI can use this packet to zoom +/- around a blob before the trainer
    applies a brush correction.  It is just UI guidance; it does not become
    durable memory.
    """

    width = max(1, int(source_width or 1))
    height = max(1, int(source_height or 1))
    zoom_value = round(_clamp(zoom, MIN_ZOOM, MAX_ZOOM, DEFAULT_ZOOM), 3)
    padding = _clamp_int(padding_px, 0, max(width, height), DEFAULT_VIEWPORT_PADDING_PX)

    if bbox_xywh and len(bbox_xywh) >= 4:
        x = _clamp_int(bbox_xywh[0], 0, width - 1, 0)
        y = _clamp_int(bbox_xywh[1], 0, height - 1, 0)
        w = _clamp_int(bbox_xywh[2], 1, width, 1)
        h = _clamp_int(bbox_xywh[3], 1, height, 1)
        cx = x + w / 2.0
        cy = y + h / 2.0
        view_w = max(1, min(width, int(round((w + padding * 2) / zoom_value))))
        view_h = max(1, min(height, int(round((h + padding * 2) / zoom_value))))
    elif center_xy and len(center_xy) >= 2:
        cx = _clamp(center_xy[0], 0, width - 1, width / 2.0)
        cy = _clamp(center_xy[1], 0, height - 1, height / 2.0)
        view_w = max(1, min(width, int(round(width / zoom_value))))
        view_h = max(1, min(height, int(round(height / zoom_value))))
    else:
        cx, cy = width / 2.0, height / 2.0
        view_w = max(1, min(width, int(round(width / zoom_value))))
        view_h = max(1, min(height, int(round(height / zoom_value))))

    x0 = max(0, min(width - view_w, int(round(cx - view_w / 2.0))))
    y0 = max(0, min(height - view_h, int(round(cy - view_h / 2.0))))
    return {
        "schema": "vision.zoom_region.v1",
        "source_shape": [int(height), int(width)],
        "viewport_xywh": [int(x0), int(y0), int(view_w), int(view_h)],
        "center_xy_px": [round(float(cx), 3), round(float(cy), 3)],
        "zoom": zoom_value,
        "padding_px": int(padding),
        "policy": "ui_runtime_only; zoom does not change object truth",
    }


def build_brush_tool_state(
    *,
    target_track_id: str,
    source_width: int,
    source_height: int,
    bbox_xywh: Sequence[Any] | None = None,
    zoom: Any = DEFAULT_ZOOM,
    mode: str = "subtract",
    radius_px: Any = DEFAULT_BRUSH_RADIUS_PX,
) -> Dict[str, Any]:
    mode_norm = str(mode or "subtract").strip().lower()
    if mode_norm not in VALID_BRUSH_MODES:
        mode_norm = "subtract"
    return {
        "schema": VISION_BRUSH_TOOL_SCHEMA,
        "target_track_id": str(target_track_id or ""),
        "mode": mode_norm,
        "radius_px": _clamp_int(radius_px, MIN_BRUSH_RADIUS_PX, MAX_BRUSH_RADIUS_PX, DEFAULT_BRUSH_RADIUS_PX),
        "zoom_region": build_zoom_region(
            source_width=int(source_width),
            source_height=int(source_height),
            bbox_xywh=bbox_xywh,
            zoom=zoom,
        ),
        "allowed_modes": sorted(VALID_BRUSH_MODES),
        "memory_policy": "brush state is runtime tool state, not durable memory",
    }


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------


def _paint_stroke(mask_shape: Tuple[int, int], stroke: Mapping[str, Any]) -> Any:
    import numpy as np

    h, w = int(mask_shape[0]), int(mask_shape[1])
    painted = np.zeros((h, w), dtype=bool)
    radius = max(MIN_BRUSH_RADIUS_PX, min(MAX_BRUSH_RADIUS_PX, int(stroke.get("radius_px", DEFAULT_BRUSH_RADIUS_PX) or DEFAULT_BRUSH_RADIUS_PX)))
    points = list(stroke.get("points") or [])[:MAX_STROKE_POINTS]
    if not points:
        return painted

    yy, xx = np.ogrid[:h, :w]
    last: Optional[Tuple[int, int]] = None
    for raw in points:
        point = _normalize_point(raw, width=w, height=h)
        if point is None:
            continue
        x, y = point
        painted |= ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2
        # Fill small gaps along a dragged path so fast brush movement does not
        # leave dotted corrections.  This is intentionally conservative.
        if last is not None:
            lx, ly = last
            dist = max(1, int(math.ceil(math.hypot(x - lx, y - ly))))
            steps = min(96, max(1, dist // max(1, radius // 2)))
            for step in range(1, steps + 1):
                t = step / float(steps + 1)
                ix = int(round(lx + (x - lx) * t))
                iy = int(round(ly + (y - ly) * t))
                painted |= ((xx - ix) ** 2 + (yy - iy) ** 2) <= radius**2
        last = (x, y)
    return painted


def apply_brush_strokes(base_mask: Any, strokes: Iterable[Mapping[str, Any]], *, default_mode: str = "subtract") -> Dict[str, Any]:
    """Apply trainer brush strokes to a full-frame boolean object mask.

    Returns a packet with before/after masks plus additive/subtractive/split/
    uncertain masks.  Callers should store the returned delta, not the whole
    source frame.
    """

    import numpy as np

    before = np.asarray(base_mask).astype(bool)
    if before.ndim != 2:
        raise ValueError("apply_brush_strokes expects a 2D mask")
    after = before.copy()
    added = np.zeros_like(before, dtype=bool)
    removed = np.zeros_like(before, dtype=bool)
    split_candidate = np.zeros_like(before, dtype=bool)
    uncertain = np.zeros_like(before, dtype=bool)
    h, w = before.shape[:2]

    normalized: List[Dict[str, Any]] = []
    for raw in list(strokes or [])[:MAX_STROKES]:
        if not isinstance(raw, Mapping):
            continue
        stroke = _normalize_stroke(raw, width=w, height=h, default_mode=default_mode)
        if not stroke["points"]:
            continue
        painted = _paint_stroke((h, w), stroke)
        mode = stroke["mode"]
        if mode == "add":
            add_mask = painted & ~after
            after |= painted
            added |= add_mask
        elif mode == "subtract":
            remove_mask = painted & after
            after &= ~painted
            removed |= remove_mask
        elif mode == "split":
            # Split means "this painted piece should not remain part of the
            # original blob".  The split candidate can become a new vobj later.
            piece = painted & after
            split_candidate |= piece
            after &= ~painted
            removed |= piece
        elif mode == "uncertain":
            uncertain |= painted
        normalized.append(stroke)

    return {
        "before_mask": before,
        "after_mask": after,
        "added_mask": added,
        "removed_mask": removed,
        "split_candidate_mask": split_candidate,
        "uncertain_mask": uncertain,
        "strokes": normalized,
        "changed": bool(np.any(before != after) or np.any(uncertain)),
    }


def build_mask_delta_packet(
    *,
    target_track_id: str,
    before_mask: Any,
    after_mask: Any,
    added_mask: Any,
    removed_mask: Any,
    split_candidate_mask: Any | None = None,
    uncertain_mask: Any | None = None,
    strokes: Sequence[Mapping[str, Any]] | None = None,
    operation: str = "subtract",
    reason: str = "blob_too_large",
    trainer_id: str = "trainer",
    timestamp: Optional[float] = None,
    source_frame_ref: str = "",
) -> Dict[str, Any]:
    import numpy as np

    before = np.asarray(before_mask).astype(bool)
    after = np.asarray(after_mask).astype(bool)
    added = np.asarray(added_mask).astype(bool)
    removed = np.asarray(removed_mask).astype(bool)
    split_candidate = np.asarray(split_candidate_mask).astype(bool) if split_candidate_mask is not None else np.zeros_like(before, dtype=bool)
    uncertain = np.asarray(uncertain_mask).astype(bool) if uncertain_mask is not None else np.zeros_like(before, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("before and after masks must have the same shape")

    target = _safe_id(target_track_id, "vobj")
    now = _now(timestamp)
    op = str(operation or "subtract").strip().lower()
    if op not in VALID_BRUSH_MODES:
        op = "subtract"
    why = str(reason or "trainer_correction").strip().lower()
    if why not in VALID_BRUSH_REASONS:
        why = "trainer_correction"
    basis = {
        "target": target,
        "before": _mask_digest(before),
        "after": _mask_digest(after),
        "added": int(added.sum()),
        "removed": int(removed.sum()),
        "ts": round(now, 3),
    }
    digest = hashlib.blake2b(repr(basis).encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    ref_base = f"evidence/vision/mask_delta/{target}/{digest}"
    return {
        "schema": VISION_MASK_DELTA_SCHEMA,
        "target_track_id": str(target_track_id or ""),
        "mask_delta_ref": f"{ref_base}.json",
        "source_frame_ref": str(source_frame_ref or ""),
        "operation": op,
        "reason": why,
        "trainer_id": str(trainer_id or "trainer"),
        "trainer_corrected": True,
        "ts": now,
        "shape": [int(before.shape[0]), int(before.shape[1])],
        "before_digest": _mask_digest(before),
        "after_digest": _mask_digest(after),
        "before_bbox_xywh": _bbox(before),
        "after_bbox_xywh": _bbox(after),
        "added_pixel_count": int(added.sum()),
        "removed_pixel_count": int(removed.sum()),
        "split_candidate_pixel_count": int(split_candidate.sum()),
        "uncertain_pixel_count": int(uncertain.sum()),
        "after_owned_pixel_count": int(after.sum()),
        "brush_stroke_count": len(list(strokes or [])),
        "brush_strokes_summary": [
            {
                "mode": str(s.get("mode") or op),
                "radius_px": int(s.get("radius_px", DEFAULT_BRUSH_RADIUS_PX) or DEFAULT_BRUSH_RADIUS_PX),
                "point_count": int(s.get("point_count", len(list(s.get("points") or []))) or 0),
            }
            for s in list(strokes or [])[:MAX_STROKES]
            if isinstance(s, Mapping)
        ],
        "added_mask_rle": encode_binary_mask_rle(added),
        "removed_mask_rle": encode_binary_mask_rle(removed),
        "split_candidate_mask_rle": encode_binary_mask_rle(split_candidate),
        "uncertain_mask_rle": encode_binary_mask_rle(uncertain),
        "after_mask_rle": encode_binary_mask_rle(after),
        "memory_policy": "store correction delta and compact after-mask; do not store full frame by default",
        "law": "MB may guess the blob. Trainer may carve the object. Memory keeps the correction, not the whole painting.",
    }


def build_object_mask_correction(
    *,
    target_track_id: str,
    base_mask: Any,
    strokes: Iterable[Mapping[str, Any]],
    operation: str = "subtract",
    reason: str = "blob_too_large",
    trainer_id: str = "trainer",
    timestamp: Optional[float] = None,
    source_frame_ref: str = "",
    confidence: Any = 0.72,
) -> Dict[str, Any]:
    """Create the compact correction event for a trainer-brushed object mask."""

    applied = apply_brush_strokes(base_mask, strokes, default_mode=operation)
    delta = build_mask_delta_packet(
        target_track_id=target_track_id,
        before_mask=applied["before_mask"],
        after_mask=applied["after_mask"],
        added_mask=applied["added_mask"],
        removed_mask=applied["removed_mask"],
        split_candidate_mask=applied["split_candidate_mask"],
        uncertain_mask=applied["uncertain_mask"],
        strokes=applied["strokes"],
        operation=operation,
        reason=reason,
        trainer_id=trainer_id,
        timestamp=timestamp,
        source_frame_ref=source_frame_ref,
    )
    return {
        "schema": VISION_MASK_CORRECTION_SCHEMA,
        "target": str(target_track_id or ""),
        "operation": str(delta["operation"]),
        "reason": str(delta["reason"]),
        "trainer_corrected": True,
        "confidence": round(_clamp(confidence, 0.0, 1.0, 0.72), 4),
        "mask_delta_ref": delta["mask_delta_ref"],
        "source_frame_ref": str(source_frame_ref or ""),
        "before_mask_ref": f"ram:vision:mask:{_safe_id(target_track_id, 'vobj')}:before",
        "after_mask_ref": f"ram:vision:mask:{_safe_id(target_track_id, 'vobj')}:after:{hashlib.blake2b(delta['after_digest'].encode('utf-8'), digest_size=5).hexdigest()}",
        "delta": delta,
        "changed": bool(applied["changed"]),
        "memory_policy": "correction/evidence event only; not a full-frame memory",
        "surface_for_ui": {
            "brush_zoom_supported": True,
            "paint_brush_supported": True,
            "zoom_region_suggested": True,
        },
    }


def correction_from_label_map(
    *,
    scene: Mapping[str, Any],
    label_map: Any,
    target_track_id: str,
    strokes: Iterable[Mapping[str, Any]],
    operation: str = "subtract",
    reason: str = "blob_too_large",
    trainer_id: str = "trainer",
    timestamp: Optional[float] = None,
    confidence: Any = 0.72,
) -> Dict[str, Any]:
    """Build a correction packet from a current pixel-ownership label map."""

    import numpy as np

    scene_objects = list(scene.get("objects") or [])
    target_obj: Optional[Mapping[str, Any]] = None
    for obj in scene_objects:
        if str(obj.get("track_id") or "") == str(target_track_id or ""):
            target_obj = obj
            break
    if target_obj is None:
        raise ValueError(f"target_track_id not found in pixel ownership scene: {target_track_id}")
    label_id = int(target_obj.get("label_id", 0) or 0)
    if label_id <= 0:
        raise ValueError("target object has no positive label_id")
    arr = np.asarray(label_map)
    if arr.ndim != 2:
        raise ValueError("label_map must be a 2D integer map")
    base_mask = arr == label_id
    if not bool(base_mask.any()):
        raise ValueError("target label_id does not appear in label_map")
    correction = build_object_mask_correction(
        target_track_id=target_track_id,
        base_mask=base_mask,
        strokes=strokes,
        operation=operation,
        reason=reason,
        trainer_id=trainer_id,
        timestamp=timestamp,
        source_frame_ref=str(scene.get("frame_ref") or target_obj.get("source_frame_ref") or ""),
        confidence=confidence,
    )
    correction["target_label_id"] = label_id
    correction["target_bbox_xywh"] = list(target_obj.get("bbox_xywh") or [])
    correction["zoom_region"] = build_zoom_region(
        source_width=int(scene.get("source_width") or arr.shape[1]),
        source_height=int(scene.get("source_height") or arr.shape[0]),
        bbox_xywh=target_obj.get("bbox_xywh") or correction["delta"].get("before_bbox_xywh"),
    )
    return correction
