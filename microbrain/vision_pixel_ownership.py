"""Pixel ownership and object extraction helpers for monocular vision.

This module is intentionally monocular.  It does not infer stereo depth or a
full 3D object.  It turns current visual objects into per-object pixel masks,
small extracted object artifacts, contour/spline-ish shape summaries, and a
scene-map projection that other organs can inspect.

Full frames remain sensation.  Durable memory should only receive promoted
extractions/fossils from another organ after stability, salience, or user
labeling warrants it.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

PIXEL_OWNERSHIP_SCHEMA = "vision.pixel_ownership.v1"
VISION_EXTRACTION_SCHEMA = "vision.extraction.v1"
MASK_RLE_SCHEMA = "vision.mask.rle_bool.v1"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _clamp_int(value: Any, low: int, high: int) -> int:
    return max(low, min(high, int(round(_float(value, low)))))


def _track_id(obj: Mapping[str, Any], fallback_index: int = 0) -> str:
    for key in ("track_id", "object_id", "proto_id", "id", "object_key"):
        value = str(obj.get(key) or "").strip()
        if value:
            return value
    basis = repr({"bbox": obj.get("bbox") or obj.get("box") or obj.get("crop_box"), "index": fallback_index})
    digest = hashlib.blake2b(basis.encode("utf-8", errors="ignore"), digest_size=5).hexdigest()
    return f"vobj:auto:{digest}"


def _bbox_xywh(bbox: Any, *, source_width: int = 0, source_height: int = 0) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(bbox, Mapping):
        if all(k in bbox for k in ("left", "top", "right", "bottom")):
            x = _float(bbox.get("left"))
            y = _float(bbox.get("top"))
            w = _float(bbox.get("right")) - x
            h = _float(bbox.get("bottom")) - y
        elif all(k in bbox for k in ("x", "y", "w", "h")):
            x = _float(bbox.get("x"))
            y = _float(bbox.get("y"))
            w = _float(bbox.get("w"))
            h = _float(bbox.get("h"))
        elif all(k in bbox for k in ("left", "top", "width", "height")):
            x = _float(bbox.get("left"))
            y = _float(bbox.get("top"))
            w = _float(bbox.get("width"))
            h = _float(bbox.get("height"))
        else:
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, w, h = (_float(v) for v in bbox[:4])
    else:
        return None

    if source_width > 0 and source_height > 0 and max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x *= float(source_width)
        w *= float(source_width)
        y *= float(source_height)
        h *= float(source_height)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _safe_bbox_int(bbox: Any, *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    xywh = _bbox_xywh(bbox, source_width=width, source_height=height)
    if xywh is None:
        return None
    x, y, w, h = xywh
    x0 = _clamp_int(x, 0, max(0, width - 1))
    y0 = _clamp_int(y, 0, max(0, height - 1))
    x1 = _clamp_int(x + w, x0 + 1, width)
    y1 = _clamp_int(y + h, y0 + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _points_from_contour(contour: Any, *, width: int, height: int, limit: int = 256) -> List[List[float]]:
    points: List[List[float]] = []
    if not isinstance(contour, (list, tuple)):
        return points
    for point in contour[:limit]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = max(0.0, min(float(width - 1), _float(point[0])))
        y = max(0.0, min(float(height - 1), _float(point[1])))
        points.append([x, y])
    return points


# ---------------------------------------------------------------------------
# Mask encoding
# ---------------------------------------------------------------------------


def encode_binary_mask_rle(mask: Any) -> Dict[str, Any]:
    """Encode a boolean/0-1 mask using simple row-major run-length encoding."""

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
    for value in flat[1:]:
        v = int(value)
        if v == last:
            run += 1
            continue
        runs.append(run)
        last = v
        run = 1
    runs.append(run)
    return {"schema": MASK_RLE_SCHEMA, "shape": [int(arr.shape[0]), int(arr.shape[1])], "start": start, "runs": runs}


def decode_binary_mask_rle(packet: Mapping[str, Any]) -> Any:
    """Decode the RLE produced by :func:`encode_binary_mask_rle`."""

    import numpy as np

    shape = list(packet.get("shape") or [])
    if len(shape) != 2:
        raise ValueError("mask RLE packet needs shape [h, w]")
    h, w = int(shape[0]), int(shape[1])
    total = h * w
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


# ---------------------------------------------------------------------------
# Pixel ownership / extraction
# ---------------------------------------------------------------------------


def object_mask_from_contour_or_bbox(obj: Mapping[str, Any], *, source_width: int, source_height: int) -> Any:
    """Return a full-frame binary ownership mask for a visual object.

    Preferred ownership source is an object contour.  BBox is a fallback so the
    scene can still produce a pointable region while upstream segmentation is
    learning.  The output is RAM/runtime material, not durable memory.
    """

    import cv2
    import numpy as np

    mask = np.zeros((int(source_height), int(source_width)), dtype=np.uint8)
    contour = _points_from_contour(obj.get("contour") or obj.get("polygon") or obj.get("border"), width=source_width, height=source_height)
    if len(contour) >= 3:
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    safe = _safe_bbox_int(obj.get("bbox") or obj.get("box") or obj.get("crop_box"), width=source_width, height=source_height)
    if safe is not None:
        x, y, w, h = safe
        mask[y : y + h, x : x + w] = 1
    return mask.astype(bool)


def mask_bbox_xywh(mask: Any) -> Optional[List[int]]:
    """Return tight bbox [x, y, w, h] around non-zero mask pixels."""

    import numpy as np

    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2 or not bool(arr.any()):
        return None
    ys, xs = np.where(arr)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def simplify_contour_spline(contour: Any, *, source_width: int, source_height: int, max_points: int = 24) -> List[List[float]]:
    """Simplify a contour into a small spline-ish outline for scene/fossil use."""

    points = _points_from_contour(contour, width=source_width, height=source_height, limit=512)
    if len(points) <= max(3, max_points):
        return [[round(x, 2), round(y, 2)] for x, y in points]

    try:
        import cv2
        import numpy as np

        arr = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        perimeter = float(cv2.arcLength(arr, True) or 0.0)
        epsilon = max(0.75, 0.012 * perimeter)
        approx = cv2.approxPolyDP(arr, epsilon, True)
        out = [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx[: max_points * 2]]
        if len(out) > max_points:
            step = max(1, int(math.ceil(len(out) / max_points)))
            out = out[::step][:max_points]
        return out
    except Exception:
        step = max(1, int(math.ceil(len(points) / max_points)))
        return [[round(float(x), 2), round(float(y), 2)] for x, y in points[::step][:max_points]]


def _contour_from_mask(mask: Any, *, offset_x: int = 0, offset_y: int = 0, max_points: int = 24) -> List[List[float]]:
    try:
        import cv2
        import numpy as np

        arr = np.asarray(mask).astype("uint8")
        contours, _hier = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True) or 0.0)
        approx = cv2.approxPolyDP(contour, max(0.75, 0.018 * perimeter), True)
        out = []
        for point in approx[:max_points]:
            out.append([round(float(point[0][0] + offset_x), 2), round(float(point[0][1] + offset_y), 2)])
        return out
    except Exception:
        return []


def _dhash_gray(gray: Any) -> str:
    try:
        import cv2
        import numpy as np

        arr = np.asarray(gray)
        if arr.size == 0:
            return ""
        small = cv2.resize(arr, (9, 8), interpolation=cv2.INTER_AREA)
        bits = small[:, 1:] > small[:, :-1]
        value = 0
        for flag in bits.flatten():
            value = (value << 1) | int(bool(flag))
        return f"{value:016x}"
    except Exception:
        return ""


def _dominant_color_hex(frame_bgr: Any, mask: Any) -> str:
    try:
        import numpy as np

        pixels = frame_bgr[np.asarray(mask).astype(bool)]
        if pixels.size == 0:
            return ""
        # Median is stable under speckle and cheap enough for tiny crops.
        b, g, r = np.median(pixels.reshape((-1, 3)), axis=0)
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    except Exception:
        return ""


def extract_object_from_mask(
    frame_bgr: Any,
    mask: Any,
    obj: Mapping[str, Any],
    *,
    frame_ref: str,
    fallback_index: int = 0,
    timestamp: Optional[float] = None,
    max_extract_px: int = 96,
) -> Optional[Dict[str, Any]]:
    """Extract only the owned pixels for an object and return a RAM artifact.

    The returned artifact contains compressed bytes and structured metadata.  It
    is intentionally suitable for RAM/KV transport.  Another organ should decide
    whether to write a durable fossil to disk.
    """

    import cv2
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    bbox = mask_bbox_xywh(arr)
    if bbox is None:
        return None
    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    crop = frame_bgr[y : y + h, x : x + w]
    crop_mask = arr[y : y + h, x : x + w].astype("uint8")
    if crop.size == 0 or crop_mask.size == 0 or not bool(crop_mask.any()):
        return None

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask * 255
    ok_rgba, encoded_rgba = cv2.imencode(".png", rgba)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    scale = min(1.0, float(max_extract_px) / max(1.0, float(max(ch, cw))))
    if scale < 1.0:
        gray_small = cv2.resize(gray, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(crop_mask, (gray_small.shape[1], gray_small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        gray_small = gray
        mask_small = crop_mask
    # Keep grayscale fossil background neutral so shape matching is mostly object-owned pixels.
    gray_owned = np.where(mask_small.astype(bool), gray_small, 0).astype("uint8")
    ok_gray, encoded_gray = cv2.imencode(".png", gray_owned)

    track_id = _track_id(obj, fallback_index=fallback_index)
    now = float(timestamp if timestamp is not None else time.time())
    digest_basis = f"{track_id}|{frame_ref}|{bbox}|{int(crop_mask.sum())}|{now:.3f}"
    digest = hashlib.blake2b(digest_basis.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()
    extraction_ref = f"ram:vision:extract:{track_id}:{digest}"
    mask_ref = f"{extraction_ref}:mask"
    gray_ref = f"{extraction_ref}:gray"
    rgba_ref = f"{extraction_ref}:rgba"
    contour = simplify_contour_spline(obj.get("contour") or obj.get("polygon") or obj.get("border"), source_width=frame_w, source_height=frame_h)
    if not contour:
        contour = _contour_from_mask(crop_mask, offset_x=x, offset_y=y)

    ys, xs = np.where(arr)
    centroid = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)] if len(xs) else [float(x + w / 2), float(y + h / 2)]
    source_pixels = max(1, int(frame_w * frame_h))
    owned_pixels = int(crop_mask.sum())
    crop_pixels = max(1, int(crop_mask.size))

    return {
        "schema": VISION_EXTRACTION_SCHEMA,
        "extraction_ref": extraction_ref,
        "track_id": track_id,
        "label": str(obj.get("label") or "unknown"),
        "source_frame_ref": str(frame_ref or ""),
        "ts": now,
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "centroid_xy_px": centroid,
        "source_frame_shape": [int(frame_h), int(frame_w)],
        "owned_pixel_count": owned_pixels,
        "crop_pixel_count": crop_pixels,
        "source_frame_pixel_count": source_pixels,
        "mask_coverage_frame_frac": round(owned_pixels / source_pixels, 8),
        "crop_coverage_frame_frac": round(crop_pixels / source_pixels, 8),
        "fill_ratio": round(owned_pixels / crop_pixels, 6),
        "mask_ref": mask_ref,
        "mask_rle": encode_binary_mask_rle(crop_mask),
        "rgba_ref": rgba_ref,
        "rgba_png_bytes": bytes(encoded_rgba.tobytes()) if ok_rgba else b"",
        "gray_ref": gray_ref,
        "gray_png_bytes": bytes(encoded_gray.tobytes()) if ok_gray else b"",
        "gray_dhash": _dhash_gray(gray_owned),
        "dominant_color_hex": _dominant_color_hex(crop, crop_mask),
        "contour_spline": contour,
        "storage_policy": "ram_extraction_only; durable fossil requires later promotion",
        "diskspace_policy": "save object-owned crop/mask, not whole source frame",
    }


def build_pixel_ownership_scene(
    frame_bgr: Any,
    objects: Iterable[Mapping[str, Any]],
    *,
    frame_ref: str = "",
    timestamp: Optional[float] = None,
    max_extract_px: int = 96,
    max_objects: int = 24,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Any]:
    """Build a monocular pixel-ownership scene map and RAM extractions.

    Returns ``(scene, artifacts, label_map)``.  ``label_map`` is a RAM-only
    integer map where 0 is background and each positive integer maps to one
    scene object entry.  It should not be written to durable memory by default.
    """

    import numpy as np

    now = float(timestamp if timestamp is not None else time.time())
    frame_h, frame_w = frame_bgr.shape[:2]
    label_map = np.zeros((int(frame_h), int(frame_w)), dtype=np.int32)
    scene_objects: List[Dict[str, Any]] = []
    artifacts: Dict[str, Dict[str, Any]] = {}

    for idx, obj in enumerate(list(objects)[:max(1, int(max_objects))], start=1):
        mask = object_mask_from_contour_or_bbox(obj, source_width=int(frame_w), source_height=int(frame_h))
        if not bool(mask.any()):
            continue
        # Later objects do not erase earlier owned pixels.  This keeps stable
        # higher-confidence tracks from being overwritten by overlapping noise.
        free_mask = mask & (label_map == 0)
        if not bool(free_mask.any()):
            continue
        label_map[free_mask] = idx
        extraction = extract_object_from_mask(
            frame_bgr,
            free_mask,
            obj,
            frame_ref=frame_ref,
            fallback_index=idx,
            timestamp=now,
            max_extract_px=max_extract_px,
        )
        if extraction is None:
            continue
        artifacts[extraction["extraction_ref"]] = extraction
        track_id = extraction["track_id"]
        scene_objects.append(
            {
                "track_id": track_id,
                "label_id": idx,
                "label": extraction.get("label", "unknown"),
                "bbox_xywh": list(extraction["bbox_xywh"]),
                "centroid_xy_px": list(extraction["centroid_xy_px"]),
                "mask_ref": extraction["mask_ref"],
                "extraction_ref": extraction["extraction_ref"],
                "gray_ref": extraction["gray_ref"],
                "rgba_ref": extraction["rgba_ref"],
                "contour_spline": list(extraction.get("contour_spline") or []),
                "dominant_color_hex": str(extraction.get("dominant_color_hex") or ""),
                "owned_pixel_count": int(extraction.get("owned_pixel_count", 0) or 0),
                "fill_ratio": float(extraction.get("fill_ratio", 0.0) or 0.0),
                "source_frame_ref": str(frame_ref or ""),
                "confidence": float(obj.get("confidence", obj.get("isolation_confidence", 0.0)) or 0.0),
                "status": str(obj.get("status") or "isolated"),
                "motion_state": str(obj.get("motion_state") or ""),
                "objecthood_evidence": list(obj.get("objecthood_evidence") or []),
                "scene_role": "current_pixel_owned_object",
            }
        )

    scene = {
        "schema": PIXEL_OWNERSHIP_SCHEMA,
        "ts": now,
        "frame_ref": str(frame_ref or ""),
        "source_width": int(frame_w),
        "source_height": int(frame_h),
        "label_map_ref": f"ram:vision:label_map:{hashlib.blake2b((str(frame_ref) + str(now)).encode('utf-8', errors='ignore'), digest_size=6).hexdigest()}",
        "label_map_shape": [int(frame_h), int(frame_w)],
        "label_map_policy": "ram_only_int32; not durable memory",
        "object_count": len(scene_objects),
        "objects": scene_objects,
        "storage_policy": "current_scene_projection_plus_ram_object_extractions",
        "monocular_policy": "single_camera_visible_surface_only; no stereo depth claim",
    }
    return scene, artifacts, label_map
