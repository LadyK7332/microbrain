"""Shared helpers for MicroBrain's ephemeral visual scene state.

Frames are sensory samples.  Current objects are RAM-resident perceptual state.
Durable memory is fed by meaningful vision deltas elsewhere, not by this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def visual_object_id(item: Mapping[str, Any], *, fallback_index: int = 0) -> str:
    """Return the best available stable-ish identifier for a current visual object."""

    for key in ("track_id", "object_id", "proto_id", "id", "object_key"):
        value = str(item.get(key) or "").strip()
        if value:
            return value

    label = str(
        item.get("label")
        or item.get("resolved_label")
        or item.get("name")
        or item.get("class")
        or item.get("type")
        or "object"
    ).strip().lower()
    bbox = item.get("bbox") or item.get("box") or item.get("crop_box") or item.get("focus_xy")
    basis = json.dumps({"label": label, "bbox": bbox, "index": int(fallback_index)}, sort_keys=True, default=str)
    digest = hashlib.blake2b(basis.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()
    return f"visual:{label}:{digest}"


def normalize_visual_object(
    item: Mapping[str, Any],
    *,
    fallback_index: int = 0,
    source: str = "",
    timestamp: float = 0.0,
) -> dict[str, Any]:
    """Normalize detector/proto/delta shapes into the dashboard/current-scene schema."""

    label = str(
        item.get("label")
        or item.get("resolved_label")
        or item.get("fallback_ref")
        or item.get("name")
        or item.get("class")
        or item.get("type")
        or "unknown"
    ).strip()
    track_id = visual_object_id(item, fallback_index=fallback_index)
    confidence = _float(
        item.get("confidence", item.get("conf", item.get("stability", item.get("max_stability", 0.0)))),
        0.0,
    )
    status = str(item.get("status") or ("identified" if label.lower() not in {"", "unknown", "thing", "that thing"} else "unknown")).strip().lower()
    bbox = item.get("bbox") or item.get("box") or item.get("crop_box")
    if isinstance(bbox, Mapping):
        bbox = dict(bbox)
    elif isinstance(bbox, (list, tuple)):
        bbox = list(bbox[:4])
    else:
        bbox = None

    hazard = bool(item.get("hazard") or item.get("emergency") or status in {"hazard", "danger", "emergency"})
    motion = item.get("motion") or item.get("velocity") or item.get("delta_xy")
    position = item.get("position") or item.get("xyz")
    contour = item.get("contour") or item.get("border") or item.get("polygon")
    if isinstance(contour, (list, tuple)):
        contour = [list(point[:2]) for point in contour if isinstance(point, (list, tuple)) and len(point) >= 2]
    else:
        contour = None

    return {
        "track_id": track_id,
        "label": label or "unknown",
        "confidence": max(0.0, min(1.0, confidence)),
        "status": status or "unknown",
        "bbox": bbox,
        "motion": motion,
        "motion_state": str(item.get("motion_state") or ""),
        "position": position,
        "contour": contour,
        "snippet_ref": str(item.get("snippet_ref") or ""),
        "isolation_confidence": max(0.0, min(1.0, _float(item.get("isolation_confidence", confidence), confidence))),
        "objecthood_evidence": list(item.get("objecthood_evidence") or []),
        "hazard": hazard,
        "source": str(source or item.get("source") or ""),
        "source_ref": str(item.get("source_ref") or item.get("frame_ref") or ""),
        "first_seen": _float(item.get("first_seen", timestamp), timestamp),
        "last_seen": _float(item.get("last_seen", timestamp), timestamp),
        "seen_count": int(item.get("seen_count", item.get("count", 1)) or 1),
    }


def bbox_xywh(bbox: Any, *, source_width: int = 0, source_height: int = 0) -> tuple[float, float, float, float] | None:
    """Normalize common bbox shapes to x/y/width/height in source pixels."""

    if isinstance(bbox, Mapping):
        if all(key in bbox for key in ("left", "top", "right", "bottom")):
            x = _float(bbox.get("left"))
            y = _float(bbox.get("top"))
            w = _float(bbox.get("right")) - x
            h = _float(bbox.get("bottom")) - y
        elif all(key in bbox for key in ("x", "y", "w", "h")):
            x, y, w, h = (_float(bbox.get(key)) for key in ("x", "y", "w", "h"))
        elif all(key in bbox for key in ("left", "top", "width", "height")):
            x, y, w, h = (_float(bbox.get(key)) for key in ("left", "top", "width", "height"))
        else:
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, w, h = (_float(value) for value in bbox[:4])
    else:
        return None

    if source_width > 0 and source_height > 0 and max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x *= source_width
        w *= source_width
        y *= source_height
        h *= source_height

    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def visual_semantic_fingerprint(objects: list[Mapping[str, Any]]) -> str:
    """Fingerprint identity/classification state, intentionally ignoring position jitter."""

    rows = [
        {
            "track_id": str(obj.get("track_id") or ""),
            "label": str(obj.get("label") or ""),
            "status": str(obj.get("status") or ""),
            "hazard": bool(obj.get("hazard", False)),
            "confidence_band": round(_float(obj.get("confidence"), 0.0), 1),
        }
        for obj in objects
    ]
    rows.sort(key=lambda row: (row["track_id"], row["label"]))
    return hashlib.blake2b(json.dumps(rows, sort_keys=True).encode("utf-8"), digest_size=8).hexdigest()
