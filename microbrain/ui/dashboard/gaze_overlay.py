"""Presentation-only conversion of normalized gaze to a dashboard overlay.

This module has no Qt dependency and never modifies the gaze controller.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def gaze_overlay_geometry(
    state: Mapping[str, Any] | None,
    *,
    image_left: float,
    image_top: float,
    image_width: float,
    image_height: float,
) -> dict[str, Any] | None:
    """Convert an actual gaze-state reading to pixels within the letterboxed image.

    None means no usable gaze state. Never invent a center point when state is
    absent, stale or malformed.
    """
    if not isinstance(state, Mapping) or "x" not in state or "y" not in state:
        return None
    try:
        x = float(state["x"])
        y = float(state["y"])
        radius = float(state.get("radius", 0.08))
        width = float(image_width)
        height = float(image_height)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x, y, radius, width, height)) or width <= 0 or height <= 0:
        return None

    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    radius = min(0.35, max(0.03, radius))
    mode = str(state.get("mode") or "unknown").lower()
    track_id = str(state.get("target_motion_id") or state.get("target_proto_id") or "")
    return {
        "cx": image_left + x * width,
        "cy": image_top + y * height,
        "aperture_radius": radius * min(width, height),
        "mode": mode,
        "track_id": track_id[:32],
    }
