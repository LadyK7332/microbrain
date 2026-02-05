from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import time

# NOTE: Keep imports local-friendly. These libs are already used by window_picker_preview.py
# but we still guard them to avoid hard-crashes if someone runs headless / missing deps.


@dataclass
class WindowInfo:
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def rect(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def _require_deps():
    try:
        import pygetwindow as gw  # noqa: F401
        import mss  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Vision deps missing. Install: mss numpy pygetwindow (and opencv-python for preview)."
        ) from e


def list_windows() -> list[WindowInfo]:
    _require_deps()
    import pygetwindow as gw

    wins: list[WindowInfo] = []
    for w in gw.getAllWindows():
        try:
            title = (w.title or "").strip()
            if not title:
                continue
            if w.width <= 0 or w.height <= 0:
                continue
            # Some windows return negative coords briefly; allow but clamp later on grab.
            wins.append(WindowInfo(title=title, left=int(w.left), top=int(w.top), width=int(w.width), height=int(w.height)))
        except Exception:
            continue
    return wins


def pick_window(windows: list[WindowInfo], selector: str) -> WindowInfo | None:
    """
    selector:
      - digit => index in list_windows()
      - otherwise => case-insensitive substring match on title (first match)
    """
    sel = selector.strip()
    if not sel:
        return None
    if sel.isdigit():
        idx = int(sel)
        if 0 <= idx < len(windows):
            return windows[idx]
        return None

    low = sel.lower()
    for w in windows:
        if low in w.title.lower():
            return w
    return None


def grab_bgr(rect: dict[str, int]):
    """
    Returns a numpy array (H,W,3) in BGR order.
    """
    _require_deps()
    import mss
    import numpy as np

    # Clamp width/height to sane values
    mon = {
        "left": int(rect.get("left", 0)),
        "top": int(rect.get("top", 0)),
        "width": max(1, int(rect.get("width", 1))),
        "height": max(1, int(rect.get("height", 1))),
    }

    with mss.mss() as sct:
        img = np.array(sct.grab(mon))  # BGRA
        frame = img[:, :, :3]          # BGR
        return frame


def save_jpeg(frame_bgr, out_path: str, quality: int = 85) -> None:
    """
    Save BGR frame to JPEG. Uses OpenCV if present; falls back to PIL.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import cv2
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok = cv2.imwrite(out_path, frame_bgr, params)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
        return
    except Exception:
        # Fallback to PIL
        from PIL import Image
        import numpy as np

        rgb = frame_bgr[:, :, ::-1]  # BGR->RGB
        im = Image.fromarray(rgb.astype("uint8"))
        im.save(out_path, format="JPEG", quality=int(quality), optimize=True)


def draw_focus_reticle(frame_bgr, focus_xy: dict[str, Any] | None) -> None:
    """
    Draw a clean circle reticle in-place on a BGR frame.
    focus_xy expects normalized coords {x:0..1, y:0..1}
    """
    try:
        import cv2
    except Exception:
        return  # preview-only feature

    h, w = frame_bgr.shape[:2]
    fx = 0.5
    fy = 0.5
    if isinstance(focus_xy, dict):
        try:
            fx = float(focus_xy.get("x", 0.5))
            fy = float(focus_xy.get("y", 0.5))
        except Exception:
            fx, fy = 0.5, 0.5

    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    cx = int(fx * w)
    cy = int(fy * h)

    r = int(min(w, h) * 0.08)
    th = max(1, int(min(w, h) * 0.004))

    # outer ring (white)
    cv2.circle(frame_bgr, (cx, cy), r, (255, 255, 255), th, lineType=cv2.LINE_AA)
    # inner dot
    cv2.circle(frame_bgr, (cx, cy), max(2, th), (255, 255, 255), -1, lineType=cv2.LINE_AA)
