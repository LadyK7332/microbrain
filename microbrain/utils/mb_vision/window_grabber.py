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

            # Skip minimized windows (often report -32000/-32000 and can crash mss grabs)
            try:
                if bool(getattr(w, "isMinimized", False)):
                    continue
            except Exception:
                pass
            try:
                if int(w.left) <= -32000 and int(w.top) <= -32000:
                    continue
            except Exception:
                pass

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

    left = int(rect.get("left", 0))
    top = int(rect.get("top", 0))
    width = max(1, int(rect.get("width", 1)))
    height = max(1, int(rect.get("height", 1)))

    # Minimized / invalid coords guard (prevents native crash)
    if left <= -32000 and top <= -32000:
        raise ValueError("window appears minimized/invalid (left/top ~ -32000)")

    with mss.mss() as sct:
        vb = sct.monitors[0]  # virtual desktop (multi-monitor safe)
        vleft = int(vb.get("left", 0))
        vtop = int(vb.get("top", 0))
        vright = vleft + int(vb.get("width", 0))
        vbottom = vtop + int(vb.get("height", 0))

        if vright <= vleft or vbottom <= vtop:
            raise ValueError(f"invalid virtual desktop bounds: {vb!r}")

        # Intersect requested rect with virtual desktop bounds
        right = left + width
        bottom = top + height

        ileft = max(vleft, left)
        itop = max(vtop, top)
        iright = min(vright, right)
        ibottom = min(vbottom, bottom)

        if iright <= ileft or ibottom <= itop:
            raise ValueError(f"capture rect outside virtual desktop: rect={rect!r} virtual={vb!r}")

        mon = {"left": ileft, "top": itop, "width": int(iright - ileft), "height": int(ibottom - itop)}

        with mss.mss() as sct:
            try:
                img = np.array(sct.grab(mon))  # BGRA
            except Exception as e:
                return None  # caller should log + skip this tick
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

    ``focus_xy`` accepts normalized coordinates and optional focus metadata:
      {x:0..1, y:0..1, radius:0..1, mode:str}

    The radius is interpreted as a fraction of the smaller frame dimension.
    This lets the preview reflect MB's current attention aperture instead of
    always drawing a fixed-size circle.
    """
    try:
        import cv2
    except Exception:
        return  # preview-only feature

    h, w = frame_bgr.shape[:2]
    fx = 0.5
    fy = 0.5
    radius = 0.08
    mode = "roam"
    if isinstance(focus_xy, dict):
        try:
            fx = float(focus_xy.get("x", 0.5))
            fy = float(focus_xy.get("y", 0.5))
            radius = float(focus_xy.get("radius", focus_xy.get("r", 0.08)) or 0.08)
            mode = str(focus_xy.get("mode", "roam") or "roam")
        except Exception:
            fx, fy, radius, mode = 0.5, 0.5, 0.08, "roam"

    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    radius = max(0.03, min(0.35, radius))
    cx = int(fx * w)
    cy = int(fy * h)

    r = max(8, int(min(w, h) * radius))
    th = max(1, int(min(w, h) * 0.004))

    # outer ring (white)
    cv2.circle(frame_bgr, (cx, cy), r, (255, 255, 255), th, lineType=cv2.LINE_AA)
    # small inner ring to imply "tightening" attention as radius changes
    inner_r = max(5, int(r * 0.35))
    cv2.circle(frame_bgr, (cx, cy), inner_r, (255, 255, 255), max(1, th - 1), lineType=cv2.LINE_AA)
    # inner dot
    cv2.circle(frame_bgr, (cx, cy), max(2, th), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # short mode label near the reticle for the viewable sidecar
    try:
        label = str(mode or "roam")[:12]
        cv2.putText(
            frame_bgr,
            label,
            (max(0, cx + r + 6), max(12, cy - r - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    except Exception:
        pass
