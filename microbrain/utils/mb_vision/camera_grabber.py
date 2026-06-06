from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "opencv"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "name": str(self.name),
            "width": int(self.width or 0),
            "height": int(self.height or 0),
            "fps": float(self.fps or 0.0),
            "backend": str(self.backend or "opencv"),
        }


def _require_cv2():
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception as e:
        raise RuntimeError("Camera deps missing. Install: opencv-python numpy") from e


def _open_backend_flags():
    _require_cv2()
    import cv2

    # CAP_DSHOW avoids slow MSMF probing on many Windows installs.
    if sys.platform.startswith("win"):
        return [getattr(cv2, "CAP_DSHOW", 700), 0]
    return [0]


def _camera_name_from_enumerator(index: int) -> str | None:
    """
    Optional nicer camera naming. This package is not required; plain OpenCV
    index probing remains the fallback so /camera works without extra tools.
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras  # type: ignore
    except Exception:
        return None

    try:
        for cam in enumerate_cameras():
            cam_index = getattr(cam, "index", None)
            if cam_index is None:
                cam_index = getattr(cam, "camera_index", None)
            if cam_index is None:
                continue
            if int(cam_index) != int(index):
                continue
            name = getattr(cam, "name", None) or getattr(cam, "display_name", None)
            if name:
                return str(name)
    except Exception:
        return None
    return None


def list_cameras(max_index: int = 8) -> list[CameraInfo]:
    """
    Probe a small range of OpenCV camera indices.

    This is intentionally user-triggered by /camera list, not performed during
    MB startup. It may briefly touch camera devices, so it belongs in the
    camera/control lane rather than cognition or memory.
    """
    _require_cv2()
    import cv2

    out: list[CameraInfo] = []
    seen: set[int] = set()
    max_index = max(0, min(32, int(max_index or 8)))

    for idx in range(max_index + 1):
        for backend in _open_backend_flags():
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                if not cap or not cap.isOpened():
                    continue

                # One quick read filters ghost indices on some systems.
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Some virtual devices open but do not produce immediately.
                    # Keep them only if OpenCV reports a non-zero size.
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    if width <= 0 or height <= 0:
                        continue
                else:
                    height, width = frame.shape[:2]

                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                name = _camera_name_from_enumerator(idx) or f"Camera {idx}"
                backend_name = "dshow" if backend == getattr(cv2, "CAP_DSHOW", None) else "opencv"
                if idx not in seen:
                    out.append(CameraInfo(index=idx, name=name, width=width, height=height, fps=fps, backend=backend_name))
                    seen.add(idx)
                break
            except Exception:
                continue
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

    return out


def open_camera(index: int):
    _require_cv2()
    import cv2

    last_error: Exception | None = None
    for backend in _open_backend_flags():
        try:
            cap = cv2.VideoCapture(int(index), backend) if backend else cv2.VideoCapture(int(index))
            if cap and cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(f"Could not open camera {index}: {last_error!r}") from last_error
    raise RuntimeError(f"Could not open camera {index}")


def read_bgr(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def save_jpeg(frame_bgr, out_path: str, quality: int = 85) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _require_cv2()
    import cv2

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok = cv2.imwrite(out_path, frame_bgr, params)
    if not ok:
        raise RuntimeError("cv2.imwrite returned False")
