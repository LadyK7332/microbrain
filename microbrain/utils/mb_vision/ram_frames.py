"""Bounded RAM frame buffer for live vision.

Raw camera/window frames are sensory samples, not durable memory.  These helpers
keep JPEG-compressed samples in KV for current perception/debugging and allow
explicit disk persistence elsewhere only when a caller asks for it.
"""

from __future__ import annotations

import io
from typing import Any


# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

RAM_FRAME_KEEP_DEFAULT = 120
RAM_FRAME_TTL_S_DEFAULT = 10.0
RAM_JPEG_QUALITY_DEFAULT = 82

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

RAM_FRAME_LATEST_KEY = "vision:frame:latest"
RAM_FRAME_RING_KEY = "vision:frame:ring"


def encode_jpeg_bytes(frame_bgr, quality: int = RAM_JPEG_QUALITY_DEFAULT) -> bytes:
    """Encode an OpenCV-style BGR ndarray as JPEG bytes without disk I/O."""

    quality = max(30, min(100, int(quality)))
    try:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode returned False")
        return bytes(encoded.tobytes())
    except Exception:
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb.astype("uint8"))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


async def store_ram_frame(
    ctx,
    *,
    sensor: str,
    frame_id: int,
    timestamp: float,
    jpeg_bytes: bytes,
    width: int,
    height: int,
) -> str:
    """Store a bounded recent-frame ring in KV and return its RAM reference."""

    sensor = str(sensor or "vision").strip().lower().replace(" ", "_")
    ref = f"ram:vision:{sensor}:{int(frame_id)}"
    packet = {
        "ref": ref,
        "frame_id": int(frame_id),
        "ts": float(timestamp),
        "sensor": sensor,
        "width": int(width),
        "height": int(height),
        "format": "jpeg",
        "jpeg_bytes": bytes(jpeg_bytes),
    }

    keep = int(await ctx.get_kv("vision:ram_frames_keep", RAM_FRAME_KEEP_DEFAULT) or RAM_FRAME_KEEP_DEFAULT)
    ttl_s = float(await ctx.get_kv("vision:ram_frame_ttl_s", RAM_FRAME_TTL_S_DEFAULT) or RAM_FRAME_TTL_S_DEFAULT)
    keep = max(1, min(600, keep))
    ttl_s = max(0.5, min(60.0, ttl_s))

    ring = list(await ctx.get_kv(RAM_FRAME_RING_KEY, []) or [])
    ring.append(packet)
    cutoff = float(timestamp) - ttl_s
    ring = [
        row
        for row in ring
        if isinstance(row, dict) and float(row.get("ts", 0.0) or 0.0) >= cutoff
    ][-keep:]
    await ctx.set_kv(RAM_FRAME_RING_KEY, ring)
    await ctx.set_kv(RAM_FRAME_LATEST_KEY, packet)
    return ref


async def get_ram_frame(ctx, ref: str) -> dict[str, Any] | None:
    ref = str(ref or "")
    latest = await ctx.get_kv(RAM_FRAME_LATEST_KEY, None)
    if isinstance(latest, dict) and str(latest.get("ref") or "") == ref:
        return latest
    ring = await ctx.get_kv(RAM_FRAME_RING_KEY, [])
    if isinstance(ring, list):
        for row in reversed(ring):
            if isinstance(row, dict) and str(row.get("ref") or "") == ref:
                return row
    return None
