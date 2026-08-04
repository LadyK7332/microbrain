from __future__ import annotations

import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Any, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

# Reuse the small MB vision utilities (pygetwindow + mss)
from microbrain.utils.mb_vision.ram_frames import encode_jpeg_bytes, store_ram_frame
from microbrain.utils.mb_vision.window_grabber import (
    list_windows,
    pick_window,
    grab_bgr,
    save_jpeg,
    draw_focus_reticle,
)

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("vision")


class VisionWindowCaptureNeuron(BaseNeuron):
    """
    Supervised window capture sensor.

    Control is user-invoked via text UI commands (router_text):
      - /vision list
      - /vision select <idx|title_substring>
      - /vision on|off
      - /vision preview on|off
      - /focus center
      - /focus <x> <y>   (normalized 0..1)

    Emits:
      - percept/vision payload {ts, frame_id, data_ref, width, height, format, window}

    Raw frames are RAM-first. ``vision:save_mode=ram`` is the default; durable
    JPEG writes occur only when explicitly switched to latest/gated/all.
    """
    PREVIEW_WINDOW = "MB Vision Preview"

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._preview_q: Queue = Queue(maxsize=1)  # drop-old, keep-latest
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic == "control/vision":
            return await self._handle_control(event, ctx)

        if event.topic == "control/focus":
            return await self._handle_focus(event, ctx)

        if event.topic == SERVICE_TOPIC:
            return await self._tick_capture(event, ctx)

        return []

    async def _handle_control(self, event: Event, ctx) -> list[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        action = str(payload.get("action", "") or "").lower().strip()

        # Persist these as KV so other neurons can observe
        if action == "on":
            await ctx.set_kv("vision:enabled", True)
            self.debug("vision_enabled_set", enabled=True)
            return []
        if action == "off":
            await ctx.set_kv("vision:enabled", False)
            await ctx.set_kv("vision:preview", False)
            self._preview_close()
            self.debug("vision_enabled_set", enabled=False)
            return []
        if action == "preview_on":
            await ctx.set_kv("vision:preview", True)
            self.debug("vision_preview_set", enabled=True)
            return []
        if action == "preview_off":
            await ctx.set_kv("vision:preview", False)
            self._preview_close()
            self.debug("vision_preview_set", enabled=False)
            return []
        if action == "select":
            # Prefer cached window payload (prevents re-enumeration hangs/crashes)
            win = payload.get("window", None)
            if isinstance(win, dict) and isinstance(win.get("rect"), dict):
                title = str(win.get("title", "") or "")
                rect_in = win.get("rect") or {}
                rect = {
                    "left": int(rect_in.get("left", 0)),
                    "top": int(rect_in.get("top", 0)),
                    "width": int(rect_in.get("width", 1)),
                    "height": int(rect_in.get("height", 1)),
                }
                await ctx.set_kv(
                    "vision:window",
                    {
                        "title": title,
                        "rect": rect,
                        "ts": time.time(),
                    },
                )
                self.debug("vision_selected", title=title, rect=rect, mode="cached")
                return []

            # Fallback: old selector behavior
            selector = str(payload.get("selector", "") or "").strip()
            if not selector:
                self.debug("vision_select_failed", selector=selector, reason="no selector/window")
                return []

            wins = list_windows()
            chosen = pick_window(wins, selector)
            if not chosen:
                self.debug("vision_select_failed", selector=selector, count=len(wins))
                return []

            await ctx.set_kv(
                "vision:window",
                {
                    "title": chosen.title,
                    "rect": chosen.rect,
                    "ts": time.time(),
                },
            )
            self.debug("vision_selected", title=chosen.title, rect=chosen.rect, mode="selector")
            return []

        if action == "list":
            # Listing is handled in router_text for user feedback; nothing to do here.
            return []

        # Unknown action -> ignore
        self.debug("vision_control_unknown", action=action)
        return []

    async def _handle_focus(self, event: Event, ctx) -> list[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        action = str(payload.get("action", "") or "").lower().strip()

        if action == "center":
            await self._write_focus_state(ctx, {"x": 0.5, "y": 0.5, "radius": 0.08, "mode": "manual"})
            self.debug("vision_focus_set", x=0.5, y=0.5, radius=0.08, mode="center")
            return []

        if action == "set":
            try:
                x = float(payload.get("x", 0.5))
                y = float(payload.get("y", 0.5))
            except Exception:
                x, y = 0.5, 0.5
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            prev_focus = await self._read_focus_state(ctx)
            await self._write_focus_state(ctx, {"x": x, "y": y, "radius": prev_focus.get("radius", 0.08), "mode": "manual"})
            self.debug("vision_focus_set", x=x, y=y, radius=prev_focus.get("radius", 0.08), mode="manual")
            return []

        self.debug("vision_focus_unknown", action=action)
        return []

    async def _read_focus_state(self, ctx) -> dict[str, float | str]:
        gaze_state = await ctx.get_kv("vision:gaze_state", None)
        if isinstance(gaze_state, dict):
            try:
                x = float(gaze_state.get("x", gaze_state.get("cx", 0.5)) or 0.5)
                y = float(gaze_state.get("y", gaze_state.get("cy", 0.5)) or 0.5)
                radius = float(gaze_state.get("radius", 0.08) or 0.08)
                mode = str(gaze_state.get("mode", "roam") or "roam")
                return {
                    "x": max(0.0, min(1.0, x)),
                    "y": max(0.0, min(1.0, y)),
                    "radius": max(0.03, min(0.35, radius)),
                    "mode": mode,
                }
            except Exception:
                pass

        focus_xy = await ctx.get_kv("vision:focus_xy", {"x": 0.5, "y": 0.5})
        x = 0.5
        y = 0.5
        if isinstance(focus_xy, dict):
            try:
                x = float(focus_xy.get("x", 0.5) or 0.5)
                y = float(focus_xy.get("y", 0.5) or 0.5)
            except Exception:
                x, y = 0.5, 0.5
        return {
            "x": max(0.0, min(1.0, x)),
            "y": max(0.0, min(1.0, y)),
            "radius": 0.08,
            "mode": "manual",
        }

    async def _write_focus_state(self, ctx, focus: dict[str, float | str]) -> None:
        x = max(0.0, min(1.0, float(focus.get("x", 0.5) or 0.5)))
        y = max(0.0, min(1.0, float(focus.get("y", 0.5) or 0.5)))
        radius = max(0.03, min(0.35, float(focus.get("radius", 0.08) or 0.08)))
        mode = str(focus.get("mode", "manual") or "manual")
        gaze_state = dict(await ctx.get_kv("vision:gaze_state", {}) or {})
        gaze_state.update({"x": x, "y": y, "radius": radius, "mode": mode})
        await ctx.set_kv("vision:gaze_state", gaze_state)
        await ctx.set_kv("vision:focus_xy", {"x": x, "y": y, "radius": radius, "mode": mode})

    async def _tick_capture(self, event: Event, ctx) -> list[Event]:
        enabled = bool(await ctx.get_kv("vision:enabled", False))
        if not enabled:
            return []

        window = await ctx.get_kv("vision:window", None)
        if not isinstance(window, dict):
            # No window selected; nothing to capture.
            return []

        rect = window.get("rect")
        if not isinstance(rect, dict):
            return []

        # Simple pacing
        now = time.time()
        last_ts = float(await self.load_state(ctx, "last_capture_ts", 0.0) or 0.0)
        fps = await ctx.get_kv("vision:fps", 2.0)
        try:
            fps = float(fps)
        except Exception:
            fps = 2.0
        fps = max(0.2, min(30.0, fps))
        interval = 1.0 / fps

        if (now - last_ts) < interval:
            return []

        await self.save_state(ctx, "last_capture_ts", now)

        # Capture
        try:
            frame = grab_bgr(rect)
        except Exception as e:
            self.debug("vision_capture_error", err=repr(e), rect=rect)
            return []

        if frame is None:
            self.debug("vision_capture_empty", rect=rect)
            return []

        h, w = frame.shape[:2]

        # Preview overlay (never saved; only displayed)
        preview = bool(await ctx.get_kv("vision:preview", False))
        if preview:
            focus_xy = await self._read_focus_state(ctx)
            try:
                prev = frame.copy()
                draw_focus_reticle(prev, focus_xy)
                self._preview_show(prev)
            except Exception as e:
                self.debug("vision_preview_error", err=repr(e))

        # Raw frames live in a bounded RAM ring by default. Disk persistence is
        # opt-in (latest/gated/all) and is for evidence/debugging, not perception.
        save_mode = str(await ctx.get_kv("vision:save_mode", "ram") or "ram").lower().strip()
        jpeg_quality = int(await ctx.get_kv("vision:ram_jpeg_quality", 82) or 82)

        frame_id = int(await self.load_state(ctx, "frame_id", 0) or 0) + 1
        await self.save_state(ctx, "frame_id", frame_id)
        try:
            jpeg_bytes = encode_jpeg_bytes(frame, quality=jpeg_quality)
        except Exception as e:
            self.debug("vision_ram_encode_error", err=repr(e), rect=rect)
            return []

        ram_ref = await store_ram_frame(
            ctx,
            sensor="window",
            frame_id=frame_id,
            timestamp=now,
            jpeg_bytes=jpeg_bytes,
            width=int(w),
            height=int(h),
        )

        persistent_ref = ""
        if save_mode in {"latest", "gated", "all"}:
            dupe_thresh = float(await ctx.get_kv("vision:dupe_thresh", 0.06) or 0.06)
            max_stale_s = float(await ctx.get_kv("vision:max_stale_s", 20.0) or 20.0)
            cur_hash = self._dhash64(frame)
            last_hash = int(await self.load_state(ctx, "last_saved_dhash", 0) or 0)
            last_save_ts = float(await self.load_state(ctx, "last_saved_frame_ts", 0.0) or 0.0)
            dist = (cur_hash ^ last_hash).bit_count() if last_hash else 64
            ratio = dist / 64.0
            stale = (last_save_ts <= 0.0) or ((now - last_save_ts) >= max_stale_s)
            persist = save_mode in {"latest", "all"} or not last_hash or stale or ratio >= dupe_thresh

            if persist:
                memdir = await resolve_memdir_ctx(ctx, fallback=None)
                base = Path(memdir) / "sight" / "frames"
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / ("latest.jpg" if save_mode == "latest" else f"frame-{frame_id:06d}.jpg")
                try:
                    save_jpeg(frame, str(out_path))
                    persistent_ref = str(out_path)
                    await self.save_state(ctx, "last_saved_dhash", int(cur_hash))
                    await self.save_state(ctx, "last_saved_frame_ts", now)
                except Exception as e:
                    self.debug("vision_save_error", err=repr(e), path=str(out_path))

                if save_mode != "latest":
                    frames_keep = int(await ctx.get_kv("vision:frames_keep", 500) or 500)
                    if frames_keep > 0:
                        try:
                            files = sorted(base.glob("frame-*.jpg"))
                            if len(files) > frames_keep:
                                for old_path in files[: len(files) - frames_keep]:
                                    try:
                                        old_path.unlink()
                                    except Exception:
                                        pass
                        except Exception:
                            pass

        focus_state = await self._read_focus_state(ctx)
        payload = {
            "ts": now,
            "frame_id": frame_id,
            "data_ref": ram_ref,
            "frame_ref": ram_ref,
            "persistent_ref": persistent_ref,
            "storage": "ram" if not persistent_ref else "ram+disk",
            "width": int(w),
            "height": int(h),
            "format": "jpeg",
            "window": {
                "title": str(window.get("title", "")),
                "rect": rect,
            },
            "focus": focus_state,
        }

        return [
            Event(
                topic="percept/vision",
                payload=payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "vision_frame", "sensor": "window"},
            )
        ]

    @staticmethod
    def _dhash64(frame_bgr) -> int:
        """
        Cheap perceptual hash (64-bit). Works well for 'is this basically the same frame?' gating.
        """
        import numpy as np

        h, w = frame_bgr.shape[:2]
        size = 8  # yields 8x8 comparisons => 64 bits

        # grayscale (uint16 avoids overflow)
        gray = (
            frame_bgr[:, :, 0].astype(np.uint16)
            + frame_bgr[:, :, 1].astype(np.uint16)
            + frame_bgr[:, :, 2].astype(np.uint16)
        ) // 3

        ys = ((np.arange(size) + 0.5) * h / size).astype(int)
        xs = ((np.arange(size + 1) + 0.5) * w / (size + 1)).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        sample = gray[ys[:, None], xs[None, :]]  # (8, 9)
        diff = sample[:, 1:] > sample[:, :-1]    # (8, 8) bool

        bits = 0
        for b in diff.flatten():
            bits = (bits << 1) | int(b)
        return bits

    def _ensure_preview_thread(self) -> None:
        if self._preview_thread and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        t = threading.Thread(target=self._preview_worker, daemon=True)
        self._preview_thread = t
        t.start()

    def _preview_worker(self) -> None:
        try:
            import cv2
        except Exception:
            return

        # Keep a window alive and pump events here, not on the Textual/async thread.
        try:
            cv2.namedWindow(self.PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        except Exception:
            pass

        last = None
        while not self._preview_stop.is_set():
            try:
                # Try to grab newest frame; if none, still pump waitKey.
                last = self._preview_q.get(timeout=0.05)
            except Empty:
                pass

            if last is not None:
                try:
                    cv2.imshow(self.PREVIEW_WINDOW, last)
                except Exception:
                    # If window dies, don't take down MB
                    last = None

            # Pump window events. Keep this thread responsive.
            try:
                cv2.waitKey(1)
            except Exception:
                break

        # Cleanup
        try:
            cv2.destroyWindow(self.PREVIEW_WINDOW)
        except Exception:
            pass

    def _preview_show(self, frame_bgr) -> None:
        self._ensure_preview_thread()

        # drop-old / keep-latest so we never backlog
        try:
            if self._preview_q.full():
                _ = self._preview_q.get_nowait()
        except Exception:
            pass
        try:
            self._preview_q.put_nowait(frame_bgr)
        except Exception:
            pass

    def _preview_close(self) -> None:
        self._preview_stop.set()
        # Don't hard-join; keep shutdown non-blocking for Textual
        self._preview_thread = None

def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            SERVICE_TOPIC,
            "control/vision",
            "control/focus",
        ],
        output_topics=[
            "percept/vision",
        ],
    )
    return [VisionWindowCaptureNeuron(cfg)]
