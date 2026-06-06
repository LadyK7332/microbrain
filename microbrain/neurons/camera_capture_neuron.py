from __future__ import annotations

import time
import threading
from queue import Empty, Queue
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx
from microbrain.utils.mb_vision.camera_grabber import open_camera, read_bgr, save_jpeg
from microbrain.utils.mb_vision.window_grabber import draw_focus_reticle


NEURON_NAME = Path(__file__).stem


class CameraCaptureNeuron(BaseNeuron):
    """
    Supervised webcam/camera capture sensor.

    Control is user-invoked through /camera commands in input_text.py:
      - /camera list
      - /camera select <index>
      - /camera on|off
      - /camera preview on|off
      - /camera status

    Emits:
      - percept/vision payload {ts, frame_id, data_ref, width, height, format, camera}

    The command text remains control-plane. Only actual camera frames become
    vision percepts, and only after /camera on.
    """

    PREVIEW_WINDOW = "MB Camera Preview"

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._cap = None
        self._cap_index: int | None = None
        self._preview_q: Queue = Queue(maxsize=1)
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        if event.topic == "control/camera":
            return await self._handle_control(event, ctx)
        if event.topic == "clock/tick":
            return await self._tick_capture(event, ctx)
        return []

    async def _handle_control(self, event: Event, ctx) -> list[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        action = str(payload.get("action", "") or "").lower().strip()

        if action == "select":
            cam = payload.get("camera", None)
            idx = payload.get("index", None)
            try:
                idx = int(idx if idx is not None else (cam or {}).get("index", 0))
            except Exception:
                idx = 0
            idx = max(0, min(32, idx))
            if not isinstance(cam, dict):
                cam = {"index": idx, "name": f"Camera {idx}", "ts": time.time()}
            await ctx.set_kv("camera:selected", dict(cam))
            await ctx.set_kv("camera:selected_index", idx)
            self._close_capture()
            self.debug("camera_selected", index=idx, camera=cam)
            return []

        if action == "on":
            await ctx.set_kv("camera:enabled", True)
            self.debug("camera_enabled_set", enabled=True)
            return []

        if action == "off":
            await ctx.set_kv("camera:enabled", False)
            await ctx.set_kv("camera:preview", False)
            self._preview_close()
            self._close_capture()
            self.debug("camera_enabled_set", enabled=False)
            return []

        if action == "preview_on":
            await ctx.set_kv("camera:preview", True)
            self.debug("camera_preview_set", enabled=True)
            return []

        if action == "preview_off":
            await ctx.set_kv("camera:preview", False)
            self._preview_close()
            self.debug("camera_preview_set", enabled=False)
            return []

        self.debug("camera_control_unknown", action=action)
        return []

    async def _tick_capture(self, event: Event, ctx) -> list[Event]:
        enabled = bool(await ctx.get_kv("camera:enabled", False))
        if not enabled:
            return []

        selected = await ctx.get_kv("camera:selected", None)
        selected_index = await ctx.get_kv("camera:selected_index", 0)
        try:
            index = int(selected_index if selected_index is not None else 0)
        except Exception:
            index = 0
        index = max(0, min(32, index))

        now = time.time()
        last_ts = float(await self.load_state(ctx, "last_capture_ts", 0.0) or 0.0)
        fps = await ctx.get_kv("camera:fps", 2.0)
        try:
            fps = float(fps)
        except Exception:
            fps = 2.0
        fps = max(0.2, min(30.0, fps))
        interval = 1.0 / fps
        if (now - last_ts) < interval:
            return []
        await self.save_state(ctx, "last_capture_ts", now)

        try:
            cap = self._ensure_capture(index)
            frame = read_bgr(cap)
        except Exception as e:
            self.debug("camera_capture_error", err=repr(e), index=index)
            self._close_capture()
            return []

        if frame is None:
            self.debug("camera_capture_empty", index=index)
            return []

        h, w = frame.shape[:2]

        preview = bool(await ctx.get_kv("camera:preview", False))
        if preview:
            try:
                prev = frame.copy()
                focus_xy = await self._read_focus_state(ctx)
                draw_focus_reticle(prev, focus_xy)
                self._preview_show(prev)
            except Exception as e:
                self.debug("camera_preview_error", err=repr(e))

        save_mode = str(await ctx.get_kv("camera:save_mode", "gated") or "gated").lower().strip()
        dupe_thresh = float(await ctx.get_kv("camera:dupe_thresh", 0.06) or 0.06)
        max_stale_s = float(await ctx.get_kv("camera:max_stale_s", 20.0) or 20.0)

        cur_hash = self._dhash64(frame)
        last_hash = int(await self.load_state(ctx, "last_saved_dhash", 0) or 0)
        last_save_ts = float(await self.load_state(ctx, "last_saved_frame_ts", 0.0) or 0.0)
        dist = (cur_hash ^ last_hash).bit_count() if last_hash else 64
        ratio = dist / 64.0
        stale = (last_save_ts <= 0.0) or ((now - last_save_ts) >= max_stale_s)

        if save_mode == "gated" and (not stale) and last_hash and ratio < dupe_thresh:
            self.debug("camera_frame_skip_duplicate", ratio=ratio, dupe_thresh=dupe_thresh, stale=stale)
            return []

        frame_id = int(await self.load_state(ctx, "frame_id", 0) or 0) + 1
        await self.save_state(ctx, "frame_id", frame_id)

        memdir = await resolve_memdir_ctx(ctx, fallback=None)
        base = Path(memdir) / "sight" / "camera_frames"
        base.mkdir(parents=True, exist_ok=True)
        out_path = base / ("latest.jpg" if save_mode == "latest" else f"camera-{frame_id:06d}.jpg")

        try:
            save_jpeg(frame, str(out_path))
        except Exception as e:
            self.debug("camera_save_error", err=repr(e), path=str(out_path))
            return []

        await self.save_state(ctx, "last_saved_dhash", int(cur_hash))
        await self.save_state(ctx, "last_saved_frame_ts", now)

        frames_keep = await ctx.get_kv("camera:frames_keep", 500)
        try:
            frames_keep = int(frames_keep)
        except Exception:
            frames_keep = 500
        if save_mode != "latest" and frames_keep > 0:
            try:
                files = sorted(base.glob("camera-*.jpg"))
                if len(files) > frames_keep:
                    for p in files[: len(files) - frames_keep]:
                        try:
                            p.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

        cam_info = dict(selected) if isinstance(selected, dict) else {"index": index, "name": f"Camera {index}"}
        cam_info.setdefault("index", index)
        cam_info.setdefault("name", f"Camera {index}")

        payload = {
            "ts": now,
            "frame_id": frame_id,
            "data_ref": str(out_path),
            "width": int(w),
            "height": int(h),
            "format": "jpeg",
            "camera": cam_info,
            "focus": await self._read_focus_state(ctx),
        }

        return [
            Event(
                topic="percept/vision",
                payload=payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "vision_frame", "sensor": "camera"},
            )
        ]

    def _ensure_capture(self, index: int):
        if self._cap is not None and self._cap_index == index:
            return self._cap
        self._close_capture()
        self._cap = open_camera(index)
        self._cap_index = index
        return self._cap

    def _close_capture(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        self._cap_index = None

    async def _read_focus_state(self, ctx) -> dict[str, float | str]:
        gaze_state = await ctx.get_kv("vision:gaze_state", None)
        if isinstance(gaze_state, dict):
            try:
                return {
                    "x": max(0.0, min(1.0, float(gaze_state.get("x", 0.5) or 0.5))),
                    "y": max(0.0, min(1.0, float(gaze_state.get("y", 0.5) or 0.5))),
                    "radius": max(0.03, min(0.35, float(gaze_state.get("radius", 0.08) or 0.08))),
                    "mode": str(gaze_state.get("mode", "camera") or "camera"),
                }
            except Exception:
                pass
        return {"x": 0.5, "y": 0.5, "radius": 0.08, "mode": "camera"}

    @staticmethod
    def _dhash64(frame_bgr) -> int:
        import numpy as np

        h, w = frame_bgr.shape[:2]
        size = 8
        gray = (
            frame_bgr[:, :, 0].astype(np.uint16)
            + frame_bgr[:, :, 1].astype(np.uint16)
            + frame_bgr[:, :, 2].astype(np.uint16)
        ) // 3
        ys = ((np.arange(size) + 0.5) * h / size).astype(int)
        xs = ((np.arange(size + 1) + 0.5) * w / (size + 1)).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        sample = gray[ys[:, None], xs[None, :]]
        diff = sample[:, 1:] > sample[:, :-1]
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
        try:
            cv2.namedWindow(self.PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        last = None
        while not self._preview_stop.is_set():
            try:
                last = self._preview_q.get(timeout=0.05)
            except Empty:
                pass
            if last is not None:
                try:
                    cv2.imshow(self.PREVIEW_WINDOW, last)
                except Exception:
                    last = None
            try:
                cv2.waitKey(1)
            except Exception:
                break
        try:
            cv2.destroyWindow(self.PREVIEW_WINDOW)
        except Exception:
            pass

    def _preview_show(self, frame_bgr) -> None:
        self._ensure_preview_thread()
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
        self._preview_thread = None


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "control/camera"],
        output_topics=["percept/vision"],
        priority=3,
    )
    return [CameraCaptureNeuron(cfg)]
