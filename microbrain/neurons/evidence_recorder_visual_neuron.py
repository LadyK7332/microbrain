from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.evidence import append_evidence_index, copy_file, ensure_evidence_session, sha256_file

NEURON_NAME = Path(__file__).stem


class EvidenceRecorderVisualNeuron(BaseNeuron):
    """Split-stream raw evidence recorder (visual side)."""

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._ring: Deque[dict] = deque()
        self._last_session_id: str = ""
        self._copied_srcs: set[str] = set()
        self._seq: int = 0

    def _trim_ring(self, preroll_s: float, now: float) -> None:
        while self._ring and (now - float(self._ring[0].get("ts", 0.0))) > preroll_s:
            self._ring.popleft()

    async def _copy_frame(self, ctx, session_dir: Path, item: dict) -> None:
        src = Path(str(item.get("src", "") or ""))
        if not src.exists():
            return
        src_key = str(src.resolve())
        if src_key in self._copied_srcs:
            return

        self._seq += 1
        ts = float(item.get("ts", time.time()) or time.time())
        suffix = src.suffix or ".jpg"
        out_name = f"frame-{self._seq:06d}{suffix}"
        out_path = session_dir / "video" / out_name
        copy_file(src, out_path)
        sha = sha256_file(out_path)

        extra = {
            "frame_id": int(item.get("frame_id", 0) or 0),
            "width": int(item.get("width", 0) or 0),
            "height": int(item.get("height", 0) or 0),
            "format": str(item.get("format", "") or ""),
            "window_title": str(item.get("window_title", "") or ""),
        }
        await append_evidence_index(
            ctx,
            session_dir,
            kind="video_frame",
            rel_path=f"video/{out_name}",
            ts=ts,
            sha256=sha,
            extra=extra,
        )
        self._copied_srcs.add(src_key)

    async def _ensure_session_state(self, ctx) -> tuple[str, Path]:
        trigger = {
            "reason": str(await ctx.get_kv("er:last_reason", "") or ""),
            "level": int(await ctx.get_kv("er:last_level", 0) or 0),
            "source": str(await ctx.get_kv("er:last_source", "") or ""),
        }
        sess_id, sess_dir = await ensure_evidence_session(ctx, trigger=trigger)
        if sess_id != self._last_session_id:
            self._last_session_id = sess_id
            self._copied_srcs = set()
            self._seq = 0
        return sess_id, sess_dir

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/vision":
            return []

        enabled = bool(await ctx.get_kv("er:enabled", True))
        if not enabled or not bool(await ctx.get_kv("er:visual_enabled", True)):
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        data_ref = payload.get("data_ref")
        if not data_ref:
            return []

        now = float(payload.get("ts", event.timestamp) or time.time())
        item = {
            "ts": now,
            "src": str(data_ref),
            "frame_id": payload.get("frame_id", 0),
            "width": payload.get("width", 0),
            "height": payload.get("height", 0),
            "format": payload.get("format", ""),
            "window_title": ((payload.get("window") or {}).get("title", "") if isinstance(payload.get("window"), dict) else ""),
        }

        preroll_s = float(await ctx.get_kv("er:preroll_s", 20.0) or 20.0)
        self._ring.append(item)
        self._trim_ring(preroll_s, now)

        if not bool(await ctx.get_kv("er:armed", False)):
            return []

        _sess_id, sess_dir = await self._ensure_session_state(ctx)
        for q in list(self._ring):
            await self._copy_frame(ctx, sess_dir, q)

        await ctx.set_kv("er:last_capture_ts", time.time())
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/vision"],
        output_topics=[],
        priority=34,
        cooldown_sec=0.0,
    )
    yield EvidenceRecorderVisualNeuron(cfg)
