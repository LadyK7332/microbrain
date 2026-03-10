from __future__ import annotations

import hashlib
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.evidence import append_evidence_index, ensure_evidence_session, sha256_file, write_wav_mono_i16

NEURON_NAME = Path(__file__).stem


class EvidenceRecorderAudioNeuron(BaseNeuron):
    """Split-stream raw evidence recorder (audio side)."""

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._ring: Deque[dict] = deque()
        self._last_session_id: str = ""
        self._flushed_ids: set[str] = set()
        self._seq: int = 0
        self._last_pcm_ts: float = 0.0  # last seen time of percept/audio_pcm

    def _trim_ring(self, preroll_s: float, now: float) -> None:
        while self._ring and (now - float(self._ring[0].get("ts", 0.0))) > preroll_s:
            self._ring.popleft()

    async def _write_chunk(self, ctx, session_dir: Path, item: dict) -> None:
        chunk_id = str(item.get("chunk_id", "") or "")
        if not chunk_id or chunk_id in self._flushed_ids:
            return

        self._seq += 1
        ts = float(item.get("ts", time.time()) or time.time())
        sample_rate = int(item.get("sample_rate", 16000) or 16000)
        pcm = bytes(item.get("pcm_bytes", b""))
        topic = str(item.get("topic", "") or "")

        out_name = f"chunk-{self._seq:06d}.wav"
        out_path = session_dir / "audio" / out_name
        write_wav_mono_i16(out_path, pcm, sample_rate)
        sha = sha256_file(out_path)
        await append_evidence_index(
            ctx,
            session_dir,
            kind="audio_chunk",
            rel_path=f"audio/{out_name}",
            ts=ts,
            sha256=sha,
            extra={"sample_rate": sample_rate, "topic": topic, "bytes": len(pcm)},
        )
        self._flushed_ids.add(chunk_id)

    async def _ensure_session_state(self, ctx) -> tuple[str, Path]:
        trigger = {
            "reason": str(await ctx.get_kv("er:last_reason", "") or ""),
            "level": int(await ctx.get_kv("er:last_level", 0) or 0),
            "source": str(await ctx.get_kv("er:last_source", "") or ""),
        }
        sess_id, sess_dir = await ensure_evidence_session(ctx, trigger=trigger)
        if sess_id != self._last_session_id:
            self._last_session_id = sess_id
            self._flushed_ids = set()
            self._seq = 0
        return sess_id, sess_dir

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in ("percept/audio_pcm", "percept/audio_utterance", "clock/tick"):
            return []

        enabled = bool(await ctx.get_kv("er:enabled", True))
        if not enabled:
            return []

        if event.topic == "clock/tick":
            armed = bool(await ctx.get_kv("er:armed", False))
            manual_hold = bool(await ctx.get_kv("er:manual_hold", False))
            if armed and not manual_hold:
                last_trigger = float(await ctx.get_kv("er:last_trigger_ts", 0.0) or 0.0)
                postroll_s = float(await ctx.get_kv("er:postroll_s", 30.0) or 30.0)
                if last_trigger > 0.0 and (time.time() - last_trigger) >= postroll_s:
                    await ctx.set_kv("er:armed", False)
                    await ctx.set_kv("er:session_id", "")
                    await ctx.set_kv("er:session_dir", "")
                    await ctx.set_kv("er:session_chain", "")
                    self._last_session_id = ""
                    self._flushed_ids = set()
                    self._seq = 0
            return []

        if not bool(await ctx.get_kv("er:audio_enabled", True)):
            return []


        # Prefer continuous raw PCM over utterance snippets for evidence.
        prefer_pcm = bool(await ctx.get_kv("er:audio_prefer_pcm", True))
        allow_fallback = bool(await ctx.get_kv("er:audio_allow_utterance_fallback", True))
        pcm_recent_s = float(await ctx.get_kv("er:audio_pcm_recent_s", 2.0) or 2.0)
        now_ts = float(event.timestamp or time.time())
        if event.topic == "percept/audio_pcm":
            self._last_pcm_ts = now_ts
        elif event.topic == "percept/audio_utterance" and prefer_pcm:
            if not allow_fallback:
                return []
            if self._last_pcm_ts and (now_ts - self._last_pcm_ts) <= pcm_recent_s:
                # We are receiving PCM; skip utterance audio to avoid duplicate/whacky capture.
                return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        pcm = payload.get("pcm_bytes", b"")
        if not isinstance(pcm, (bytes, bytearray)) or len(pcm) < 64:
            return []

        sample_rate = int(payload.get("sample_rate", 16000) or 16000)
        now = now_ts
        chunk_id = hashlib.sha1(bytes(pcm) + f"|{sample_rate}|{event.topic}|{now:.6f}".encode("utf-8")).hexdigest()
        item = {"ts": now, "sample_rate": sample_rate, "pcm_bytes": bytes(pcm), "topic": event.topic, "chunk_id": chunk_id}

        preroll_s = float(await ctx.get_kv("er:preroll_s", 20.0) or 20.0)
        self._ring.append(item)
        self._trim_ring(preroll_s, now)

        if not bool(await ctx.get_kv("er:armed", False)):
            return []

        _sess_id, sess_dir = await self._ensure_session_state(ctx)
        for q in list(self._ring):
            await self._write_chunk(ctx, sess_dir, q)

        await ctx.set_kv("er:last_capture_ts", time.time())
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio_pcm", "percept/audio_utterance", "clock/tick"],
        output_topics=[],
        priority=35,
        cooldown_sec=0.0,
    )
    yield EvidenceRecorderAudioNeuron(cfg)
