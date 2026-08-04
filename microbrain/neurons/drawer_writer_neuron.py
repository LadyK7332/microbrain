from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.utils.heartbeat_stream import service_topic

SERVICE_TOPIC = service_topic("maintenance")

@dataclass
class _PendingWrite:
    drawer: str
    record: Any
    data_ref: Optional[str]
    dedupe_key: Optional[str]
    ts: float


class DrawerWriterNeuron(BaseNeuron):
    """
    Buffered drawer writer + announcer.

    Input event:
      topic: "drawer/write"
      payload: {
        "drawer": "sight/exemplars"   (relative to memdir)
        "record": {...}              (JSON-serializable)
        "data_ref": "frame-000001.jpg" (optional)
        "dedupe_key": "..."            (optional; if repeated, skips)
        "format": "jsonl"              (default)
      }

    Output event (after flush):
      topic: "drawer/done"
      payload: {
        "drawer": "...",
        "count": N,
        "last_data_ref": "...",
        "last_dedupe_key": "...",
        "note": "write_committed"
      }

    Output event (on write failure):
      topic: "drawer/error"
      payload: {
        "drawer": "...",
        "error": "...",
        "path": "...",
        "count": N,
        "note": "write_failed"
      }
    """
    def __init__(
        self,
        config: NeuronConfig,
        memdir: str,
        flush_interval_s: float = 1.0,
        max_buffer: int = 25,
        dedupe_ttl_s: float = 20.0,
    ):
        super().__init__(config)

        self.memdir = memdir
        self.flush_interval_s = float(flush_interval_s)
        self.max_buffer = int(max_buffer)
        self.dedupe_ttl_s = float(dedupe_ttl_s)

        self._buf: List[_PendingWrite] = []
        self._last_flush = self._now()

        # lightweight in-RAM dedupe (prevents disk hammering by repeats)
        self._dedupe_seen: Dict[str, float] = {}

        # last-written fingerprint per drawer (extra safety against repeats)
        self._last_fingerprint: Dict[str, str] = {}

    def _now(self) -> float:
        return time.time()

    def _abs_drawer_path(self, drawer: str) -> str:
        drawer = drawer.strip().lstrip("\\/").replace("\\", "/")
        safe_parts = [p for p in drawer.split("/") if p and p not in (".", "..")]
        return os.path.join(self.memdir, *safe_parts)
    
    def _record_fingerprint(self, record: Any) -> str:
        # stable canonical JSON string (acts like a cheap hash)
        try:
            if isinstance(record, dict):
                return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        except TypeError:
            return json.dumps(str(record), separators=(",", ":"), ensure_ascii=False)

    def _prune_dedupe(self, now: float) -> None:
        if not self._dedupe_seen:
            return
        cutoff = now - self.dedupe_ttl_s
        # prune lazily
        for k in list(self._dedupe_seen.keys()):
            if self._dedupe_seen[k] < cutoff:
                del self._dedupe_seen[k]

    def _enqueue(self, drawer: str, record: Any, data_ref: Optional[str], dedupe_key: Optional[str]) -> bool:
        now = self._now()
        self._prune_dedupe(now)

        dedupe_key = dedupe_key or None
        if dedupe_key:
            last = self._dedupe_seen.get(dedupe_key)
            if last is not None and (now - last) <= self.dedupe_ttl_s:
                # skip duplicates within TTL
                return False
            self._dedupe_seen[dedupe_key] = now

        fp = self._record_fingerprint(record)
        last_fp = self._last_fingerprint.get(drawer)
        if last_fp == fp:
            # same record as last write for this drawer -> skip
            return False
        
        self._buf.append(_PendingWrite(drawer=drawer, record=record, data_ref=data_ref, dedupe_key=dedupe_key, ts=now))

        # soft pressure: if buffer is big, flush early
        if len(self._buf) >= self.max_buffer:
            # force flush on next tick or immediate via handle() return path
            pass

        return True

    def _flush(self, correlation_id: Optional[str] = None, reason: str = "flush") -> List[Event]:
        if not self._buf:
            return []
        
        # group by drawer so we append efficiently
        grouped: Dict[str, List[_PendingWrite]] = {}
        for pw in self._buf:
            grouped.setdefault(pw.drawer, []).append(pw)

        events: List[Event] = []
        now = self._now()
        remaining: List[_PendingWrite] = []
        any_success = False

        for drawer, items in grouped.items():
            abs_dir = self._abs_drawer_path(drawer)
            os.makedirs(abs_dir, exist_ok=True)

            # Default: write JSONL “stream” file per drawer.
            # If you prefer “one file per record”, we can flip this later.
            out_path = os.path.join(abs_dir, "stream.jsonl")

            wrote = 0
            last_ref = None
            last_dk = None

            try:
                with open(out_path, "a", encoding="utf-8") as f:
                    for pw in items:
                        line = json.dumps(pw.record, ensure_ascii=False, separators=(",", ":"))
                        f.write(line + "\n")
                        wrote += 1
                        last_ref = pw.data_ref
                        last_dk = pw.dedupe_key
                        self._last_fingerprint[drawer] = self._record_fingerprint(pw.record)
                events.append(
                    Event(
                        topic="drawer/done",
                        payload={
                            "drawer": drawer,
                            "count": wrote,
                            "last_data_ref": last_ref,
                            "last_dedupe_key": last_dk,
                            "note": "write_committed",
                        },
                        timestamp=now,
                        source=self.name,
                        correlation_id=correlation_id or uuid.uuid4().hex,
                        meta={"reason": reason},
                    )
                )
                any_success = True
            except OSError as e:
                err = str(e)
                self.debug("flush_failed", drawer=drawer, error=err)
                events.append(
                    Event(
                        topic="drawer/error",
                        payload={
                            "drawer": drawer,
                            "error": err,
                            "path": out_path,
                            "count": len(items),
                            "note": "write_failed",
                        },
                        timestamp=now,
                        source=self.name,
                        correlation_id=correlation_id or uuid.uuid4().hex,
                        meta={"reason": reason},
                    )
                )
                remaining.extend(items)
                continue

        # keep only items that failed to write
        self._buf = remaining
        if any_success:
            self._last_flush = now
        return events

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        # -----------------------------
        # Periodic flush on body/service/maintenance
        # -----------------------------
        if event.topic == SERVICE_TOPIC:
            now = time.time()
            if self._buf and (now - self._last_flush) >= float(self.flush_interval_s):
                return self._flush(correlation_id=event.correlation_id, reason=SERVICE_TOPIC)
            return []

        if event.topic != "drawer/write":
            return []
        
        payload = event.payload if isinstance(event.payload, dict) else {}
        drawer = str(payload.get("drawer", "") or "").strip()
        data_ref = str(payload.get("data_ref", "") or "").strip() or None
        dedupe_key = str(payload.get("dedupe_key", "") or "").strip() or None
        record = payload.get("record", None)

        if not drawer or record is None:
            await ctx.log_debug(
                f"[{self.name}] Missing drawer/record; skipping",
                topic=event.topic,
            )
            return []
        
        try:
            json.dumps(record, ensure_ascii=False)
        except TypeError:
            await ctx.log_warn(f"[{self.name}] Record not JSON-serializable; skipping", topic=event.topic)
            return []

        # Buffer it (dedupe + cheap batching)
        enq_ok = self._enqueue(drawer=drawer, record=record, data_ref=data_ref, dedupe_key=dedupe_key)
        if not enq_ok:
            return []

        now = time.time()
        should_flush = (
            len(self._buf) >= int(self.max_buffer)
            or (now - self._last_flush) >= float(self.flush_interval_s)
        )
        
        if should_flush:
            return self._flush(correlation_id=event.correlation_id, reason="threshold")

        return []
        
def build_neurons(orchestrator) -> Iterable[BaseNeuron]:
    cfg = getattr(orchestrator, "cfg", None)
    memdir = getattr(cfg, "memdir", None) or os.environ.get("MB_MEMDIR") or r"Z:\memory"

    config = NeuronConfig(
        name="drawer_writer",
        subscribed_topics=["drawer/write", SERVICE_TOPIC],
        output_topics=["drawer/done", "drawer/error"],
        priority=5,
        cooldown_sec=0.0,
    )

    return [
        DrawerWriterNeuron(
            config=config,
            memdir=memdir,
            flush_interval_s=float(getattr(cfg, "drawer_flush_every_s", 0.75) if cfg else 0.75),
            max_buffer=int(getattr(cfg, "drawer_max_buffer", 8) if cfg else 8),
            dedupe_ttl_s=float(getattr(cfg, "drawer_dedupe_ttl_s", 20.0) if cfg else 20.0),
        )
    ]
