from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from microbrain.memory.memory_store import JSONLStore


@dataclass(frozen=True)
class EdgeKey:
    edge_type: str
    src: str
    dst: str


class PatternEdgeLog:
    """
    Append-only pattern edges (token↔concept, concept↔sense, concept↔frame, etc.)

    By default we write into memdir/synapses.jsonl so everything "connection-ish"
    lives in one place (HRM deltas + pattern edges).
    """

    def __init__(self, memdir: str | Path, filename: str = "synapses.jsonl") -> None:
        self.memdir = Path(memdir)
        self.memdir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._log = JSONLStore(str(self.memdir / filename))
        self._W: dict[EdgeKey, float] = {}

        # Load existing weights (only our rows)
        try:
            for row in self._log.read_all():
                if not isinstance(row, dict):
                    continue
                if row.get("kind") != "pattern_edge":
                    continue
                et = str(row.get("edge_type", "") or "")
                src = str(row.get("src", "") or "")
                dst = str(row.get("dst", "") or "")
                if not et or not src or not dst:
                    continue
                delta = float(row.get("delta", 0.0) or 0.0)
                k = EdgeKey(et, src, dst)
                self._W[k] = self._W.get(k, 0.0) + delta
        except Exception:
            pass

    def weight(self, edge_type: str, src: str, dst: str) -> float:
        k = EdgeKey(str(edge_type), str(src), str(dst))
        with self._lock:
            return float(self._W.get(k, 0.0))

    def add(
        self,
        edge_type: str,
        src: str,
        dst: str,
        delta: float,
        role: str = "system",
        channel: str = "default",
        ts: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> float:
        ts = float(ts if ts is not None else time.time())
        k = EdgeKey(str(edge_type), str(src), str(dst))
        d = float(delta)

        with self._lock:
            w = self._W.get(k, 0.0) + d
            self._W[k] = w
            row = {
                "kind": "pattern_edge",
                "edge_type": k.edge_type,
                "src": k.src,
                "dst": k.dst,
                "delta": d,
                "w": w,
                "role": role,
                "channel": channel,
                "ts": ts,
            }
            if meta:
                row["meta"] = meta
            self._log.append(row)

        return w
