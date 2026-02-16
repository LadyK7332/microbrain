from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from microbrain.memory.memory_store import JSONLStore


def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum((x * x) for x in v))


def _normalize(v: list[float]) -> list[float]:
    n = _l2_norm(v)
    if n <= 0.0:
        return v
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum((a[i] * b[i]) for i in range(n))


@dataclass
class ProtoConcept:
    proto_id: str
    centroid: list[float]
    n: int = 1
    last_seen_ts: float = 0.0


class ProtoConceptStore:
    """
    Online proto-concept clustering for a single modality (vision/audio/touch/...).

    File (append-only):
      memdir/proto_concepts.jsonl
        rows with kind="proto_update"
    """

    def __init__(self, memdir: str | Path, modality: str = "vision") -> None:
        self.memdir = Path(memdir)
        self.memdir.mkdir(parents=True, exist_ok=True)

        self.modality = str(modality or "vision")
        self._lock = threading.RLock()  # re-entrant: assign() -> best_match()
        self._log = JSONLStore(str(self.memdir / "proto_concepts.jsonl"))

        self._protos: dict[str, ProtoConcept] = {}
        self._next_id: int = 1

        # Load existing protos (small + simple; if this grows huge we’ll add compaction later)
        try:
            for row in self._log.read_all():
                if not isinstance(row, dict):
                    continue
                if row.get("kind") != "proto_update":
                    continue
                if row.get("modality") != self.modality:
                    continue

                pid = str(row.get("proto_id", "") or "").strip()
                cent = row.get("centroid", None)
                n = int(row.get("n", 0) or 0)
                ts = float(row.get("ts", 0.0) or 0.0)

                if not pid or not isinstance(cent, list) or n <= 0:
                    continue

                cent_f = [float(x) for x in cent]
                self._protos[pid] = ProtoConcept(proto_id=pid, centroid=cent_f, n=n, last_seen_ts=ts)

                # parse numeric suffix
                try:
                    suffix = pid.split(":")[-1]
                    num = int(suffix)
                    self._next_id = max(self._next_id, num + 1)
                except Exception:
                    pass
        except Exception:
            pass

    def _alloc_id(self) -> str:
        pid = f"proto:{self.modality}:{self._next_id:06d}"
        self._next_id += 1
        return pid

    def list_protos(self) -> list[ProtoConcept]:
        with self._lock:
            return list(self._protos.values())

    def best_match(self, vec: list[float]) -> tuple[Optional[ProtoConcept], float]:
        v = _normalize([float(x) for x in vec])
        best: Optional[ProtoConcept] = None
        best_sim = -1.0
        with self._lock:
            for p in self._protos.values():
                sim = _dot(v, p.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best = p
        return best, float(best_sim)

    def assign(
        self,
        vec: list[float],
        asset_id: str,
        channel: str = "default",
        ts: Optional[float] = None,
        thresh_attach: float = 0.86,
        alpha_ema: float = 0.10,
    ) -> tuple[str, float]:
        """
        Returns (proto_id, similarity_to_centroid_before_update).
        """
        ts = float(ts if ts is not None else time.time())
        v = _normalize([float(x) for x in vec])

        with self._lock:
            best, best_sim = self.best_match(v)
            if best is None or best_sim < float(thresh_attach):
                pid = self._alloc_id()
                p = ProtoConcept(proto_id=pid, centroid=v, n=1, last_seen_ts=ts)
                self._protos[pid] = p
                self._append(pid, v, 1, asset_id, best_sim, channel, ts, spawned=True)
                return pid, float(best_sim)

            # attach/update
            p = best
            old_cent = p.centroid
            # EMA-ish update (stable, drift tolerant)
            a = float(alpha_ema)
            new_cent = _normalize([(1.0 - a) * old_cent[i] + a * v[i] for i in range(min(len(old_cent), len(v)))])
            p.centroid = new_cent
            p.n = int(p.n) + 1
            p.last_seen_ts = ts

            self._append(p.proto_id, new_cent, p.n, asset_id, best_sim, channel, ts, spawned=False)
            return p.proto_id, float(best_sim)

    def _append(
        self,
        proto_id: str,
        centroid: list[float],
        n: int,
        asset_id: str,
        sim: float,
        channel: str,
        ts: float,
        spawned: bool,
    ) -> None:
        row: dict[str, Any] = {
            "kind": "proto_update",
            "schema_ver": 1,
            "modality": self.modality,
            "proto_id": proto_id,
            "centroid": centroid,
            "n": int(n),
            "asset_id": str(asset_id or ""),
            "sim": float(sim),
            "spawned": bool(spawned),
            "channel": str(channel or "default"),
            "ts": float(ts),
        }
        self._log.append(row)
