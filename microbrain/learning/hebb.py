# microbrain/learning/hebb.py
from __future__ import annotations

import heapq
import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microbrain.core.bus import Event, EventBus


def _topk_indices(vec: list[float], k: int) -> list[int]:
    if k <= 0 or k >= len(vec):
        return list(range(len(vec)))
    # indices of top-|value| elements
    return [i for _, i in heapq.nlargest(k, ((abs(v), i) for i, v in enumerate(vec)))]


def _outer_updates(pre: list[float], post: list[float], k: int) -> Iterable[tuple[int, int, float]]:
    pre_idx = _topk_indices(pre, k)
    post_idx = _topk_indices(post, k)
    for i in pre_idx:
        pi = pre[i]
        if pi == 0.0:
            continue
        for j in post_idx:
            pj = post[j]
            if pj == 0.0:
                continue
            yield i, j, pi * pj


@dataclass
class HebbianConfig:
    lr: float = 1e-3
    decay: float = 1e-5
    k: int = 8
    clip: float | None = 0.1  # abs clip on per-step delta


class HebbianLearner:
    """
    Online Hebbian updater:
        Δw_ij = η * pre_i * post_j - λ * w_ij

    - Sparse (top-k × top-k) to keep CPU/RAM light.
    - Writes JSONL rows into memdir/synapses.jsonl
    - Uses MemoryStore's embedder for vectors.
    """

    def __init__(
        self,
        memdir: str,
        embedder: Any,
        bus: EventBus,
        cfg: HebbianConfig | None = None,
    ) -> None:
        self.memdir = Path(memdir)
        self.embedder = embedder
        self.cfg = cfg or HebbianConfig()
        self.syn_path = self.memdir / "synapses.jsonl"
        self._W: dict[tuple[int, int], float] = {}
        self._lock = threading.Lock()

        # Subscribe to bus
        bus.subscribe("nl.input", self._on_user)
        bus.subscribe("llm.output", self._on_bot)

        # working buffer
        self._last_user_vec: list[float] | None = None

    # ---------- events ----------

    def _on_user(self, evt: Event) -> None:
        text = evt.payload.get("text")
        if not text:
            return
        try:
            self._last_user_vec = self._embed(text)
        except Exception as e:
            print(f"[hebb] user embed failed: {e}")

    def _on_bot(self, evt: Event) -> None:
        text = evt.payload.get("text")
        sal = float(evt.payload.get("salience", 1.0) or 1.0)
        if not text or self._last_user_vec is None:
            return
        try:
            post = self._embed(text)
        except Exception as e:
            print(f"[hebb] bot embed failed: {e}")
            return
        pre = self._last_user_vec
        self._learn(pre, post, salience=sal)

    # ---------- core ----------

    def _embed(self, text: str) -> list[float]:
        vec = self.embedder.embed(text)
        return [float(x) for x in vec]

    def _learn(self, pre: list[float], post: list[float], salience: float = 1.0) -> None:
        lr = self.cfg.lr * max(0.0, salience)
        decay = self.cfg.decay
        clip = self.cfg.clip
        k = self.cfg.k

        t = time.time()
        lines: list[str] = []

        with self._lock:
            for i, j, prod in _outer_updates(pre, post, k):
                key = (i, j)
                w = self._W.get(key, 0.0)
                delta = lr * prod - decay * w
                if clip is not None:
                    if delta > clip:
                        delta = clip
                    elif delta < -clip:
                        delta = -clip
                w += delta
                self._W[key] = w
                lines.append(json.dumps({"ts": t, "i": i, "j": j, "delta": delta}))

        if lines:
            self.syn_path.parent.mkdir(parents=True, exist_ok=True)
            with self.syn_path.open("a", encoding="utf-8") as f:
                for L in lines:
                    f.write(L + "\n")
