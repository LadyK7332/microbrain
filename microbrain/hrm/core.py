from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Deque

import time
import math
from collections import deque

import numpy as np

from microbrain.memory.memory_store import JSONLStore, _local_embed


@dataclass
class HRMNode:
    """
    A single "concept" node in the HRM graph.

    - idx:   stable index into the synapse matrix
    - text:  surface text that led to this node
    - vec:   embedding vector (local hash embed for now; ONNX later)
    - role:  "user", "assistant", "system", etc.
    - ts:    timestamp when the node was created
    """

    idx: int
    text: str
    vec: List[float]
    role: str
    ts: float

    salience: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)


class HRMCore:
    """
    Hierarchical Recurrent Memory core (v1).

    This is the *concept + synapse* layer:

    - Maintains a fixed-size pool of concept nodes (N slots).
    - For each new observation, creates a node (or reuses a slot).
    - Applies simple Hebbian updates between the new node and
      a window of recent nodes ("what fires together wires together").
    - Logs all weight deltas to synapses.jsonl so you can analyze
      or visualize the graph offline.

    Higher-level "hierarchical" structures (clusters, regions, PDNA hooks)
    will sit *on top* of this core.
    """

    def __init__(
        self,
        memdir: str | Path,
        max_nodes: int = 1024,
        hebb_delta: float = 3.333333333333334e-05,
        recent_window: int = 8,
    ):
        memdir_path = Path(memdir)
        memdir_path.mkdir(parents=True, exist_ok=True)

        self.memdir = memdir_path
        self.max_nodes = max_nodes
        self.hebb_delta = float(hebb_delta)
        self.recent_window = int(recent_window)

        # Dense synapse weight matrix W[i, j] between concept indices.
        self.W = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)

        # Node table: idx -> HRMNode
        self.nodes: Dict[int, HRMNode] = {}

        # Small FIFO of recently active indices (temporal neighborhood)
        self.recent_indices: Deque[int] = deque(maxlen=self.recent_window)

        # Where we log synapse deltas (ts, i, j, delta)
        self.synapse_log = JSONLStore(str(self.memdir / "synapses.jsonl"))

        # Simple pointer for next free slot (wraps around)
        self._next_idx: int = 0

    # ------------------------------------------------------------------ #
    #  Core public API
    # ------------------------------------------------------------------ #

    def observe(
        self,
        text: str,
        role: str = "user",
        meta: Optional[Dict[str, Any]] = None,
        
    ) -> HRMNode:
        """
        Add a new observation into the HRM.

        - Embeds the text (local hash embed for now).
        - Allocates a node index.
        - Applies Hebbian updates vs recent nodes.
        - Logs synapse deltas to synapses.jsonl.

        Returns the created HRMNode.
        """
        ts = time.time()
        text = (text or "").strip()
        if not text:
            # Don't create empty nodes
            raise ValueError("HRM.observe called with empty text")

        vec = _local_embed(text)
        idx = self._allocate_index()

        node = HRMNode(
            idx=idx,
            text=text,
            vec=vec,
            role=role,
            ts=ts,
        )
        
        now_utc = datetime.utcnow()
        now_local = datetime.now()
        days_since_epoch = int(time.time() // 86400)

        node.created_at = now_utc.isoformat()
        node.day_index = days_since_epoch
        node.week_index = days_since_epoch // 7
        node.local_hour = now_local.hour
        node.local_weekday = now_local.weekday()  # 0=Monday
        
        self.nodes[idx] = node

        # Hebbian update: strengthen connections with recent nodes
        for j in list(self.recent_indices):
            if j == idx:
                continue
            self._apply_hebb(ts, idx, j, self.hebb_delta)

        # New node also "fires with itself" – diagonal trace
        self._apply_hebb(ts, idx, idx, self.hebb_delta)

        # Update temporal window
        self.recent_indices.append(idx)

        return node

    def get_node(self, idx: int) -> Optional[HRMNode]:
        return self.nodes.get(idx)

    def neighbors(self, idx: int, k: int = 5) -> List[tuple[int, float]]:
        """
        Return top-k neighbors by absolute synapse weight.

        Returns list of (neighbor_idx, weight), sorted by |weight| descending.
        """
        if idx < 0 or idx >= self.max_nodes:
            return []
        row = self.W[idx]
        # We'll ignore zero-weight entries for speed
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            return []
        pairs = [(int(j), float(row[j])) for j in nz]
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:k]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _allocate_index(self) -> int:
        """
        Round-robin allocator over [0, max_nodes).

        If we wrap and reuse an index, we overwrite the previous node in that slot.
        That acts like a form of forgetting / bounded memory.
        """
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self.max_nodes
        return idx

    def _apply_hebb(self, ts: float, i: int, j: int, delta: float) -> None:
        """
        Apply a weight delta to W[i, j] and W[j, i], and log it.

        This mirrors your existing synapses.jsonl format:
            {"ts": ..., "i": i, "j": j, "delta": delta}
        """
        if not (0 <= i < self.max_nodes and 0 <= j < self.max_nodes):
            return

        self.W[i, j] += delta
        if i != j:
            self.W[j, i] += delta

        self.synapse_log.append({"ts": ts, "i": i, "j": j, "delta": float(delta)})

    # ------------------------------------------------------------------ #
    #  Future hooks: hierarchical + PDNA integration
    # ------------------------------------------------------------------ #

    def set_affect(
        self,
        idx: int,
        salience: float | None = None,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> None:
        """
        Optional hook for affective neurons:
        let them annotate nodes with emotional state.
        """
        node = self.nodes.get(idx)
        if not node:
            return
        if salience is not None:
            node.salience = float(salience)
        if valence is not None:
            node.valence = float(valence)
        if arousal is not None:
            node.arousal = float(arousal)

    def tag(self, idx: int, **tags: Any) -> None:
        """
        Optional hook for relation/PDNA neurons:
        attach arbitrary tags ("work", "family", "kink", "safety", etc.).
        """
        node = self.nodes.get(idx)
        if not node:
            return
        node.tags.update(tags)
