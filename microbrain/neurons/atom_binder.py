from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.memory.memory_store import JSONLStore
from microbrain.patterns.pattern_edge_log import PatternEdgeLog


NEURON_NAME = Path(__file__).stem


def _sha16(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _short(x: str) -> str:
    # "attr:color" -> "color", "value:red" -> "red"
    s = str(x or "").strip()
    if ":" in s:
        return s.split(":", 1)[1]
    return s


class AtomBinderNeuron(BaseNeuron):
    """
    Persists 'memory/atom' events and converts them into lightweight pattern edges.

    Writes:
      - memdir/atoms.jsonl  (append-only)
      - memdir/synapses.jsonl via PatternEdgeLog (append-only edges)

    Anti-bloat:
      - Only writes an atom row the FIRST time we see an identical atom_key.
      - Still reinforces edges each time (frequency lives in synapses, not atoms.jsonl).
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._atoms_log: Optional[JSONLStore] = None
        self._edges: Optional[PatternEdgeLog] = None
        self._mem_store = None
        self._memdir: Optional[Path] = None
        self._seen: Set[str] = set()
        self._loaded_seen: bool = False

    async def _ensure_ready(self, ctx) -> bool:
        if self._atoms_log is not None and self._edges is not None and self._memdir is not None:
            return True

        mem_store = await ctx.get_kv("memory:store", None)
        if mem_store is None:
            return False

        # Prefer base_dir (always valid), not memdir (may be None)
        base_dir = getattr(mem_store, "base_dir", None)
        if base_dir is None:
            return False

        memdir = Path(str(base_dir))
        memdir.mkdir(parents=True, exist_ok=True)

        # Init logs
        self._mem_store = mem_store
        self._memdir = memdir
        self._atoms_log = JSONLStore(str(memdir / "atoms.jsonl"))

        # Reuse shared PatternEdgeLog if present, else create
        shared = await ctx.get_kv("patterns:edges", None)
        if isinstance(shared, PatternEdgeLog):
            self._edges = shared
        else:
            self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")
            await ctx.set_kv("patterns:edges", self._edges)

        # Load seen atom keys once (keeps atoms.jsonl sparse across restarts)
        if not self._loaded_seen:
            try:
                for row in self._atoms_log.read_all():
                    if isinstance(row, dict):
                        k = str(row.get("atom_key", "") or "")
                        if k:
                            self._seen.add(k)
            except Exception:
                pass
            self._loaded_seen = True

        return True

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "memory/atom":
            return []

        if not await self._ensure_ready(ctx):
            return []

        atom = event.payload if isinstance(event.payload, dict) else {}
        if not isinstance(atom, dict):
            return []

        schema = str(atom.get("schema", "") or "")
        if schema != "atom.v1":
            return []

        atom_type = str(atom.get("atom_type", "") or "")
        subj = str(atom.get("subj", "") or "").strip()
        channel = str(atom.get("channel", "default") or "default")
        source = str(atom.get("source", event.source or "unknown") or "unknown")
        ts = _safe_float(atom.get("ts", event.timestamp), float(time.time()))

        if not atom_type or not subj:
            return []

        # Reinforcement delta (user statements teach more than assistant restatements)
        roleish = source.lower()
        base_delta = 0.08 if "user" in roleish else 0.03

        # Build a stable atom_key (dedupe for atoms.jsonl)
        if atom_type == "isa":
            pred = str(atom.get("pred", "") or "").strip()
            if not pred:
                return []
            atom_key = _sha16("isa", subj, pred)
            await self._apply_isa_edges(subj=subj, pred=pred, delta=base_delta, channel=channel, ts=ts)
            row = dict(atom)
            row["atom_key"] = atom_key
            row["first_seen_ts"] = ts

        elif atom_type == "prop":
            attr = str(atom.get("attr", "") or "").strip()
            value = str(atom.get("value", "") or "").strip()
            if not attr or not value:
                return []
            atom_key = _sha16("prop", subj, attr, value)
            await self._apply_prop_edges(subj=subj, attr=attr, value=value, delta=base_delta, channel=channel, ts=ts)
            row = dict(atom)
            row["atom_key"] = atom_key
            row["first_seen_ts"] = ts

        else:
            return []

        # Write atom row only once (sparse)
        if atom_key not in self._seen:
            self._seen.add(atom_key)
            try:
                self._atoms_log.append(row)
            except Exception:
                pass

        # Expose last written atom for quick debug
        await ctx.set_kv("atoms:last_written", row)

        return []

    async def _apply_isa_edges(self, *, subj: str, pred: str, delta: float, channel: str, ts: float) -> None:
        assert self._edges is not None

        # Entity is-a concept
        if subj.startswith("ent:") and pred.startswith("concept:"):
            self._edges.add("ent_isa", subj, pred, delta, role="system", channel=channel, ts=ts)
            self._edges.add("isa_ent", pred, subj, delta, role="system", channel=channel, ts=ts)
            return

        # Concept is-a concept (taxonomy)
        if subj.startswith("concept:") and pred.startswith("concept:"):
            self._edges.add("concept_isa", subj, pred, delta, role="system", channel=channel, ts=ts)
            self._edges.add("concept_sub", pred, subj, delta, role="system", channel=channel, ts=ts)
            return

        # Fallback: still log something (keeps it debuggable)
        self._edges.add("isa", subj, pred, delta, role="system", channel=channel, ts=ts)

    async def _apply_prop_edges(self, *, subj: str, attr: str, value: str, delta: float, channel: str, ts: float) -> None:
        assert self._edges is not None

        # Build a compact property node: prop:<attr_short>:<value_short>
        prop_node = f"prop:{_short(attr)}:{_short(value)}"

        # entity -> prop
        self._edges.add("ent_prop", subj, prop_node, delta, role="system", channel=channel, ts=ts)
        self._edges.add("prop_ent", prop_node, subj, delta, role="system", channel=channel, ts=ts)

        # prop -> attr/value (helps later realization)
        self._edges.add("prop_attr", prop_node, attr, delta, role="system", channel=channel, ts=ts)
        self._edges.add("attr_prop", attr, prop_node, delta, role="system", channel=channel, ts=ts)

        self._edges.add("prop_value", prop_node, value, delta, role="system", channel=channel, ts=ts)
        self._edges.add("value_prop", value, prop_node, delta, role="system", channel=channel, ts=ts)


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["memory/atom"],
        output_topics=[],
        priority=7,
    )
    yield AtomBinderNeuron(cfg)