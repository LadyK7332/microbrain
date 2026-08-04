from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.patterns.pattern_edge_log import PatternEdgeLog
from microbrain.utils.memdir import resolve_memdir_ctx

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("maintenance")


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


class EntropyNeuron(BaseNeuron):
    """
    Sleep-only entropy / pruning for PatternEdgeLog.

    HARD RULE:
      - Does nothing unless KV `entropy:allowed == True`
        (set by your battery/sleep gate: charging AND sleep)

    Strategy:
      - Keep top-K edges per src (per edge_type)
      - For the rest:
          - if w < prune_floor: zero it (delta = -w)
          - else: decay it a bit (delta = -(w * (1-decay)))
      - Bounded work per run: max_updates

    Notes:
      - This is designed to be cheap + incremental.
      - synapses.jsonl stays append-only; we add NEGATIVE deltas to decay/prune.
    """
    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._edges: PatternEdgeLog | None = None

    async def _ensure_edges(self, ctx) -> PatternEdgeLog | None:
        # Reuse shared edge log if available
        if self._edges is not None:
            return self._edges

        shared = await ctx.get_kv("patterns:edges", None)
        if isinstance(shared, PatternEdgeLog):
            self._edges = shared
            return self._edges

        # Sleep mode may not boot PatternBinderNeuron; create our own.
        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")

        # Publish so other neurons can reuse it
        await ctx.set_kv("patterns:edges", self._edges)
        return self._edges
    
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != SERVICE_TOPIC:
            return []

        allowed = bool(await ctx.get_kv("entropy:allowed", False))
        if not allowed:
            return []

        now = time.time()

        run_every_s = _safe_float(await ctx.get_kv("entropy:run_every_s", 10.0), 10.0)
        last_run = _safe_float(await self.load_state(ctx, "last_run_ts", 0.0), 0.0)
        if last_run and (now - last_run) < run_every_s:
            return []

        edges = await self._ensure_edges(ctx)
        if edges is None:
            return []
        
        # --- knobs ---
        edge_types = await ctx.get_kv("entropy:edge_types", None)
        if not isinstance(edge_types, list):
            edge_types = [
                "token_concept",
                "concept_token",
                "noun_concept",
                "concept_noun",

                # grounded perception bindings
                "sense_concept",
                "concept_sense",

                # atom/structure edges (new)
                "ent_isa",
                "isa_ent",
                "ent_prop",
                "prop_ent",
                "concept_isa",
                "concept_sub",
                "prop_attr",
                "attr_prop",
                "prop_value",
                "value_prop",

                # outcomes/safety
                "concept_outcome",
                "outcome_concept",
            ]
            
        top_k = _safe_int(await ctx.get_kv("entropy:top_k_per_src", 80), 80)
        max_updates = _safe_int(await ctx.get_kv("entropy:max_updates", 200), 200)

        # Priority tiers (what decays fastest vs slowest)
        prune_floor = _safe_float(await ctx.get_kv("entropy:prune_floor", 0.12), 0.12)
        decay_factor = _safe_float(await ctx.get_kv("entropy:decay_factor", 0.985), 0.985)

        lexical_edge_types = await ctx.get_kv("entropy:lexical_edge_types", None)
        if not isinstance(lexical_edge_types, list):
            lexical_edge_types = ["token_concept", "concept_token", "noun_concept", "concept_noun"]

        grounded_edge_types = await ctx.get_kv("entropy:grounded_edge_types", None)
        if not isinstance(grounded_edge_types, list):
            grounded_edge_types = [
                "sense_concept",
                "concept_sense",

                # atom/structure edges (should persist longer than pure lexical)
                "ent_isa",
                "isa_ent",
                "ent_prop",
                "prop_ent",
                "concept_isa",
                "concept_sub",
                "prop_attr",
                "attr_prop",
                "prop_value",
                "value_prop",
            ]
            
        top_k_lexical = _safe_int(await ctx.get_kv("entropy:top_k_lexical", 50), 50)
        top_k_grounded = _safe_int(await ctx.get_kv("entropy:top_k_grounded", 120), 120)

        lexical_prune_floor = _safe_float(await ctx.get_kv("entropy:lexical_prune_floor", 0.18), 0.18)
        grounded_prune_floor = _safe_float(await ctx.get_kv("entropy:grounded_prune_floor", 0.08), 0.08)

        lexical_decay_factor = _safe_float(await ctx.get_kv("entropy:lexical_decay_factor", 0.97), 0.97)
        grounded_decay_factor = _safe_float(await ctx.get_kv("entropy:grounded_decay_factor", 0.99), 0.99)

        # Protected: outcomes + self (and other "core identity / safety" anchors)
        protected_prefixes = await ctx.get_kv("entropy:protected_prefixes", None)
        if not isinstance(protected_prefixes, list):
            protected_prefixes = ["outcome:", "concept:hazard", "concept:self"]

        protected_edge_types = await ctx.get_kv("entropy:protected_edge_types", None)
        if not isinstance(protected_edge_types, list):
            protected_edge_types = ["concept_outcome", "outcome_concept"]

        protected_decay_factor = _safe_float(await ctx.get_kv("entropy:protected_decay_factor", 0.995), 0.995)

        # Collect current weights (PatternEdgeLog stores them in-memory)
        W = getattr(edges, "_W", None)

        if not isinstance(W, dict) or not W:
            await self.save_state(ctx, "last_run_ts", now)
            return []

        # Build per-src buckets, per edge_type
        buckets: Dict[Tuple[str, str], List[Tuple[Any, float]]] = {}
        items: List[Tuple[Any, float]] = []

        for k, w in W.items():
            try:
                et = str(getattr(k, "edge_type"))
                src = str(getattr(k, "src"))
                dst = str(getattr(k, "dst"))
            except Exception:
                continue

            if et not in edge_types:
                continue

            fw = float(w)
            items.append((k, fw))
            buckets.setdefault((et, src), []).append((k, fw))

        # Keep set = union of top-K per (edge_type, src) with tier-aware budgets
        keep: set[Any] = set()
        for (et, src), lst in buckets.items():
            lst.sort(key=lambda t: t[1], reverse=True)

            k_keep = top_k
            if et in lexical_edge_types:
                k_keep = top_k_lexical
            elif et in grounded_edge_types:
                k_keep = top_k_grounded

            for k, _w in lst[:k_keep]:
                keep.add(k)

        # Candidates = not kept (weak/unimportant edges)
        candidates: List[Tuple[Any, float]] = [(k, w) for (k, w) in items if k not in keep]
        candidates.sort(key=lambda t: t[1])  # weakest first

        # Early on, graphs are tiny; top-K may keep everything.
        # Still apply gentle decay so entropy is observable and prevents unbounded growth.
        decay_factor_fallback = None
        if not candidates and items:
            candidates = sorted(items, key=lambda t: t[1])
            decay_factor_fallback = _safe_float(await ctx.get_kv("entropy:decay_factor_fallback", 0.98), 0.98)

        updates = 0
        pruned = 0
        decayed = 0

        ts = now
        for k, w in candidates:
            if updates >= max_updates:
                break

            et = str(getattr(k, "edge_type"))
            src = str(getattr(k, "src"))
            dst = str(getattr(k, "dst"))

            if w <= 0.0:
                continue

            is_protected = (et in protected_edge_types) or any(
                src.startswith(p) or dst.startswith(p) for p in protected_prefixes
            )

            # Tier selection
            if is_protected:
                tier = "protected"
                local_prune_floor = 0.0
                factor = float(protected_decay_factor)
            elif et in grounded_edge_types:
                tier = "grounded"
                local_prune_floor = float(grounded_prune_floor)
                factor = float(grounded_decay_factor)
            elif et in lexical_edge_types:
                tier = "lexical"
                local_prune_floor = float(lexical_prune_floor)
                factor = float(lexical_decay_factor)
            else:
                tier = "default"
                local_prune_floor = float(prune_floor)
                factor = float(decay_factor_fallback if decay_factor_fallback is not None else decay_factor)

            # Prune/decay
            if (w < local_prune_floor) and (not is_protected):
                delta = -w
                new_w = 0.0
                pruned += 1
                meta = {"kind": "entropy_prune", "old_w": w, "new_w": new_w, "factor": factor, "protected": False, "tier": tier}
            else:
                new_w = w * factor

                delta = new_w - w  # negative                
                # Skip vanishingly small changes to avoid log spam
                if abs(delta) < 1e-6:
                    continue
                decayed += 1
                meta = {"kind": "entropy_decay", "old_w": w, "new_w": new_w, "factor": factor, "protected": is_protected, "tier": tier}
                                                
            edges.add(et, src, dst, float(delta), role="system", channel="entropy", ts=ts, meta=meta)
            updates += 1

        await self.save_state(ctx, "last_run_ts", now)
        await ctx.set_kv(
            "entropy:last_report",
            {"ts": now, "updates": updates, "pruned": pruned, "decayed": decayed, "allowed": True},
        )

        # No speech output; entropy is silent by design.
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[SERVICE_TOPIC],
        output_topics=[],
        priority=9,       # late; after learning edges were written
        cooldown_sec=0.0,  # quiet: use internal run_every_s instead of base cooldown spam
    )
    yield EntropyNeuron(cfg)
