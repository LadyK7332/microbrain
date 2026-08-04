from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.utils.memdir import resolve_memdir_ctx

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("curiosity")


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


class LightProbeNeuron(BaseNeuron):
    """
    Keeps weak-but-interesting memory cells alive during the day, then lets
    sleep/charge windows run pruning and promotion.

    Daytime:
      - lightly probes weak cells so they can survive long enough to settle
      - emits a tiny thought/probe event for future introspection

    Sleep/charge:
      - no probing
      - runs lifecycle maintenance (promotion/pruning)
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._mem_cells: MemCellStore | None = None

    async def _ensure_store(self, ctx) -> MemCellStore | None:
        if self._mem_cells is not None:
            return self._mem_cells

        shared = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(shared, MemCellStore):
            self._mem_cells = shared
            return self._mem_cells

        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        self._mem_cells = MemCellStore(memdir)
        await ctx.set_kv("memory:mem_cell_store", self._mem_cells)
        return self._mem_cells

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != SERVICE_TOPIC:
            return []

        store = await self._ensure_store(ctx)
        if store is None:
            return []

        now_ts = time.time()
        sleep_mode = bool(await ctx.get_kv("power:sleep", False))
        entropy_allowed = bool(await ctx.get_kv("entropy:allowed", False))

        if sleep_mode or entropy_allowed:
            maintenance_every_s = _safe_float(await ctx.get_kv("probe:maintenance_every_s", 1800.0), 1800.0)
            last_maint = _safe_float(await self.load_state(ctx, "last_maint_ts", 0.0), 0.0)
            if last_maint and (now_ts - last_maint) < maintenance_every_s:
                return []

            retention = {
                "now": _safe_float(await ctx.get_kv("mem_cell:now_hours", 36.0), 36.0),
                "short": _safe_float(await ctx.get_kv("mem_cell:short_hours", 72.0), 72.0),
                "long": _safe_float(await ctx.get_kv("mem_cell:long_hours", 96.0), 96.0),
                "learned": _safe_float(await ctx.get_kv("mem_cell:learned_hours", 336.0), 336.0),
            }
            stats = store.maintain_lifecycle(retention_hours=retention)
            stats["ts"] = now_ts
            await ctx.set_kv("probe:last_maintenance", stats)
            await self.save_state(ctx, "last_maint_ts", now_ts)
            return []

        enabled = bool(await ctx.get_kv("probe:enabled", True))
        if not enabled:
            return []

        probe_every_s = _safe_float(await ctx.get_kv("probe:every_s", 300.0), 300.0)
        last_probe = _safe_float(await self.load_state(ctx, "last_probe_ts", 0.0), 0.0)
        if last_probe and (now_ts - last_probe) < probe_every_s:
            return []

        candidates = store.probe_candidates(limit=24, tiers=("now", "short", "long"))
        if not candidates:
            return []

        # weighted pick: newer / weaker / less settled cells get a little more air time
        weights = []
        for row in candidates:
            activation = max(0.05, min(1.0, float(row.get("activation", 0.2) or 0.2)))
            encounters = max(1.0, float(row.get("encounter_count", 1) or 1))
            novelty = max(0.10, 1.0 - min(encounters / 8.0, 1.0))
            weights.append((1.0 - activation) * 0.55 + novelty * 0.45)

        row = random.choices(candidates, weights=weights, k=1)[0]
        updated = store.bump_cell(
            str(row.get("id", "") or ""),
            activation_delta=_safe_float(await ctx.get_kv("probe:activation_delta", 0.03), 0.03),
            promotion_delta=_safe_float(await ctx.get_kv("probe:promotion_delta", 0.01), 0.01),
        )
        await self.save_state(ctx, "last_probe_ts", now_ts)

        info: Dict[str, Any] = {
            "cell_id": str((updated or row).get("id", "") or ""),
            "tier": str((updated or row).get("tier", "") or ""),
            "kind": str((updated or row).get("kind", "") or ""),
            "anchor": str((((updated or row).get("anchor", {}) or {}).get("ref", "") or ""))[:120],
            "activation": float((updated or row).get("activation", 0.0) or 0.0),
            "ts": now_ts,
        }
        await ctx.set_kv("probe:last", info)

        return [
            Event(
                topic="thought/probe",
                payload={"text": info["anchor"], "cell_id": info["cell_id"]},
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"channel": "thought", "kind": "light_probe", "quiet": True},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[SERVICE_TOPIC],
        output_topics=["thought/probe"],
        priority=1,
        cooldown_sec=0.0,
    )
    yield LightProbeNeuron(cfg)
