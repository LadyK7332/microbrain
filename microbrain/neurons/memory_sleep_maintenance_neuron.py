from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class MemorySleepMaintenanceNeuron(BaseNeuron):
    """
    Sleep-time memory maintenance pipeline.

    Order:
      1) prune/promote live tiers
      2) build compressed derived layer from survivors
      3) prune derived layer so compression itself does not bloat

    This keeps heavy consolidation out of the interaction path.
    """

    async def _mem_cell_store(self, ctx) -> MemCellStore:
        store = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(store, MemCellStore):
            return store
        memdir = await ctx.get_kv("app:memdir", None)
        store = MemCellStore(memdir)
        await ctx.set_kv("memory:mem_cell_store", store)
        return store

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "power/sleep_cycle":
            return []

        store = await self._mem_cell_store(ctx)

        retention = {
            "now": float(await ctx.get_kv("mem_cell:now_hours", 36.0) or 36.0),
            "short": float(await ctx.get_kv("mem_cell:short_hours", 72.0) or 72.0),
            "long": float(await ctx.get_kv("mem_cell:long_hours", 96.0) or 96.0),
            "learned": float(await ctx.get_kv("mem_cell:learned_hours", 336.0) or 336.0),
        }
        min_support = int(await ctx.get_kv("memory:compression:min_support_count", 2) or 2)
        min_encounters = int(await ctx.get_kv("memory:compression:min_encounter_sum", 3) or 3)
        derived_retention_h = float(await ctx.get_kv("memory:compression:derived_retention_h", 336.0) or 336.0)
        derived_max_rows = int(await ctx.get_kv("memory:compression:derived_max_rows", 512) or 512)

        prune_stats = store.maintain_lifecycle(retention_hours=retention)
        compress_stats = store.build_compressed_layer(
            source_tiers=("long", "learned"),
            min_support_count=min_support,
            min_encounter_sum=min_encounters,
        )
        derived_prune_stats = store.prune_derived_layer(
            retention_hours=derived_retention_h,
            max_rows=derived_max_rows,
        )

        snapshot = {
            "ts": time.time(),
            "prune": prune_stats,
            "compress": compress_stats,
            "derived_prune": derived_prune_stats,
        }
        await ctx.set_kv("memory:last_sleep_maintenance", snapshot)
        await ctx.log_info(
            f"[{self.name}] sleep maintenance completed",
            prune=prune_stats,
            compress=compress_stats,
            derived_prune=derived_prune_stats,
        )
        return []



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["power/sleep_cycle"],
        output_topics=[],
        priority=55,
        cooldown_sec=0.0,
    )
    yield MemorySleepMaintenanceNeuron(cfg)
