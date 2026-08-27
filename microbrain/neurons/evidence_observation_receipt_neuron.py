from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.evidence.evidence_observation_receipt import (
    build_evidence_observation_receipt,
    build_memcell_for_evidence_receipt,
    should_stage_observation_receipt,
    tier_for_observation_receipt,
)
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
INPUT_TOPIC = "evidence/observation"
OUTPUT_TOPIC = "memory/evidence_receipt"


class EvidenceObservationReceiptNeuron(BaseNeuron):
    """Stage tiny receipts for opened evidence windows, never raw samples."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        receipt = build_evidence_observation_receipt(event.payload, event_meta=event.meta)
        stage = should_stage_observation_receipt(event.payload)
        tier = tier_for_observation_receipt(event.payload)
        staged_count = 0
        stage_error = ""

        if stage:
            try:
                base_dir = await self._memory_base_dir(ctx)
                row = build_memcell_for_evidence_receipt(receipt)
                staged_count = MemCellStore(base_dir).stage_cells([row], tier=tier, touch=True)
            except Exception as exc:  # pragma: no cover - defensive runtime receipt only
                stage_error = f"{type(exc).__name__}: {exc}"

        payload = dict(receipt)
        payload["staged"] = bool(stage and staged_count > 0 and not stage_error)
        payload["staged_count"] = int(staged_count)
        payload["tier"] = tier
        if stage_error:
            payload["stage_error"] = stage_error

        return [
            Event(
                topic=OUTPUT_TOPIC,
                payload=payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "evidence_observation_receipt",
                    "cognitive_visible": True,
                    "store_in_memory": False,
                    "raw_payload_policy": "receipt_only_no_sample_items",
                    "staged": bool(payload.get("staged", False)),
                    "tier": tier,
                    "trigger_topic": receipt.get("trigger_topic", ""),
                    "route_reason": receipt.get("route_reason", ""),
                },
            )
        ]

    async def _memory_base_dir(self, ctx) -> Path:
        for key in (
            "memory:base_dir",
            "memory:memdir",
            "memdir",
            "paths:memory",
            "config:memdir",
        ):
            try:
                value = await ctx.get_kv(key, None)
            except Exception:
                value = None
            if isinstance(value, Mapping):
                for subkey in ("base_dir", "memdir", "path", "root"):
                    candidate = value.get(subkey)
                    if candidate:
                        return Path(candidate)
            elif value:
                return Path(str(value))
        env_value = os.getenv("MB_MEMDIR") or os.getenv("MICROBRAIN_MEMDIR")
        if env_value:
            return Path(env_value)
        return Path("memory")


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[INPUT_TOPIC],
        output_topics=[OUTPUT_TOPIC],
        priority=1,
    )
    return [EvidenceObservationReceiptNeuron(cfg)]
