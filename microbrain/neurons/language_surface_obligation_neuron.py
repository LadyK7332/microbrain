from __future__ import annotations

"""Connect surface learning and explicit response obligations."""

import asyncio
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microbrain.language.surface_structure_memory_v2 import (
    LAST_CANDIDATE_KV_KEY,
    LAST_PLAN_KV_KEY,
    STORE_KV_KEY,
    build_surface_candidate,
    build_surface_plan,
    merge_structure,
    normalize_structure_candidate,
    structure_candidates_from_text,
    structure_from_mem_cell,
    structure_to_mem_cell,
)
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
SUBSCRIBED_TOPICS = [
    "language/structure_candidate",
    "language/quote_context",
    "language/parsed",
    "speech/response_obligation",
    "speech/delivered",
]
OUTPUT_TOPICS = [
    "language/surface_structure_v2",
    "language/surface_plan_v2",
    "language/surface_candidate",
    "language/expression_gap",
    "thought/self_question",
]
HYDRATED_KEY = "language:surface_structures_v2_hydrated"
CLAIMS_KEY = "language:surface_obligation_claims_v2"


class LanguageSurfaceObligationNeuron(BaseNeuron):
    async def _mem_store(self, ctx: Any) -> MemCellStore | None:
        value = await ctx.get_kv("memory:mem_cell_store", None)
        return value if isinstance(value, MemCellStore) else None

    async def _hydrate(self, ctx: Any) -> dict[str, dict[str, Any]]:
        current = await ctx.get_kv(STORE_KV_KEY, {})
        if bool(await ctx.get_kv(HYDRATED_KEY, False)):
            return dict(current) if isinstance(current, Mapping) else {}
        store = await self._mem_store(ctx)
        if store is not None:
            try:
                rows = await asyncio.to_thread(store.find_cells_by_kind, "surface_structure_v2", limit=256)
                for row in rows:
                    structure = structure_from_mem_cell(row)
                    if structure:
                        current = merge_structure(current, structure)
            except Exception as exc:
                await ctx.log_warn(f"[{self.name}] surface v2 hydration failed", error=repr(exc))
        await ctx.set_kv(STORE_KV_KEY, current)
        await ctx.set_kv(HYDRATED_KEY, True)
        return dict(current) if isinstance(current, Mapping) else {}

    async def _store_structure(self, ctx: Any, structure: Mapping[str, Any]) -> None:
        current = await self._hydrate(ctx)
        await ctx.set_kv(STORE_KV_KEY, merge_structure(current, structure))
        store = await self._mem_store(ctx)
        if store is not None:
            tier = "now" if str(structure.get("source", "")).startswith("reading") else "short"
            await asyncio.to_thread(store.upsert_cell, structure_to_mem_cell(structure, tier), tier)

    async def _reinforce(self, ctx: Any, payload: Mapping[str, Any]) -> None:
        structure_id = str(payload.get("selected_structure_id") or "")
        if not structure_id:
            return
        current = await ctx.get_kv(STORE_KV_KEY, {})
        if isinstance(current, Mapping) and isinstance(current.get(structure_id), Mapping):
            updated = {str(key): dict(value) for key, value in current.items() if isinstance(value, Mapping)}
            structure = dict(updated[structure_id])
            structure["use_count"] = int(structure.get("use_count", 0) or 0) + 1
            structure["last_used_ts"] = time.time()
            updated[structure_id] = structure
            await ctx.set_kv(STORE_KV_KEY, updated)
        store = await self._mem_store(ctx)
        if store is not None:
            await asyncio.to_thread(store.note_cell_usage, structure_id, success=True)

    async def process(self, event: Event, ctx: Any) -> Iterable[Event]:
        if event.topic not in SUBSCRIBED_TOPICS:
            return []
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        if event.topic == "speech/delivered":
            await self._reinforce(ctx, payload)
            return []

        outputs: list[Event] = []
        if event.topic in {"language/structure_candidate", "language/quote_context", "language/parsed"}:
            candidates: list[Mapping[str, Any]]
            if event.topic == "language/parsed":
                candidates = structure_candidates_from_text(
                    payload.get("text", ""),
                    source=f"live_parse:{payload.get('source', event.source)}",
                    evidence_ref=event.correlation_id,
                    confidence=0.52,
                )
            else:
                candidates = [payload]
            for candidate in candidates[:8]:
                structure = normalize_structure_candidate(candidate)
                if not structure:
                    continue
                await self._store_structure(ctx, structure)
                outputs.append(
                    Event(
                        topic="language/surface_structure_v2",
                        payload=structure,
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={"kind": "language_surface_structure_v2", "store_in_memory": True},
                    )
                )
            return outputs[:8]

        obligation_id = str(payload.get("obligation_id") or "")
        claims = await ctx.get_kv(CLAIMS_KEY, {})
        claims = dict(claims) if isinstance(claims, Mapping) else {}
        now_ts = time.time()
        claims = {
            str(key): dict(value)
            for key, value in claims.items()
            if isinstance(value, Mapping) and (now_ts - float(value.get("ts", 0.0) or 0.0)) <= 900.0
        }
        if len(claims) > 512:
            claims = dict(
                sorted(claims.items(), key=lambda item: float(item[1].get("ts", 0.0) or 0.0), reverse=True)[:512]
            )
        prior = claims.get(obligation_id) if obligation_id else None
        if isinstance(prior, Mapping) and bool(prior.get("surface_complete", False)):
            return []
        store = await self._hydrate(ctx)
        plan = build_surface_plan(payload, store)
        candidate = build_surface_candidate(plan)
        await ctx.set_kv(LAST_PLAN_KV_KEY, plan)
        await ctx.set_kv(LAST_CANDIDATE_KV_KEY, candidate)
        if obligation_id:
            claims[obligation_id] = {
                "plan_id": plan.get("plan_id", ""),
                "surface_complete": bool(candidate.get("surface_complete", False)),
                "ts": now_ts,
            }
            await ctx.set_kv(CLAIMS_KEY, claims)
        outputs.extend(
            [
                Event(
                    topic="language/surface_plan_v2",
                    payload=plan,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "language_surface_plan_v2", "store_in_memory": False},
                ),
                Event(
                    topic="language/surface_candidate",
                    payload=candidate,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "language_surface_candidate_v2",
                        "obligation_id": obligation_id,
                        "requires_review_by_mouth": True,
                    },
                ),
            ]
        )
        mismatch = candidate.get("capability_mismatch") if isinstance(candidate.get("capability_mismatch"), Mapping) else {}
        self_question = str(mismatch.get("self_question") or "").strip()
        if bool(candidate.get("surface_floor", False)) and self_question:
            outputs.append(
                Event(
                    topic="thought/self_question",
                    payload={
                        "schema": "thought.self_question.v1",
                        "obligation_id": obligation_id,
                        "plan_id": plan.get("plan_id", ""),
                        "question": self_question,
                        "reason": "capability_surface_mismatch",
                        "step_budget": int(mismatch.get("self_question_step_budget", 2) or 2),
                        "capability_mismatch": dict(mismatch),
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "capability_surface_mismatch",
                        "channel": "thought",
                        "store_in_memory": False,
                        "reinforcement_eligible": False,
                    },
                )
            )
        if bool(payload.get("required", False)) and not bool(candidate.get("surface_complete", False)):
            outputs.append(
                Event(
                    topic="language/expression_gap",
                    payload={
                        "schema": "language.expression_gap.v1",
                        "obligation_id": obligation_id,
                        "required_act": payload.get("required_act", ""),
                        "reason": "required_response_has_no_complete_compatible_surface",
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "language_expression_gap", "store_in_memory": True},
                )
            )
        return outputs


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield LanguageSurfaceObligationNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=SUBSCRIBED_TOPICS,
            output_topics=OUTPUT_TOPICS,
            priority=3,
            cooldown_sec=0.0,
        )
    )
