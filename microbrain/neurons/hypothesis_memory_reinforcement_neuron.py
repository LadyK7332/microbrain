from __future__ import annotations

"""Assign memory and connection credit to hypothesis work.

The reinforcement direction is intentionally asymmetric:

* Querying a cell gives it a small accessibility touch.
* Only directly traversed, one-hop neighbors receive a smaller positive touch.
* Cells and patterns used by the selected action receive stronger direct credit.
* Positive observed outcomes add durable success/promotion credit.
* Negative outcomes weaken only the exact evidence/action/outcome route.  They do
  not diffuse negative credit into neighboring memory.

Raw hypotheses remain ephemeral.  This neuron modifies the reusable cells and
stable connection strings that participated in the thought.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.patterns.pattern_edge_log import PatternEdgeLog
from microbrain.utils.memdir import resolve_memdir_ctx

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------
# Keep these values together.  They are deliberately named and bounded so the
# user can tune memory stickiness without hunting for magic numbers in logic.

# A cell was directly returned by a memory query.
RETRIEVAL_ACTIVATION_DELTA = 0.003
RETRIEVAL_COUNT_INCREMENT = 1

# A one-hop explicit neighbor was traversed from a queried cell.
NEIGHBOR_ACTIVATION_DELTA = 0.001
NEIGHBOR_ASSOCIATION_INCREMENT = 1
NEIGHBOR_MAX_HOPS = 1
NEIGHBOR_MAX_PER_ROOT = 6
NEIGHBOR_CYCLE_ACTIVATION_CAP = 0.008

# Evidence helped select the hypothesis action, but has not yet been proven.
DECISION_USE_ACTIVATION_DELTA = 0.008
DECISION_USE_PROMOTION_DELTA = 0.0015
DECISION_USE_COUNT_INCREMENT = 1

# A memory cell directly supplied material to the final outward output.
DIRECT_OUTPUT_ACTIVATION_DELTA = 0.015
DIRECT_OUTPUT_PROMOTION_DELTA = 0.004
DIRECT_OUTPUT_USE_COUNT_INCREMENT = 1

# The selected action/output later received a positive observed outcome.
SUCCESS_ACTIVATION_DELTA = 0.025
SUCCESS_PROMOTION_DELTA = 0.008
SUCCESS_COUNT_INCREMENT = 1
SUCCESS_SCORE_THRESHOLD = 0.24

# Directly attributable failure.  Negative credit never diffuses to neighbors.
DIRECT_ROUTE_FAILURE_DELTA = -0.020
DIRECT_EVIDENCE_ACTION_FAILURE_DELTA = -0.012
DIRECT_CONTRADICTION_TRUST_DELTA = -0.030
FAILURE_SCORE_THRESHOLD = -0.24
FAILURE_COUNT_INCREMENT = 1

# Stable connection-string reinforcement.
QUERY_NEIGHBOR_EDGE_DELTA = 0.001
DECISION_EVIDENCE_ACTION_EDGE_DELTA = 0.008
DIRECT_OUTPUT_ACTION_EDGE_DELTA = 0.014
SUCCESS_EVIDENCE_OUTCOME_EDGE_DELTA = 0.018
SUCCESS_PATTERN_ACTION_EDGE_DELTA = 0.014
SUCCESS_ACTION_OUTCOME_EDGE_DELTA = 0.020

# Hard per-event limits prevent large recalls from creating runaway credit.
MAX_QUERY_TRACE_ITEMS = 18
MAX_DIRECT_EVIDENCE_PER_EVENT = 8
MAX_OUTPUT_CELLS_PER_EVENT = 12
MAX_PATTERN_REFS_PER_EVENT = 8
MAX_SEEN_EVENT_KEYS = 192
MAX_REINFORCEMENT_HISTORY = 64

# Minimum contribution floors stop a weak-but-selected item from receiving a
# vanishingly small touch while preserving the relative query/action score.
MIN_RETRIEVAL_SCORE_FACTOR = 0.25
MIN_DECISION_CONTRIBUTION = 0.35
MIN_PATTERN_CONFIDENCE = 0.25

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

NEURON_NAME = Path(__file__).stem

# Bus routes and fixed attribution markers. Changing these requires updating
# the producers/consumers that participate in the reinforcement protocol.
HYPOTHESIS_READY_TOPIC = "hypothesis/ready"
ACTION_COMMITTED_TOPIC = "hypothesis/action_committed"
SPEECH_TOPIC = "act/speech"
OUTCOME_TOPIC = "hypothesis/outcome"
MEMORY_FALLBACK_DIR = r"Z:\memory"

# Trust is changed only when an observer explicitly says memory itself was
# contradicted. General conversational failure must not lower factual trust.
DIRECT_MEMORY_CONTRADICTION_REASONS = {"explicit_memory_contradiction"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cell_node(cell_id: str) -> str:
    value = str(cell_id or "").strip()
    return value if value.startswith("cell:") else f"cell:{value}"


def _unique_strings(values: Sequence[Any], limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


class HypothesisMemoryReinforcementNeuron(BaseNeuron):
    """Warm queried memory and assign bounded direct outcome credit."""

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._store: MemCellStore | None = None
        self._edges: PatternEdgeLog | None = None

    async def _ensure_resources(self, ctx) -> tuple[MemCellStore | None, PatternEdgeLog | None]:
        if self._store is None:
            shared = await ctx.get_kv("memory:mem_cell_store", None)
            if isinstance(shared, MemCellStore):
                self._store = shared
            else:
                memdir = await resolve_memdir_ctx(ctx, fallback=MEMORY_FALLBACK_DIR)
                self._store = MemCellStore(memdir)
                await ctx.set_kv("memory:mem_cell_store", self._store)

        if self._edges is None:
            memdir = await resolve_memdir_ctx(ctx, fallback=MEMORY_FALLBACK_DIR)
            self._edges = PatternEdgeLog(memdir)
        return self._store, self._edges

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        store, edges = await self._ensure_resources(ctx)
        if store is None or edges is None:
            return []

        if event.topic == HYPOTHESIS_READY_TOPIC:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
            hypothesis_id = str(hypothesis.get("hypothesis_id", "") or "")
            if not await self._claim_event(ctx, f"query|{hypothesis_id}"):
                return []
            stats = self._warm_query(store, edges, hypothesis)
            await self._publish_stats(ctx, "query", hypothesis_id, stats)
            return []

        if event.topic == ACTION_COMMITTED_TOPIC:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
            trigger = payload.get("trigger", {}) if isinstance(payload.get("trigger", {}), Mapping) else {}
            hypothesis_id = str(hypothesis.get("hypothesis_id", "") or "")
            action = str(trigger.get("recommended_action", hypothesis.get("recommended_action", "silence")) or "silence")
            if not await self._claim_event(ctx, f"decision|{hypothesis_id}|{action}"):
                return []
            stats = self._credit_decision(store, edges, hypothesis, action)
            await self._publish_stats(ctx, "decision", hypothesis_id, stats)
            return []

        if event.topic == SPEECH_TOPIC:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            ids = _unique_strings(
                list(payload.get("memory_cell_ids", (event.meta or {}).get("memory_cell_ids", [])) or []),
                MAX_OUTPUT_CELLS_PER_EVENT,
            )
            if not ids:
                return []
            pending = await ctx.get_kv("hypothesis:pending_outcome", {})
            pending = pending if isinstance(pending, Mapping) else {}
            hypothesis_id = str(pending.get("hypothesis_id", "") or event.correlation_id or "")
            action = str(pending.get("selected_action", "speak") or "speak")
            key = f"output|{hypothesis_id}|{'|'.join(ids)}"
            if not await self._claim_event(ctx, key):
                return []
            stats = self._credit_output_cells(store, edges, ids, action, hypothesis_id)
            await self._publish_stats(ctx, "output", hypothesis_id, stats)
            return []

        if event.topic == OUTCOME_TOPIC:
            outcome = event.payload if isinstance(event.payload, Mapping) else {}
            outcome_id = str(outcome.get("outcome_id", "") or "")
            hypothesis_id = str(outcome.get("hypothesis_id", "") or "")
            if not await self._claim_event(ctx, f"outcome|{outcome_id or hypothesis_id}"):
                return []
            stats = self._credit_outcome(store, edges, outcome)
            await self._publish_stats(ctx, "outcome", hypothesis_id, stats)
            return []

        return []

    def _warm_query(
        self,
        store: MemCellStore,
        edges: PatternEdgeLog,
        hypothesis: Mapping[str, Any],
    ) -> Dict[str, Any]:
        memory_check = hypothesis.get("memory_check", {}) if isinstance(hypothesis.get("memory_check", {}), Mapping) else {}
        trace = [dict(item) for item in list(memory_check.get("evidence_trace", []) or []) if isinstance(item, Mapping)]
        now = time.time()
        updates: List[Dict[str, Any]] = []
        direct_ids: List[str] = []
        neighbor_ids: List[str] = []
        neighbor_budget = NEIGHBOR_CYCLE_ACTIVATION_CAP

        for item in trace[:MAX_QUERY_TRACE_ITEMS]:
            cell_id = str(item.get("cell_id", "") or "").strip()
            if not cell_id:
                continue
            found = store.find_cell(cell_id, tier_hint=str(item.get("tier", "") or ""))
            if not found:
                continue
            tier = str(found.get("tier", "") or "")
            score_factor = max(MIN_RETRIEVAL_SCORE_FACTOR, _clamp01(_safe_float(item.get("retrieval_score", 0.0), 0.0)))
            updates.append(
                {
                    "cell_id": cell_id,
                    "tier": tier,
                    "retrieval_inc": RETRIEVAL_COUNT_INCREMENT,
                    "activation_delta": RETRIEVAL_ACTIVATION_DELTA * score_factor,
                    "last_retrieved_ts": now,
                    "meta": {"stage": "query", "hypothesis_id": hypothesis.get("hypothesis_id", "")},
                }
            )
            direct_ids.append(cell_id)

            if NEIGHBOR_MAX_HOPS < 1 or neighbor_budget <= 0.0:
                continue
            links = _unique_strings(list(item.get("links_explicit", []) or []), NEIGHBOR_MAX_PER_ROOT)
            for neighbor_id in links:
                if neighbor_id == cell_id or neighbor_id in neighbor_ids:
                    continue
                neighbor = store.find_cell(neighbor_id)
                if not neighbor:
                    continue
                delta = min(neighbor_budget, NEIGHBOR_ACTIVATION_DELTA * score_factor)
                if delta <= 0.0:
                    break
                updates.append(
                    {
                        "cell_id": neighbor_id,
                        "tier": str(neighbor.get("tier", "") or ""),
                        "association_inc": NEIGHBOR_ASSOCIATION_INCREMENT,
                        "activation_delta": delta,
                        "last_associated_ts": now,
                        "meta": {
                            "stage": "query_neighbor",
                            "hypothesis_id": hypothesis.get("hypothesis_id", ""),
                            "root_cell_id": cell_id,
                        },
                    }
                )
                edges.add(
                    "memory_neighbor_touch",
                    _cell_node(cell_id),
                    _cell_node(neighbor_id),
                    QUERY_NEIGHBOR_EDGE_DELTA * score_factor,
                    role="hypothesis_memory_reinforcement",
                    channel="memory",
                    meta={"stage": "query", "hypothesis_id": hypothesis.get("hypothesis_id", "")},
                )
                neighbor_ids.append(neighbor_id)
                neighbor_budget -= delta
                if neighbor_budget <= 0.0:
                    break

        staged = store.stage_reinforcements(updates)
        return {
            "staged": staged,
            "direct_query_cells": len(set(direct_ids)),
            "neighbor_cells": len(set(neighbor_ids)),
            "neighbor_activation_used": round(NEIGHBOR_CYCLE_ACTIVATION_CAP - max(0.0, neighbor_budget), 6),
        }

    def _credit_decision(
        self,
        store: MemCellStore,
        edges: PatternEdgeLog,
        hypothesis: Mapping[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        candidate = self._candidate_for(hypothesis, action)
        evidence_refs = [dict(item) for item in list(candidate.get("evidence_refs", []) or []) if isinstance(item, Mapping)]
        pattern_refs = [dict(item) for item in list(candidate.get("pattern_refs", []) or []) if isinstance(item, Mapping)]
        now = time.time()
        updates: List[Dict[str, Any]] = []
        used_ids: List[str] = []

        for item in evidence_refs[:MAX_DIRECT_EVIDENCE_PER_EVENT]:
            cell_id = str(item.get("cell_id", "") or "").strip()
            if not cell_id:
                continue
            found = store.find_cell(cell_id, tier_hint=str(item.get("tier", "") or ""))
            if not found:
                continue
            contribution = max(MIN_DECISION_CONTRIBUTION, _clamp01(_safe_float(item.get("score", 0.0), 0.0)))
            updates.append(
                {
                    "cell_id": cell_id,
                    "tier": str(found.get("tier", "") or ""),
                    "usage_inc": DECISION_USE_COUNT_INCREMENT,
                    "activation_delta": DECISION_USE_ACTIVATION_DELTA * contribution,
                    "promotion_delta": DECISION_USE_PROMOTION_DELTA * contribution,
                    "last_used_ts": now,
                    "meta": {"stage": "decision_use", "hypothesis_id": hypothesis.get("hypothesis_id", ""), "action": action},
                }
            )
            edges.add(
                "evidence_action",
                _cell_node(cell_id),
                f"action:{action}",
                DECISION_EVIDENCE_ACTION_EDGE_DELTA * contribution,
                role="hypothesis_memory_reinforcement",
                channel="memory",
                meta={"stage": "decision", "hypothesis_id": hypothesis.get("hypothesis_id", "")},
            )
            used_ids.append(cell_id)

        for item in pattern_refs[:MAX_PATTERN_REFS_PER_EVENT]:
            pattern = str(item.get("pattern", "") or "").strip()
            if not pattern:
                continue
            confidence = max(MIN_PATTERN_CONFIDENCE, _clamp01(_safe_float(item.get("confidence", 0.0), 0.0)))
            edges.add(
                "pattern_action",
                f"pattern:{pattern}",
                f"action:{action}",
                DECISION_EVIDENCE_ACTION_EDGE_DELTA * confidence,
                role="hypothesis_memory_reinforcement",
                channel="memory",
                meta={"stage": "decision", "hypothesis_id": hypothesis.get("hypothesis_id", "")},
            )

        staged = store.stage_reinforcements(updates)
        return {"staged": staged, "direct_decision_cells": len(set(used_ids)), "action": action}

    def _credit_output_cells(
        self,
        store: MemCellStore,
        edges: PatternEdgeLog,
        cell_ids: Sequence[str],
        action: str,
        hypothesis_id: str,
    ) -> Dict[str, Any]:
        now = time.time()
        updates: List[Dict[str, Any]] = []
        used: List[str] = []
        for cell_id in _unique_strings(cell_ids, MAX_OUTPUT_CELLS_PER_EVENT):
            found = store.find_cell(cell_id)
            if not found:
                continue
            updates.append(
                {
                    "cell_id": cell_id,
                    "tier": str(found.get("tier", "") or ""),
                    "usage_inc": DIRECT_OUTPUT_USE_COUNT_INCREMENT,
                    "activation_delta": DIRECT_OUTPUT_ACTIVATION_DELTA,
                    "promotion_delta": DIRECT_OUTPUT_PROMOTION_DELTA,
                    "last_used_ts": now,
                    "meta": {"stage": "direct_output", "hypothesis_id": hypothesis_id, "action": action},
                }
            )
            edges.add(
                "evidence_action",
                _cell_node(cell_id),
                f"action:{action}",
                DIRECT_OUTPUT_ACTION_EDGE_DELTA,
                role="hypothesis_memory_reinforcement",
                channel="memory",
                meta={"stage": "direct_output", "hypothesis_id": hypothesis_id},
            )
            used.append(cell_id)
        staged = store.stage_reinforcements(updates)
        return {"staged": staged, "direct_output_cells": len(set(used)), "action": action}

    def _credit_outcome(
        self,
        store: MemCellStore,
        edges: PatternEdgeLog,
        outcome: Mapping[str, Any],
    ) -> Dict[str, Any]:
        score = max(-1.0, min(1.0, _safe_float(outcome.get("score", 0.0), 0.0)))
        reliability = _clamp01(_safe_float(outcome.get("reliability", 0.0), 0.0))
        action = str(outcome.get("selected_action", "silence") or "silence")
        reason = str(outcome.get("reason", "") or "")
        outcome_status = str(outcome.get("status", "neutral") or "neutral")
        hypothesis_id = str(outcome.get("hypothesis_id", "") or "")
        outcome_id = str(outcome.get("outcome_id", "") or "")
        evidence_refs = [dict(item) for item in list(outcome.get("evidence_refs", []) or []) if isinstance(item, Mapping)]
        pattern_refs = [dict(item) for item in list(outcome.get("pattern_refs", []) or []) if isinstance(item, Mapping)]
        output_ids = _unique_strings(list(outcome.get("output_cell_ids", []) or []), MAX_OUTPUT_CELLS_PER_EVENT)
        evidence_ids = _unique_strings([item.get("cell_id", "") for item in evidence_refs], MAX_DIRECT_EVIDENCE_PER_EVENT)
        direct_ids = _unique_strings([*output_ids, *evidence_ids], MAX_OUTPUT_CELLS_PER_EVENT + MAX_DIRECT_EVIDENCE_PER_EVENT)
        now = time.time()
        updates: List[Dict[str, Any]] = []

        if score >= SUCCESS_SCORE_THRESHOLD and reliability > 0.0:
            credit = _clamp01(score) * reliability
            for cell_id in direct_ids:
                found = store.find_cell(cell_id)
                if not found:
                    continue
                updates.append(
                    {
                        "cell_id": cell_id,
                        "tier": str(found.get("tier", "") or ""),
                        "success_inc": SUCCESS_COUNT_INCREMENT,
                        "activation_delta": SUCCESS_ACTIVATION_DELTA * credit,
                        "promotion_delta": SUCCESS_PROMOTION_DELTA * credit,
                        "last_used_ts": now,
                        "meta": {
                            "stage": "outcome_success",
                            "hypothesis_id": hypothesis_id,
                            "outcome_id": outcome_id,
                            "score": score,
                            "reliability": reliability,
                        },
                    }
                )
                edges.add(
                    "evidence_outcome",
                    _cell_node(cell_id),
                    f"outcome:{outcome_status}",
                    SUCCESS_EVIDENCE_OUTCOME_EDGE_DELTA * credit,
                    role="hypothesis_memory_reinforcement",
                    channel="memory",
                    meta={"hypothesis_id": hypothesis_id, "outcome_id": outcome_id, "reason": reason},
                )

            for item in pattern_refs[:MAX_PATTERN_REFS_PER_EVENT]:
                pattern = str(item.get("pattern", "") or "").strip()
                if not pattern:
                    continue
                confidence = max(MIN_PATTERN_CONFIDENCE, _clamp01(_safe_float(item.get("confidence", 0.0), 0.0)))
                edges.add(
                    "pattern_action",
                    f"pattern:{pattern}",
                    f"action:{action}",
                    SUCCESS_PATTERN_ACTION_EDGE_DELTA * credit * confidence,
                    role="hypothesis_memory_reinforcement",
                    channel="memory",
                    meta={"stage": "outcome_success", "hypothesis_id": hypothesis_id, "outcome_id": outcome_id},
                )
            edges.add(
                "action_outcome",
                f"action:{action}",
                f"outcome:{outcome_status}",
                SUCCESS_ACTION_OUTCOME_EDGE_DELTA * credit,
                role="hypothesis_memory_reinforcement",
                channel="memory",
                meta={"hypothesis_id": hypothesis_id, "outcome_id": outcome_id, "reason": reason},
            )
            staged = store.stage_reinforcements(updates)
            return {"staged": staged, "positive_direct_cells": len(direct_ids), "negative_neighbors": 0}

        if score <= FAILURE_SCORE_THRESHOLD and reliability > 0.0:
            penalty = abs(score) * reliability
            for cell_id in direct_ids:
                found = store.find_cell(cell_id)
                if found:
                    update = {
                        "cell_id": cell_id,
                        "tier": str(found.get("tier", "") or ""),
                        "failure_inc": FAILURE_COUNT_INCREMENT,
                        "last_used_ts": now,
                        "meta": {
                            "stage": "direct_failure",
                            "hypothesis_id": hypothesis_id,
                            "outcome_id": outcome_id,
                            "reason": reason,
                        },
                    }
                    if reason in DIRECT_MEMORY_CONTRADICTION_REASONS:
                        update["trust_delta"] = DIRECT_CONTRADICTION_TRUST_DELTA * penalty
                    updates.append(update)
                edges.add(
                    "evidence_action",
                    _cell_node(cell_id),
                    f"action:{action}",
                    DIRECT_EVIDENCE_ACTION_FAILURE_DELTA * penalty,
                    role="hypothesis_memory_reinforcement",
                    channel="memory",
                    meta={"stage": "direct_failure", "hypothesis_id": hypothesis_id, "outcome_id": outcome_id, "reason": reason},
                )

            for item in pattern_refs[:MAX_PATTERN_REFS_PER_EVENT]:
                pattern = str(item.get("pattern", "") or "").strip()
                if not pattern:
                    continue
                edges.add(
                    "pattern_action",
                    f"pattern:{pattern}",
                    f"action:{action}",
                    DIRECT_ROUTE_FAILURE_DELTA * penalty,
                    role="hypothesis_memory_reinforcement",
                    channel="memory",
                    meta={"stage": "direct_failure", "hypothesis_id": hypothesis_id, "outcome_id": outcome_id, "reason": reason},
                )
            edges.add(
                "action_outcome",
                f"action:{action}",
                f"outcome:{outcome_status}",
                DIRECT_ROUTE_FAILURE_DELTA * penalty,
                role="hypothesis_memory_reinforcement",
                channel="memory",
                meta={"stage": "direct_failure", "hypothesis_id": hypothesis_id, "outcome_id": outcome_id, "reason": reason},
            )
            staged = store.stage_reinforcements(updates)
            return {
                "staged": staged,
                "negative_direct_cells": len(direct_ids),
                "negative_neighbors": 0,
                "trust_changed": reason in DIRECT_MEMORY_CONTRADICTION_REASONS,
            }

        return {"staged": 0, "neutral": True, "negative_neighbors": 0}

    def _candidate_for(self, hypothesis: Mapping[str, Any], action: str) -> Dict[str, Any]:
        for item in list(hypothesis.get("action_candidates", []) or []):
            if isinstance(item, Mapping) and str(item.get("action", "") or "") == action:
                return dict(item)
        return {}

    async def _claim_event(self, ctx, key: str) -> bool:
        key = str(key or "").strip()
        if not key:
            return True
        raw = await ctx.get_kv("hypothesis:memory_reinforcement_seen", [])
        seen = [str(item) for item in list(raw or [])] if isinstance(raw, list) else []
        if key in seen:
            return False
        seen.append(key)
        await ctx.set_kv("hypothesis:memory_reinforcement_seen", seen[-MAX_SEEN_EVENT_KEYS:])
        return True

    async def _publish_stats(self, ctx, stage: str, hypothesis_id: str, stats: Mapping[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "stage": stage,
            "hypothesis_id": hypothesis_id,
            **dict(stats),
        }
        await ctx.set_kv("hypothesis:last_memory_reinforcement", row)
        history = await ctx.get_kv("hypothesis:memory_reinforcement_history", [])
        history = list(history) if isinstance(history, list) else []
        history.append(row)
        await ctx.set_kv("hypothesis:memory_reinforcement_history", history[-MAX_REINFORCEMENT_HISTORY:])


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            HYPOTHESIS_READY_TOPIC,
            ACTION_COMMITTED_TOPIC,
            SPEECH_TOPIC,
            OUTCOME_TOPIC,
        ],
        output_topics=[],
        priority=25,
        cooldown_sec=0.0,
    )
    yield HypothesisMemoryReinforcementNeuron(cfg)
