from __future__ import annotations

"""Neuron that remembers reading-derived sentence molds and proposes surfaces.

It does not speak directly.  It publishes inspectable surface plans/candidates
for a later mouth/speech realizer to review.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microbrain.language.surface_structure_memory import (
    LAST_SURFACE_CANDIDATE_KV_KEY,
    LAST_SURFACE_PLAN_KV_KEY,
    STORE_KV_KEY,
    build_surface_candidate_from_plan,
    build_surface_plan_for_gap,
    merge_structure_into_store,
    normalize_structure_candidate,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

try:  # Optional: present when Language Reference Grounding v1 is applied.
    from microbrain.language.quote_context import build_structure_candidate_from_quote
except Exception:  # pragma: no cover - fallback for partial installs
    build_structure_candidate_from_quote = None  # type: ignore[assignment]

NEURON_NAME = Path(__file__).stem

SUBSCRIBED_TOPICS = [
    "language/structure_candidate",
    "language/quote_context",
    "cognition/gap_identified",
    "speech/response_obligation",
]

OUTPUT_TOPICS = [
    "language/surface_structure",
    "language/surface_plan",
    "language/surface_candidate",
]

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

ENABLE_STRUCTURE_STORE = True
ENABLE_SURFACE_PLANNING = True
MAX_EVENTS_PER_INPUT = 6

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

STRUCTURE_INPUT_TOPICS = {"language/structure_candidate", "language/quote_context"}
GAP_INPUT_TOPICS = {"cognition/gap_identified", "speech/response_obligation"}


def _as_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {"surface": str(value or "")}


def _fallback_target_from_context(ctx) -> str:
    """Best-effort current handle lookup without forcing deep memory."""

    # This remains async-call friendly through the process method below.  The
    # function name is kept simple; actual awaits happen there.
    return ""


class LanguageSurfaceStructureNeuron(BaseNeuron):
    """Runtime bridge between learned sentence shapes and active gaps."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in SUBSCRIBED_TOPICS:
            return []
        payload = _as_payload(event.payload)
        outputs: list[Event] = []

        if ENABLE_STRUCTURE_STORE and event.topic in STRUCTURE_INPUT_TOPICS:
            candidate_payload: Mapping[str, Any] | None = payload
            if event.topic == "language/quote_context" and build_structure_candidate_from_quote is not None:
                built = build_structure_candidate_from_quote(payload)
                candidate_payload = built if isinstance(built, Mapping) else None
            if candidate_payload:
                structure = normalize_structure_candidate(candidate_payload)
                if structure:
                    current_store = await ctx.get_kv(STORE_KV_KEY, {})
                    next_store = merge_structure_into_store(current_store, structure)
                    await ctx.set_kv(STORE_KV_KEY, next_store)
                    outputs.append(
                        Event(
                            topic="language/surface_structure",
                            payload=structure,
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={
                                "kind": "language_surface_structure",
                                "structure_kind": structure.get("structure_kind", ""),
                                "store_in_memory": True,
                                "not_canned_response": True,
                                "truth_status": "structure_shape_not_answer_truth",
                            },
                        )
                    )

        if ENABLE_SURFACE_PLANNING and event.topic in GAP_INPUT_TOPICS:
            current_store = await ctx.get_kv(STORE_KV_KEY, {})
            fallback_target = ""
            for key in ("vision:focus", "object:last", "object:current_focus", "visual:current_focus"):
                value = await ctx.get_kv(key, "")
                if isinstance(value, str) and value:
                    fallback_target = value
                    break
                if isinstance(value, Mapping):
                    fallback_target = str(value.get("object_id") or value.get("id") or value.get("ref") or "")
                    if fallback_target:
                        break
            plan = build_surface_plan_for_gap(payload, current_store, fallback_target=fallback_target)
            candidate = build_surface_candidate_from_plan(plan)
            await ctx.set_kv(LAST_SURFACE_PLAN_KV_KEY, plan)
            await ctx.set_kv(LAST_SURFACE_CANDIDATE_KV_KEY, candidate)
            outputs.append(
                Event(
                    topic="language/surface_plan",
                    payload=plan,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "language_surface_plan",
                        "surface_status": plan.get("surface_status", ""),
                        "store_in_memory": False,
                        "not_canned_response": True,
                    },
                )
            )
            outputs.append(
                Event(
                    topic="language/surface_candidate",
                    payload=candidate,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "language_surface_candidate",
                        "surface_status": candidate.get("surface_status", ""),
                        "store_in_memory": False,
                        "not_canned_response": True,
                        "requires_review_by_mouth": True,
                    },
                )
            )
        return outputs[:MAX_EVENTS_PER_INPUT]


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=SUBSCRIBED_TOPICS,
        output_topics=OUTPUT_TOPICS,
        priority=2,
        cooldown_sec=0.0,
    )
    return [LanguageSurfaceStructureNeuron(cfg)]
