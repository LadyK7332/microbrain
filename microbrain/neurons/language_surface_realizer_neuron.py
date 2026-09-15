from __future__ import annotations

"""Review complete learned-surface candidates and hand them to the mouth."""

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
SLOT_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
VOBJ_RE = re.compile(r"\bvobj:[a-z0-9_.:-]+\b", re.IGNORECASE)
ORPHAN_FRAGMENTS = {"with", "to", "for", "from", "what is", "what are", "do you", "can you"}


def validate_surface_candidate(payload: Mapping[str, Any]) -> tuple[bool, str]:
    text = " ".join(str(payload.get("surface") or "").split())
    if str(payload.get("schema") or "") != "language.surface_candidate.v2":
        return False, "unsupported_surface_candidate_schema"
    if not text:
        return False, "empty_surface"
    if not bool(payload.get("surface_complete", False)):
        return False, "planner_marked_incomplete"
    if SLOT_RE.search(text):
        return False, "unresolved_slot"
    if VOBJ_RE.search(text):
        return False, "internal_visual_handle_leak"
    if "??" in text:
        return False, "double_question_placeholder"
    if text.lower().strip(" .?!") in ORPHAN_FRAGMENTS:
        return False, "orphan_fragment"
    required_act = str(payload.get("required_act") or "").strip().lower()
    if required_act == "clarify" and not text.endswith("?"):
        return False, "clarification_without_question_surface"
    return True, "complete_surface"


class LanguageSurfaceRealizerNeuron(BaseNeuron):
    async def process(self, event: Event, ctx: Any) -> Iterable[Event]:
        if event.topic != "language/surface_candidate":
            return []
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        allowed, reason = validate_surface_candidate(payload)
        if not allowed:
            await ctx.set_kv(
                "language:last_surface_rejection",
                {
                    "obligation_id": payload.get("obligation_id", ""),
                    "plan_id": payload.get("plan_id", ""),
                    "reason": reason,
                },
            )
            return []

        text = " ".join(str(payload.get("surface") or "").split())
        is_floor = bool(payload.get("surface_floor", False))
        floor_meta = {
            "surface_floor": is_floor,
            "surface_floor_policy": payload.get("surface_floor_policy", ""),
            "capability_mismatch": dict(payload.get("capability_mismatch", {}) or {}),
        }
        return [
            Event(
                topic="act/speech",
                payload={
                    "text": text,
                    "channel": "textual",
                    "style": "assistant",
                    "obligation_id": payload.get("obligation_id", ""),
                    "plan_id": payload.get("plan_id", ""),
                    "selected_structure_id": payload.get("selected_structure_id", ""),
                    "required_act": payload.get("required_act", ""),
                    "surface_status": payload.get("surface_status", ""),
                    **floor_meta,
                    "store_in_memory": False if is_floor else payload.get("store_in_memory", None),
                    "reinforcement_eligible": False if is_floor else payload.get("reinforcement_eligible", None),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "language_surface_floor_realization" if is_floor else "language_surface_realization",
                    "language_surface_v2": True,
                    "obligation_id": payload.get("obligation_id", ""),
                    "plan_id": payload.get("plan_id", ""),
                    "selected_structure_id": payload.get("selected_structure_id", ""),
                    "required_act": payload.get("required_act", ""),
                    "surface_status": payload.get("surface_status", ""),
                    **floor_meta,
                    "store_in_memory": False if is_floor else payload.get("store_in_memory", None),
                    "reinforcement_eligible": False if is_floor else payload.get("reinforcement_eligible", None),
                },
            )
        ]


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield LanguageSurfaceRealizerNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=["language/surface_candidate"],
            output_topics=["act/speech"],
            priority=1,
            cooldown_sec=0.0,
        )
    )
