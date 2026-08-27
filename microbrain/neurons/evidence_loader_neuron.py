from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.evidence.evidence_loader import load_evidence_reference
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

EVIDENCE_REQUEST_TOPICS = (
    "memory/evidence_request",
    "evidence/request",
    "hypothesis/evidence_request",
    "review/evidence_request",
)

# Context keys copied from the request into evidence/loaded.  The loader opens a
# bounded sample; this context tells the next organ which chain asked for proof
# without embedding the whole trigger event again.
EVIDENCE_LOADED_CONTEXT_KEYS = (
    "request_id",
    "trigger_topic",
    "trigger_source",
    "route_reason",
    "priority",
    "ref_card",
    "correlation_id",
)


def _payload_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"artifact_ref": payload}


def attach_request_context_to_loaded(
    loaded: Mapping[str, Any] | Any,
    request: Mapping[str, Any],
    *,
    request_topic: str,
    requested_by: str,
) -> dict[str, Any]:
    """Copy bounded request context onto an evidence-loaded envelope."""
    out = dict(loaded) if isinstance(loaded, Mapping) else {"ok": False, "error": "loaded payload was not a mapping", "value": loaded}
    out.setdefault("request_topic", request_topic)
    out.setdefault("requested_by", requested_by)
    out.setdefault("load_context_schema", "evidence.loaded_context.v1")
    for key in EVIDENCE_LOADED_CONTEXT_KEYS:
        if key in request and request.get(key) not in (None, "", [], {}):
            out.setdefault(key, request.get(key))
    return out


class EvidenceLoaderNeuron(BaseNeuron):
    """Open artifact evidence only when a deliberation path asks for it."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        request = _payload_mapping(event.payload)
        base_dir = await self._memory_base_dir(ctx)
        loaded = load_evidence_reference(base_dir, request)
        loaded = attach_request_context_to_loaded(
            loaded,
            request,
            request_topic=event.topic,
            requested_by=event.source,
        )

        return [
            Event(
                topic=str(request.get("emit_topic", "evidence/loaded") or "evidence/loaded"),
                payload=loaded,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "evidence_loaded",
                    "cognitive_visible": False,
                    "store_in_memory": False,
                    "raw_payload_policy": "bounded_deliberation_sample_only",
                    "mode": loaded.get("mode"),
                    "ok": bool(loaded.get("ok", False)),
                    "trigger_topic": loaded.get("trigger_topic", ""),
                    "route_reason": loaded.get("route_reason", ""),
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
        subscribed_topics=list(EVIDENCE_REQUEST_TOPICS),
        output_topics=["evidence/loaded"],
        priority=1,
    )
    return [EvidenceLoaderNeuron(cfg)]
