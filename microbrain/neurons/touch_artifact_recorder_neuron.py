from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.evidence.touch_artifact_recorder import (
    TOUCH_COMPACT_SCHEMA,
    record_touch_artifact,
    should_record_touch_payload,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

RAW_TOUCH_TOPICS = (
    "body/touch/raw",
    "sensor/touch/raw",
    "touch/raw",
    "percept/touch/raw",
)


def _payload_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


class TouchArtifactRecorderNeuron(BaseNeuron):
    """Persist raw-ish touch packets before they become object-frame evidence.

    Raw topics become compact ``percept/touch`` events. If an existing producer
    still emits raw-ish ``percept/touch`` directly, this neuron records the
    artifact and emits a side-channel compact packet, but it does not re-emit on
    ``percept/touch`` to avoid double object frames.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        data = _payload_mapping(event.payload)
        if not should_record_touch_payload(data):
            return []

        base_dir = await self._memory_base_dir(ctx)
        recorded = record_touch_artifact(
            base_dir,
            data,
            timestamp=float(event.timestamp),
            source=self.name,
        )
        compact_payload = dict(recorded.get("percept_payload", {}) or {})
        evidence_ref = dict(recorded.get("evidence_ref", {}) or {})
        card = dict(recorded.get("artifact_card", {}) or {})

        artifact_event = Event(
            topic="evidence/touch_artifact",
            payload={
                "schema": "touch.artifact_event.v1",
                "artifact_ref": str(card.get("artifact_ref", "") or ""),
                "evidence_ref": evidence_ref,
                "summary": str(compact_payload.get("summary", "") or ""),
                "claims_supported": list(compact_payload.get("claims_supported", []) or []),
                "confidence": compact_payload.get("confidence", 0.0),
                "source_topic": event.topic,
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "touch_artifact",
                "cognitive_visible": False,
                "store_in_memory": False,
                "raw_payload_policy": "artifact_written_reference_only",
            },
        )

        outputs: list[Event] = [artifact_event]
        if event.topic in RAW_TOUCH_TOPICS:
            outputs.append(
                Event(
                    topic="percept/touch",
                    payload=compact_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "percept",
                        "modality": "touch",
                        "normalized": True,
                        "raw_payload_policy": "artifact_written_reference_only",
                        "store_in_memory": True,
                    },
                )
            )
        else:
            outputs.append(
                Event(
                    topic="percept/touch/compact",
                    payload=compact_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "percept",
                        "modality": "touch",
                        "normalized": True,
                        "raw_payload_policy": "artifact_written_reference_only",
                        "store_in_memory": False,
                        "note": "side_channel_for_legacy_direct_percept_touch",
                    },
                )
            )
        return outputs

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
        subscribed_topics=[
            "body/touch/raw",
            "sensor/touch/raw",
            "touch/raw",
            "percept/touch/raw",
            "percept/touch",
        ],
        output_topics=["evidence/touch_artifact", "percept/touch", "percept/touch/compact"],
        priority=5,
    )
    return [TouchArtifactRecorderNeuron(cfg)]
