from __future__ import annotations

"""
Sensory Fossil Organ neuron.

This neuron owns fossil storage/query events.  It does not declare beliefs; it
only turns modality-specific fossils into comparable evidence packets.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sensory_fossils import (
    DEFAULT_FOSSIL_MATCH_THRESHOLD,
    DEFAULT_MAX_MATCHES,
    EvidencePacket,
    SensoryFossilStore,
    clamp01,
)

NEURON_NAME = Path(__file__).stem
STORE_KEY = "sensory:fossil_store:v1"
LAST_STORE_KEY = "sensory:fossil:last_store"
LAST_QUERY_KEY = "sensory:fossil:last_query"

STORE_TOPIC = "sensory/fossil/store"
QUERY_TOPIC = "sensory/fossil/query"
VISION_QUERY_TOPIC = "vision/fossil/query"
TOUCH_QUERY_TOPIC = "touch/fossil/query"
AUDIO_QUERY_TOPIC = "audio/fossil/query"


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


class SensoryFossilNeuron(BaseNeuron):
    async def _load_store(self, ctx) -> SensoryFossilStore:
        snapshot = await ctx.get_kv(STORE_KEY, {})
        if isinstance(snapshot, SensoryFossilStore):
            return snapshot
        return SensoryFossilStore.from_snapshot(snapshot if isinstance(snapshot, Mapping) else {})

    async def _save_store(self, ctx, store: SensoryFossilStore) -> None:
        await ctx.set_kv(STORE_KEY, store.to_snapshot())

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        payload = _payload_dict(event.payload)
        store = await self._load_store(ctx)

        if event.topic == STORE_TOPIC:
            fossil = store.store_from_payload(payload)
            await self._save_store(ctx, store)
            info = {
                "fossil_id": fossil.fossil_id,
                "modality": fossil.modality,
                "concept": fossil.concept,
                "branch": fossil.branch,
                "tags": fossil.trailing_tags,
                "source_ref": fossil.source_ref,
            }
            await ctx.set_kv(LAST_STORE_KEY, info)
            return [
                Event(
                    topic="sensory/fossil/stored",
                    payload=info,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "event_class": "evidence",
                        "store_in_memory": False,
                        "semantic_input": False,
                        "meaning_boundary": "fossil_is_evidence_not_belief",
                    },
                )
            ]

        if event.topic in {QUERY_TOPIC, VISION_QUERY_TOPIC, TOUCH_QUERY_TOPIC, AUDIO_QUERY_TOPIC}:
            # Convenience modality inference for per-sense query topics.
            if "modality" not in payload:
                if event.topic == VISION_QUERY_TOPIC:
                    payload["modality"] = "vision"
                elif event.topic == TOUCH_QUERY_TOPIC:
                    payload["modality"] = "touch"
                elif event.topic == AUDIO_QUERY_TOPIC:
                    payload["modality"] = "audio"

            threshold = clamp01(
                await ctx.get_kv("sensory:fossil:match_threshold", DEFAULT_FOSSIL_MATCH_THRESHOLD),
                DEFAULT_FOSSIL_MATCH_THRESHOLD,
            )
            max_matches = int(await ctx.get_kv("sensory:fossil:max_matches", DEFAULT_MAX_MATCHES) or DEFAULT_MAX_MATCHES)
            packets = store.query_packets(payload, threshold=threshold, max_matches=max(1, max_matches))
            await ctx.set_kv(
                LAST_QUERY_KEY,
                {
                    "source_ref": payload.get("source_ref") or payload.get("track_id") or "",
                    "modality": payload.get("modality"),
                    "packet_count": len(packets),
                    "packets": [p.to_dict() for p in packets],
                },
            )
            return [
                Event(
                    topic="evidence/packet",
                    payload=packet.to_dict(),
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "event_class": "evidence",
                        "store_in_memory": False,
                        "semantic_input": False,
                        "meaning_boundary": "similarity_is_not_identity",
                    },
                )
                for packet in packets
                if isinstance(packet, EvidencePacket)
            ]

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="sensory_fossil",
        subscribed_topics=[STORE_TOPIC, QUERY_TOPIC, VISION_QUERY_TOPIC, TOUCH_QUERY_TOPIC, AUDIO_QUERY_TOPIC],
        output_topics=["sensory/fossil/stored", "evidence/packet"],
        cooldown_sec=0.0,
    )
    yield SensoryFossilNeuron(cfg)
