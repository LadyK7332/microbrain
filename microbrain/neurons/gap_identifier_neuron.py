from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from microbrain.cognition.gap_identifier import (
    AUTO_CLARIFY_SILENT_USER_GAPS,
    build_clarification_need,
    build_evidence_need,
    build_gap_speech_payload,
    build_speech_obligation,
    identify_gap,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

SUBSCRIBED_TOPICS = [
    "hypothesis/action_committed",
    "context/built",
    "language/parsed",
    "percept/audio",
    "percept/vision",
    "vision/object_delta",
]

OUTPUT_TOPICS = [
    "cognition/gap_identified",
    "cognition/evidence_need",
    "cognition/clarification_need",
    "speech/response_obligation",
    "act/speech",
]


class GapIdentifierNeuron(BaseNeuron):
    """Identify missing intent/evidence gaps before silence becomes the answer."""

    async def process(self, event: Event, ctx: Any) -> Iterable[Event]:
        gap = identify_gap(
            event.topic,
            event.payload,
            source=event.source,
            event_meta=event.meta,
            now=event.timestamp,
        )
        if not gap.get("identified"):
            return []

        outputs: list[Event] = [
            Event(
                topic="cognition/gap_identified",
                payload=gap,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "gap_identified",
                    "gap_kind": gap.get("gap_kind", ""),
                    "source_topic": event.topic,
                    "store_in_memory": False,
                    "ephemeral": True,
                    "silence_allowed": bool(gap.get("silence_allowed")),
                },
            )
        ]

        evidence_need = build_evidence_need(gap)
        if evidence_need:
            outputs.append(
                Event(
                    topic="cognition/evidence_need",
                    payload=evidence_need,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "gap_evidence_need",
                        "gap_kind": gap.get("gap_kind", ""),
                        "modality": evidence_need.get("modality", ""),
                        "store_in_memory": False,
                        "ephemeral": True,
                    },
                )
            )

        clarification_need = build_clarification_need(gap)
        if clarification_need:
            outputs.append(
                Event(
                    topic="cognition/clarification_need",
                    payload=clarification_need,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "gap_clarification_need",
                        "gap_kind": gap.get("gap_kind", ""),
                        "store_in_memory": False,
                        "ephemeral": True,
                    },
                )
            )

        obligation = build_speech_obligation(gap)
        if obligation:
            outputs.append(
                Event(
                    topic="speech/response_obligation",
                    payload=obligation,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "gap_response_obligation",
                        "gap_kind": gap.get("gap_kind", ""),
                        "store_in_memory": False,
                        "ephemeral": True,
                    },
                )
            )

        # Operational bridge: current mouth pipeline may not consume
        # speech/response_obligation yet.  For safe user-originated ambiguity,
        # emit a tiny clarification surface directly to act/speech so the gap is
        # actually asked instead of recorded and ignored.
        speech_payload = build_gap_speech_payload(gap) if AUTO_CLARIFY_SILENT_USER_GAPS else None
        if speech_payload:
            outputs.append(
                Event(
                    topic="act/speech",
                    payload=speech_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "gap_clarification",
                        "channel": "textual",
                        "transport": "textual",
                        "suppress_tts": True,
                        "gap_id": gap.get("gap_id", ""),
                        "self_output_track": False,
                    },
                )
            )

        return outputs


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=SUBSCRIBED_TOPICS,
        output_topics=OUTPUT_TOPICS,
        priority=2,
        cooldown_sec=0.0,
    )
    return [GapIdentifierNeuron(cfg)]
