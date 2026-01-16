from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class AudioCortexNeuron(BaseNeuron):
    """
    Audio cortex adapter.

    Listens on:
        - "percept/audio"

    Expected payload shape from an external STT / audio daemon:

        {
            "text": "recognized speech here",
            "confidence": 0.92,           # optional
            "speaker": "user",            # optional (user / other / unknown)
            "channel": "repl",            # optional, defaults to "repl"
            "raw_meta": { ... }           # any extra info
        }

    Behavior:

      - Validates and normalizes the incoming audio transcription.
      - Emits a "percept/text" event so the rest of the system can
        treat it exactly like normal text input (CLI, etc.).
      - Marks the source as "mic" in raw_meta, so downstream neurons
        can distinguish spoken vs typed input if they care.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "percept/audio":
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload type for percept/audio",
                payload_type=str(type(payload)),
            )
            return []

        raw_text = str(payload.get("text", "") or "").strip()
        if not raw_text:
            await ctx.log_debug(
                f"[{self.name}] Empty audio transcription; nothing to forward",
                topic=event.topic,
            )
            return []

        confidence = payload.get("confidence", None)
        speaker = payload.get("speaker", "user")
        channel = str(payload.get("channel", "repl"))

        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        # Tag that this came from mic/audio, not typed CLI
        raw_meta.update(
            {
                "input_modality": "audio",
                "speaker": speaker,
                "stt_confidence": confidence,
                "source": "mic",
                "channel": channel,
            }
        )

        # Build normalized text percept payload
        text_payload: Dict[str, Any] = {
            "text": raw_text,
            "source": "mic",
            "channel": channel,
            "raw_meta": raw_meta,
        }

        await ctx.log_debug(
            f"[{self.name}] Forwarding audio transcription as text",
            channel=channel,
            speaker=speaker,
            confidence=confidence,
            text_preview=raw_text[:80],
        )

        text_event = Event(
            topic="percept/text",
            payload=text_payload,
            source=self.name,
            correlation_id=event.correlation_id,
        )

        return [text_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio"],
        output_topics=["percept/text"],
        # Priority: early, so text router / attention see it quickly.
        priority=1,
    )
    yield AudioCortexNeuron(cfg)
