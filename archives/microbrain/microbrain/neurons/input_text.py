from __future__ import annotations

from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class TextInputNeuron(BaseNeuron):
    """
    First-stop neuron for incoming text.

    Listens on:
        - "input/text"

    Emits:
        - "percept/text" with a normalized payload:
            {
                "text": <str>,
                "source": <str>,   # e.g. "user", "ui", "minecraft"
                "channel": <str>,  # e.g. "cli", "webui", "discord"
                "raw_meta": {...}, # merged view of any extra metadata
            }

    This keeps the rest of the system talking in a consistent shape,
    regardless of how external systems format their text messages.
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

        # ----------------------------------------------
        # 1) Extract text + side metadata from payload
        # ----------------------------------------------
        text: str
        extra_meta: Dict[str, Any]

        if isinstance(event.payload, str):
            text = event.payload
            extra_meta = {}
        elif isinstance(event.payload, dict):
            # Common shape: {"text": "...", "source": "...", "channel": "...", ...}
            text = str(event.payload.get("text", ""))
            extra_meta = {k: v for k, v in event.payload.items() if k != "text"}
        else:
            # Fallback: stringify whatever was handed to us
            text = str(event.payload)
            extra_meta = {}

        text_norm = text.strip()
        if not text_norm:
            # Don't generate percepts for empty/whitespace-only input
            await ctx.log_debug(
                f"[{self.name}] Ignoring empty input payload",
                topic=event.topic,
            )
            return []

        # ----------------------------------------------
        # 2) Derive source/channel & merge metadata
        # ----------------------------------------------
        # Event meta wins over payload meta if both provide the same key.
        merged_meta: Dict[str, Any] = {}
        merged_meta.update(extra_meta)
        merged_meta.update(event.meta)

        source = merged_meta.get("source", "user")
        channel = merged_meta.get("channel", "default")

        # ----------------------------------------------
        # 3) Construct normalized percept payload
        # ----------------------------------------------
        percept_payload: Dict[str, Any] = {
            "text": text_norm,
            "source": source,
            "channel": channel,
            "raw_meta": merged_meta,
        }

        # Optionally: PDNA hints could be attached here later based on channel/source.

        percept_event = Event(
            topic="percept/text",
            payload=percept_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "percept",
                "modality": "text",
                "normalized": True,
            },
        )

        await ctx.log_debug(
            f"[{self.name}] Emitted percept/text",
            source=source,
            channel=channel,
        )

        return [percept_event]


def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    The orchestrator.neuron_loader.auto_register_neurons() will call this.
    """
    cfg = NeuronConfig(
        name="text_input",
        subscribed_topics=["input/text"],
        output_topics=["percept/text"],
        priority=10,  # early in the chain; feeds other percept neurons
    )
    yield TextInputNeuron(cfg)
