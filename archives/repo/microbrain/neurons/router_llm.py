from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event

# Neuron name = this file's basename without .py
# e.g. file "router_llm.py" -> NEURON_NAME = "router_llm"
NEURON_NAME = Path(__file__).stem


class TemplateNeuron(BaseNeuron):
    """
    Template neuron.

    Copy this file, rename it, tweak NEURON_NAME / topics / logic,
    and it will auto-register via neuron_loader.build_neurons().
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

        # TODO: replace this with whatever logic your neuron needs.
        #
        # Example: simple pass-through that just logs and does nothing:
        return []


def build_neurons(orchestrator):
    """
    Hook used by microbrain.orchestrator.neuron_loader.auto_register_neurons().

    This is what makes the neuron auto-register when you drop the file
    under microbrain/neurons/.
    """
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            # TODO: set the topics this neuron should listen to.
            # e.g. "percept/text", "reason/request", etc.
            # "percept/text",
        ],
        output_topics=[
            # TODO: set topics this neuron may emit.
            # e.g. "reason/request", "act/speech", etc.
            # "act/speech",
        ],
        priority=10,  # higher = earlier when multiple neurons fire on same topic
    )
    yield TemplateNeuron(cfg)
