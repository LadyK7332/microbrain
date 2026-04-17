from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class ReadingModeNeuron(BaseNeuron):
    """
    Deprecated inline reader shim.

    Read-mode chewing now lives in microbrain.sidecars.read_sidecar so heavy
    file parsing stays out of the interaction layer.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        return []



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[],
        output_topics=[],
        priority=-10,
        cooldown_sec=0.0,
    )
    yield ReadingModeNeuron(cfg)
