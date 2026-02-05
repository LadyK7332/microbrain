from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class AffectStateNeuron(BaseNeuron):
    """
    Aggregates affect/* signals into a single affect/state snapshot.

    This does NOT generate any explicit content. It just tracks:
      - overall salience
      - affection / warmth
      - teasing energy
      - power dynamic direction
      - taboo-edge pressure
      - vulnerability / intimacy

    and produces a "last known emotional tone" object that other neurons
    (especially the LLM reasoner) can consume.
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._state: Dict[str, Any] = {
            "salience": 0.0,
            "affection": 0.0,
            "tease": 0.0,
            "power": 0.0,
            "taboo_edge": 0.0,
            "intimacy": 0.0,
            "last_text": "",
        }

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        # Debug roll-call; gated by --debug
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        topic = event.topic
        payload = event.payload or {}

        # Update internal state based on incoming affect/* events
        if topic == "affect/salience":
            self._state["salience"] = float(payload.get("score", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        elif topic == "affect/affection":
            self._state["affection"] = float(payload.get("level", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        elif topic == "affect/tease":
            self._state["tease"] = float(payload.get("level", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        elif topic == "affect/power":
            self._state["power"] = float(payload.get("direction", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        elif topic == "affect/taboo_edge":
            self._state["taboo_edge"] = float(payload.get("pressure", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        elif topic == "affect/intimacy":
            self._state["intimacy"] = float(payload.get("level", 0.0))
            self._state["last_text"] = payload.get("text", self._state["last_text"])

        else:
            # Not one of our topics
            return []

        # Emit a consolidated snapshot
        snapshot = dict(self._state)  # shallow copy
        out = Event(
            topic="affect/state",
            payload=snapshot,
            source=self.name,
            correlation_id=event.correlation_id,
        )
        return [out]


def build_neurons(orchestrator: Orchestrator):

    """
    Auto-loader hook.

    Picked up by auto_register_neurons(...).
    """
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "affect/salience",
            "affect/affection",
            "affect/tease",
            "affect/power",
            "affect/taboo_edge",
            "affect/intimacy",
        ],
        output_topics=["affect/state"],
        priority=1,  # run after salience_affection_neuron (priority 0)
    )
    yield AffectStateNeuron(cfg)
