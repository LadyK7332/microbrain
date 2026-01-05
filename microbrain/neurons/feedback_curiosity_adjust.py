from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class FeedbackCuriosityAdjust(BaseNeuron):
    """
    Detect short, human-style negative correction tokens in user text and emit:

        topic="curiosity/adjust"
        payload={"boost":..., "pause_s":..., "reason":..., "text":...}

    AttentionController consumes pause_s to enforce a refractory quiet window.
    Curiosity (Step 2) will consume boost to "try again" with a smaller probe.
    """

    _RULES = [
        # reason, regex, boost, pause_s
        ("stop",  re.compile(r"\bstop\b", re.IGNORECASE), 0.60, 6.0),
        ("no",    re.compile(r"\bno\b", re.IGNORECASE),   0.35, 3.5),
        ("bad",   re.compile(r"\bbad\b", re.IGNORECASE),  0.20, 2.0),
        ("dont",  re.compile(r"\bdon[’']?t\b|\bdont\b", re.IGNORECASE), 0.10, 1.0),
    ]

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/text":
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            return []

        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        # Ignore internal/self text
        src = str(payload.get("source", "") or "")
        if src == "internal":
            return []

        # Find strongest matching rule
        best: Optional[Dict[str, Any]] = None
        for reason, rx, boost, pause_s in self._RULES:
            if rx.search(text):
                if best is None or float(pause_s) > float(best["pause_s"]):
                    best = {"reason": reason, "boost": float(boost), "pause_s": float(pause_s)}

        if not best:
            return []

        self.debug(
            "feedback_hit",
            reason=best["reason"],
            boost=best["boost"],
            pause_s=best["pause_s"],
            src=src,
        )

        return [
            Event(
                topic="curiosity/adjust",
                payload={
                    "boost": best["boost"],
                    "pause_s": best["pause_s"],
                    "reason": best["reason"],
                    "text": text,
                    "ts": time.time(),
                },
                source=self.name,
                correlation_id=event.correlation_id,
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text"],
        output_topics=["curiosity/adjust"],
        priority=6,  # early-ish: user feedback should clamp babble quickly
    )
    yield FeedbackCuriosityAdjust(cfg)
