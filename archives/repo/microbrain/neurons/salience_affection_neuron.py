"""
Salience / affection neuron

Heuristic detector for "oh my" / taboo / intimacy / trust signals in text.
Outputs a salience score + tags that other neurons (e.g. reason_llm) can consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

class SalienceAffectionNeuron(BaseNeuron):
    """
    NEURON_NAME is derived from filename by your loader, but we keep it explicit for clarity.
    If your framework auto-derives it, this constant should still be harmless.
    """

    NEURON_NAME = "salience_affection_neuron"
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        # Only handle normalized text percepts
        if event.topic != "percept/text":
            return []

        payload = event.payload or {}
        text = str(payload.get("text", "") or "")
        if not text.strip():
            return []

        lowered = text.lower()
        score, tags = self._score_text(lowered)

        # Very simple component signals derived from tags.
        # We can get fancier later; for now this just tells the rest of the brain:
        # "this felt affectionate / teasing / taboo / intimate".
        affection_level = 1.0 if ("intimacy" in tags or "trust" in tags) else 0.0
        tease_level = 1.0 if "taboo_spicy" in tags else 0.0
        intimacy_level = 1.0 if "intimacy" in tags else 0.0
        taboo_level = 1.0 if "taboo_spicy" in tags else 0.0
        power_direction = 0.0  # placeholder (dominance/submission) for later

        events: List[Event] = []

        # Global salience event (what AffectStateNeuron uses for "salience")
        events.append(
            Event(
                topic="affect/salience",
                payload={
                    "score": score,
                    "tags": tags,
                    "text": text,
                },
                source=self.name,
                correlation_id=event.correlation_id,
            )
        )

        if affection_level > 0.0:
            events.append(
                Event(
                    topic="affect/affection",
                    payload={"level": affection_level, "text": text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        if tease_level > 0.0:
            events.append(
                Event(
                    topic="affect/tease",
                    payload={"level": tease_level, "text": text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        if power_direction != 0.0:
            events.append(
                Event(
                    topic="affect/power",
                    payload={"direction": power_direction, "text": text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        if taboo_level > 0.0:
            events.append(
                Event(
                    topic="affect/taboo_edge",
                    payload={"level": taboo_level, "text": text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        if intimacy_level > 0.0:
            events.append(
                Event(
                    topic="affect/intimacy",
                    payload={"level": intimacy_level, "text": text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        self.debug(
            "emitted_salience",
            score=score,
            tags=tags,
            affection=affection_level,
            tease=tease_level,
            taboo=taboo_level,
        )

        return events

    # --- keyword pools -----------------------------------------------------

    INTIMACY_KEYWORDS = {
        "hug", "cuddle", "kiss", "snuggle", "hold you", "close to you",
        "intimate", "affection", "tender", "gentle", "caress",
    }

    TRUST_KEYWORDS = {
        "trust you", "safe with you", "rely on you", "depend on you",
        "i trust", "confide", "vulnerable with you",
    }

    TABOO_SPICY_KEYWORDS = {
        # keep this list PG-ish text-wise, even if interpretation is spicier
        "kink", "fetish", "nsfw", "spank", "collar", "dom", "sub",
        "bondage", "lewd", "naughty",
    }

    NEGATIVE_BOUNDARY_KEYWORDS = {
        "no consent", "did not consent", "non-consensual",
        "uncomfortable", "unsafe", "don’t like this", "stop",
    }

    # ----------------------------------------------------------------------

    def on_register(self) -> None:
        """
        Called by the framework when the neuron is loaded.
        We subscribe to percept/text so we see user + world text.
        """
        super().on_register()

        # If your bus API differs, adjust this call:
        #   - some versions use bus.subscribe(topic, callback)
        #   - others use bus.add_listener(...)
        self.bus.subscribe("percept/text", self.on_text_percept)

        # Optional roll-call for --debug runs
        self.debug("Registered SalienceAffectionNeuron", extra={"neuron": self.NEURON_NAME})

    async def on_text_percept(self, event: Dict[str, Any]) -> None:
        """
        Handle an incoming text percept.
        Expected shape (roughly):
            {
                "text": "...",
                "source": "cli" | "webui" | ...,
                "channel": "repl" | ...
                "raw_meta": {...}
            }
        We try not to assume too much; missing keys just degrade gracefully.
        """
        payload = event or {}
        text = (payload.get("text") or "").lower()

        if not text.strip():
            return

        score, tags = self._score_text(text)

        # Construct salience event for downstream consumers like reason_llm
        salience_event = {
            "score": score,          # 0.0–1.0
            "tags": tags,            # ["intimacy", "taboo", ...]
            "raw_text": payload.get("text"),
            "source": payload.get("source"),
            "channel": payload.get("channel"),
            "raw_meta": payload.get("raw_meta", {}),
        }

        # Topic name is intentionally generic "affect/salience"
        # so reason_llm and persona/LLM tone can listen.
        await self.bus.publish("affect/salience", salience_event)

        self.debug(
            "Emitted salience/affection signal",
            extra={"score": score, "tags": tags, "source": payload.get("source")},
        )

    # ------------------------------------------------------------------ #
    #  Internal scoring logic
    # ------------------------------------------------------------------ #

    def _score_text(self, text: str) -> Tuple[float, List[str]]:
        """
        Very simple heuristic.
        We’re not trying to be clever; just enough to steer persona / LLM tone.
        """

        tags: List[str] = []

        intimacy_hits = self._count_hits(text, self.INTIMACY_KEYWORDS)
        trust_hits = self._count_hits(text, self.TRUST_KEYWORDS)
        taboo_hits = self._count_hits(text, self.TABOO_SPICY_KEYWORDS)
        boundary_hits = self._count_hits(text, self.NEGATIVE_BOUNDARY_KEYWORDS)

        # basic tag assignment
        if intimacy_hits:
            tags.append("intimacy")
        if trust_hits:
            tags.append("trust")
        if taboo_hits:
            tags.append("taboo_spicy")
        if boundary_hits:
            tags.append("boundary_warning")

        # scoring:
        #   intimacy + trust push score up
        #   taboo pushes up but boundary_warning dampens it
        raw = (
            0.4 * self._squash(intimacy_hits)
            + 0.3 * self._squash(trust_hits)
            + 0.4 * self._squash(taboo_hits)
        )

        # boundary warnings dampen salience (1 hit ~ 40% damp)
        damp = 1.0 - min(0.8, 0.4 * boundary_hits)
        score = max(0.0, min(1.0, raw * damp))

        return score, tags

    @staticmethod
    def _count_hits(text: str, patterns: set[str]) -> int:
        hits = 0
        for pat in patterns:
            if pat in text:
                hits += 1
        return hits

    @staticmethod
    def _squash(n: int) -> float:
        """
        Tiny squashing function:
            0 -> 0.0
            1 -> 0.5
            2 -> ~0.75
            >=3 -> ~0.875
        """
        if n <= 0:
            return 0.0
        # simple 1 - 0.5^n curve
        value = 1.0 - (0.5 ** n)
        return value
    
def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    Picked up by auto_register_neurons(...).
    """
    cfg = NeuronConfig(
        name="salience_affection_neuron",
        subscribed_topics=[
            "percept/text",
        ],
        output_topics=[
            "affect/salience",
            "affect/affection",
            "affect/tease",
            "affect/power",
            "affect/taboo_edge",
            "affect/intimacy",
        ],
        priority=0,  # run before AffectStateNeuron (priority 1)
    )
    yield SalienceAffectionNeuron(cfg)

