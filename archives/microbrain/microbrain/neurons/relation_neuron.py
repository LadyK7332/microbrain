"""
Relation neuron

Classifies and tags:
- interpersonal relations (boss, coworker, friend, family, partner, assistant, self)
- context state (work, play, afterhours)
- rough entity types (person/place/thing) based on simple heuristics.

Outputs:
- relation/context
- relation/entities
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


ContextState = Literal["work", "play", "afterhours", "unknown"]


class RelationNeuron(BaseNeuron):
    """
    NEURON_NAME will usually be derived from filename by the loader,
    but we keep it explicit for clarity.
    """

    NEURON_NAME = "relation_neuron"

    # --- keyword pools -----------------------------------------------------

    WORK_KEYWORDS = {
        "shift", "at work", "on the clock", "clocked in",
        "supervisor", "manager", "boss", "coworker", "colleague",
        "meeting", "ticket", "case", "customer", "client",
        "call center", "support queue", "work schedule",
    }

    PLAY_KEYWORDS = {
        "game", "gaming", "minecraft", "server", "raid",
        "match", "pvp", "dungeon", "party", "hanging out",
        "anime", "movie night", "watching a show",
        "vrchat", "discord call", "stream", "streaming",
        "playing around", "goofing off", "just chilling",
    }

    AFTERHOURS_KEYWORDS = {
        "after work", "off work", "off the clock",
        "at home", "in bed", "winding down", "late night",
        "nsfw", "afterhours", "after hours", "private time",
        "alone time", "nightcap",
    }

    INTERPERSONAL_PATTERNS = {
        "boss": ["my boss", "the boss", "manager", "supervisor"],
        "coworker": ["coworker", "co-worker", "colleague"],
        "friend": ["my friend", "a friend", "friends", "bestie"],
        "family": ["my mom", "my dad", "my sister", "my brother",
                   "my parents", "my family"],
        "partner": ["girlfriend", "boyfriend", "partner", "spouse",
                    "husband", "wife"],
        "self": ["i feel", "i think", "i want", "i am", "i'm"],
        "assistant": ["you", "my ai", "assistant", "microbrain",
                      "chatgpt", "bot", "you there"],
    }

    PLACE_KEYWORDS = {
        "home": ["home", "apartment", "house", "my place"],
        "workplace": ["office", "work", "call center", "site"],
        "public": ["cafe", "restaurant", "store", "mall", "park"],
        "online": ["server", "discord", "vrchat", "forum", "stream"],
    }

    THING_KEYWORDS = {
        "device": ["pc", "computer", "phone", "laptop", "console", "rig"],
        "project": ["project", "build", "repo", "codebase", "microbrain"],
        "game": ["minecraft", "game", "match", "run", "instance"],
    }

    # ------------------------------------------------------------------ #

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "percept/text":
            return []

        payload = event.payload or {}
        raw_text = str(payload.get("text", "") or "")
        if not raw_text.strip():
            return []

        text = raw_text.lower()

        context_state = self._classify_context_state(text)
        interpersonal_tags = self._extract_interpersonal_tags(text)
        entities = self._extract_entities(text)

        # Build relation/context summary
        context_payload: Dict[str, Any] = {
            "state": context_state,
            "interpersonal_tags": interpersonal_tags,
            "has_people": any(e["type"] == "person" for e in entities),
            "has_places": any(e["type"] == "place" for e in entities),
            "has_things": any(e["type"] == "thing" for e in entities),
            "text": raw_text,
        }

        events: List[Event] = []

        events.append(
            Event(
                topic="relation/context",
                payload=context_payload,
                source=self.name,
                correlation_id=event.correlation_id,
            )
        )

        if entities:
            events.append(
                Event(
                    topic="relation/entities",
                    payload={"entities": entities, "text": raw_text},
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        self.debug(
            "emitted_relations",
            state=context_state,
            interpersonal_tags=interpersonal_tags,
            entity_count=len(entities),
        )

        return events

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _classify_context_state(self, text: str) -> ContextState:
        """
        Heuristic classification into: work, play, afterhours, unknown.
        Priority: afterhours > work > play (if multiple match).
        """

        has_work = self._contains_any(text, self.WORK_KEYWORDS)
        has_play = self._contains_any(text, self.PLAY_KEYWORDS)
        has_after = self._contains_any(text, self.AFTERHOURS_KEYWORDS)

        if has_after:
            return "afterhours"
        if has_work:
            return "work"
        if has_play:
            return "play"
        return "unknown"

    def _extract_interpersonal_tags(self, text: str) -> List[str]:
        tags: List[str] = []
        for label, patterns in self.INTERPERSONAL_PATTERNS.items():
            if self._contains_any(text, patterns):
                tags.append(label)
        return tags

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        # people via interpersonal patterns
        for label, patterns in self.INTERPERSONAL_PATTERNS.items():
            for pat in patterns:
                if pat in text:
                    entities.append(
                        {
                            "type": "person",
                            "subtype": label,
                            "pattern": pat,
                        }
                    )

        # places
        for label, patterns in self.PLACE_KEYWORDS.items():
            for pat in patterns:
                if pat in text:
                    entities.append(
                        {
                            "type": "place",
                            "subtype": label,
                            "pattern": pat,
                        }
                    )

        # things
        for label, patterns in self.THING_KEYWORDS.items():
            for pat in patterns:
                if pat in text:
                    entities.append(
                        {
                            "type": "thing",
                            "subtype": label,
                            "pattern": pat,
                        }
                    )

        return entities

    @staticmethod
    def _contains_any(text: str, patterns: Iterable[str]) -> bool:
        return any(p in text for p in patterns)
    

def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    Picked up by auto_register_neurons(...).
    """
    cfg = NeuronConfig(
        name="relation_neuron",
        subscribed_topics=[
            "percept/text",
        ],
        output_topics=[
            "relation/context",
            "relation/entities",
        ],
        priority=0,  # run early, right after percept/text is born
    )
    yield RelationNeuron(cfg)
