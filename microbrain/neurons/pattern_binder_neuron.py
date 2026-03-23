from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.memory.filters import classify_event_for_memory

from microbrain.patterns.lexicon_store import LexiconStore
from microbrain.patterns.pattern_edge_log import PatternEdgeLog

NEURON_NAME = Path(__file__).stem

# Keep this list short + obvious (we only want "meaningful" tokens as concepts early)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "it", "this", "that", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
}

def _is_concept_candidate(tok: str) -> bool:
    t = (tok or "").strip().lower()
    if not t:
        return False
    if t in _STOPWORDS:
        return False
    if len(t) < 3:
        return False
    # Avoid pure numbers; allow mixed alnum (e.g. "rx6600") if you want later
    if t.isdigit():
        return False
    return True


class PatternBinderNeuron(BaseNeuron):
    """
    Non-LLM binder v0:
      - observes percept/text
      - updates lexicon.jsonl (token_seen)
      - writes synapse-like pattern edges to synapses.jsonl (kind=pattern_edge)
      - creates ONE concept anchor per token (first-seen only) into semantic.jsonl

    This is the "word becomes a thing" spine that later binds to vision/audio/touch.
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._lex: Optional[LexiconStore] = None
        self._edges: Optional[PatternEdgeLog] = None
        self._mem_store = None
        self._memdir: Optional[str] = None

    async def _ensure_ready(self, ctx) -> bool:
        if self._lex is not None and self._edges is not None and self._mem_store is not None:
            return True

        mem_store = await ctx.get_kv("memory:store", None)
        if mem_store is None:
            return False

        # MemoryStore exposes base_dir
        memdir = str(getattr(mem_store, "base_dir", None) or "")
        if not memdir:
            return False

        if self._memdir != memdir:
            # (Re)init if memdir changed
            self._memdir = memdir
            self._lex = LexiconStore(memdir)
            self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")
            self._mem_store = mem_store

            # Expose handles to other neurons (in-memory KV, not persisted)
            await ctx.set_kv("patterns:lexicon", self._lex)
            await ctx.set_kv("patterns:edges", self._edges)

        return True

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # Debug roll-call
        self.debug("received", topic=event.topic, source=event.source, payload=event.payload)

        if event.topic != "reason/request":
            return []

        if not await self._ensure_ready(ctx):
            return []

        payload = event.payload
        if not isinstance(payload, dict):
            return []

        guard = classify_event_for_memory(event)
        if not guard.get("allow_pattern", False):
            self.debug("pattern_skip", reason=guard.get("junk_reason") or "blocked", channel=guard.get("channel"), kind=guard.get("kind"))
            return []

        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        role = str(payload.get("source", "user") or "user")
        if role not in ("user", "assistant", "system"):
            role = "user"

        raw_meta = payload.get("raw_meta", {})
        if not isinstance(raw_meta, dict):
            raw_meta = {}
        noun_id = str(raw_meta.get("noun_id", "") or "").strip() or None
        if noun_id and not noun_id.startswith("noun:"):
            noun_id = f"noun:{noun_id}"

        channel = str(payload.get("channel", "default") or "default")
        ts = float(payload.get("ts", 0.0) or event.timestamp or time.time())

        # 1) Tokenize + observe
        tokens = self._lex.observe_text(text, role=role, channel=channel, ts=ts)

        # 2) For each token, strengthen token<->concept edges
        # Slightly higher delta for user input vs assistant chatter
        base_delta = 0.06 if role == "user" else 0.03

        for tok in tokens:
            t = tok.lower()
            if not _is_concept_candidate(t):
                continue

            token_id = f"token:{t}"
            concept_id = f"concept:{t}"

            # bidirectional edges for cheap spreading activation later
            self._edges.add("token_concept", token_id, concept_id, base_delta, role="system", channel=channel, ts=ts)
            self._edges.add("concept_token", concept_id, token_id, base_delta, role="system", channel=channel, ts=ts)

            # speaker(noun) <-> concept edges (personal association spine)
            if noun_id and role == "user":
                self._edges.add("noun_concept", noun_id, concept_id, base_delta, role="system", channel=channel, ts=ts)
                self._edges.add("concept_noun", concept_id, noun_id, base_delta, role="system", channel=channel, ts=ts)

            # 3) First-seen concept anchor: only when lexicon count hits 1
            st = self._lex.get(t)
            if st and st.count == 1:
                meta: Dict[str, Any] = {
                    "role": "system",
                    "schema_ver": 2,
                    "kind": "concept_anchor",
                    "concept_id": concept_id,
                    "token_id": token_id,
                }
                # Neutral baseline; later reinforcement can bump it.
                sal = {"score": 0.0, "valence": 0.0, "satisfaction": 0.0, "arousal": 0.0}
                try:
                    self._mem_store.add_semantic(t, meta, salience=sal)
                except Exception:
                    # Never kill the bus for a memory write
                    pass

        # No emitted events yet (purely structural learning)
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["reason/request"],
        output_topics=[],
        priority=7,
    )
    yield PatternBinderNeuron(cfg)
