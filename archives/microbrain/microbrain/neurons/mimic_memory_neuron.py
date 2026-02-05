from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:]")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _trim_counts(d: Dict[str, int], max_items: int) -> Dict[str, int]:
    # Keep the most frequent items. Deterministic ordering for ties.
    if len(d) <= max_items:
        return d
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(items[:max_items])


class MimicMemoryNeuron(BaseNeuron):
    """
    Collects lightweight language patterns from user text so babble can imitate.

    Subscribes:
      - percept/text  (user input)

    KV keys written:
      - mimic:unigrams          Dict[str,int]
      - mimic:bigrams           Dict[str,int]   key format: "t1|t2"
      - mimic:recent_phrases    List[str]       (clipped phrases)
      - mimic:last_user_text    str
      - mimic:last_update_ts    float
    """

    MAX_UNIGRAMS = 600
    MAX_BIGRAMS = 2500
    MAX_RECENT_PHRASES = 60

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/text":
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            return []

        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        # Basic filters: avoid storing giant blobs (paths/log dumps/etc.)
        if len(text) > 500:
            text = text[:500]

        tokens = _tokenize(text)
        if not tokens:
            return []

        # Load current state
        unigrams: Dict[str, int] = dict(await ctx.get_kv("mimic:unigrams", {}) or {})
        bigrams: Dict[str, int] = dict(await ctx.get_kv("mimic:bigrams", {}) or {})
        recent: List[str] = list(await ctx.get_kv("mimic:recent_phrases", []) or [])

        # Update counts
        for t in tokens:
            unigrams[t] = int(unigrams.get(t, 0)) + 1

        for a, b in zip(tokens, tokens[1:]):
            key = f"{a}|{b}"
            bigrams[key] = int(bigrams.get(key, 0)) + 1

        # Keep a short phrase sample (6..12 tokens)
        phrase_tokens = tokens[:12]
        if len(phrase_tokens) >= 3:
            phrase = " ".join(phrase_tokens)
            recent.append(phrase)
            if len(recent) > self.MAX_RECENT_PHRASES:
                recent = recent[-self.MAX_RECENT_PHRASES :]

        # Trim counts to caps
        unigrams = _trim_counts(unigrams, self.MAX_UNIGRAMS)
        bigrams = _trim_counts(bigrams, self.MAX_BIGRAMS)

        now = time.time()
        await ctx.set_kv("mimic:unigrams", unigrams)
        await ctx.set_kv("mimic:bigrams", bigrams)
        await ctx.set_kv("mimic:recent_phrases", recent)
        await ctx.set_kv("mimic:last_user_text", text)
        await ctx.set_kv("mimic:last_update_ts", now)

        self.debug(
            "mimic_updated",
            tokens=len(tokens),
            unigrams=len(unigrams),
            bigrams=len(bigrams),
            recent=len(recent),
        )
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text"],
        output_topics=[],
        priority=20,  # after router normalization, before curiosity fires is fine
    )
    yield MimicMemoryNeuron(cfg)
