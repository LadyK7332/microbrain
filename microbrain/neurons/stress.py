from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.patterns.pattern_edge_log import PatternEdgeLog

NEURON_NAME = Path(__file__).stem

# We keep the outcome namespace separate so later you can have:
# outcome:hazard, outcome:reward, outcome:social_good, outcome:maintenance_need, etc.
OUTCOME_HAZARD = "outcome:hazard"

# Edge types written into synapses.jsonl via PatternEdgeLog
EDGE_CONCEPT_OUTCOME = "concept_outcome"
EDGE_OUTCOME_CONCEPT = "outcome_concept"

# Learning + trigger knobs (intentionally conservative defaults)
TEACH_DELTA = 0.20          # explicit teaching bump ("alarm means dangerous")
TRIGGER_W = 0.40            # weight threshold before we consider it a real hazard concept
MAX_STRESS = 1.0
STRESS_DECAY_PER_SEC = 0.18 # passive decay toward 0

# Short-term association window (for future: co-occur learning)
RECENT_WINDOW_SEC = 6.0

# Small, explicit teaching patterns (bootstraps learning without hardcoding behavior)
# Examples it will catch:
#   "alarm means dangerous"
#   "alarm is dangerous"
#   "sirens mean danger"
_TEACH_RX: List[re.Pattern] = [
    re.compile(r"\b(?P<lemma>[a-zA-Z0-9_'-]{3,})\b\s+means\s+(danger|dangerous|a\s+hazard)\b", re.IGNORECASE),
    re.compile(r"\b(?P<lemma>[a-zA-Z0-9_'-]{3,})\b\s+is\s+(danger|dangerous|a\s+hazard)\b", re.IGNORECASE),
]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "it", "this", "that", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
}

def _tokenize(text: str) -> List[str]:
    # Simple + robust tokenizer (works for early bootstrapping)
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_'-]+", text or "")]

def _is_concept_candidate(tok: str) -> bool:
    t = (tok or "").strip().lower()
    if not t:
        return False
    if t in _STOPWORDS:
        return False
    if len(t) < 3:
        return False
    if t.isdigit():
        return False
    return True


class StressNeuron(BaseNeuron):
    """
    Learned "alarm reflex" neuron.

    Watches:
      - reason/request  (normalized text stream, good for teaching phrases)
      - percept/vision  (vision descriptions/objects)
      - clock/tick      (stress decay)

    Learns:
      concept:<x> -> outcome:hazard  (PatternEdgeLog edge: concept_outcome)

    Acts:
      - Emits curiosity/adjust with pause_s when hazard concepts fire,
        which AttentionController treats as a quiet/refractory window.
      - Optionally emits control/vision "on" when hazard comes from mic/audio.

    Note:
      This is an early reflex layer. Later, you can also reinforce concept_outcome
      edges using true co-occurrence events (touch spikes, audio signatures, etc.).
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._edges: Optional[PatternEdgeLog] = None
        self._memdir: Optional[str] = None

        # Most recent concept mentions across modalities, for near-term association
        self._recent: Deque[Tuple[float, str, str]] = deque(maxlen=256)  # (ts, concept_id, modality)

        # Current stress scalar (0..1)
        self._stress_level: float = 0.0
        self._last_tick_ts: float = time.time()

    async def _ensure_edges(self, ctx) -> bool:
        if self._edges is not None:
            return True

        # Prefer the shared PatternEdgeLog created by PatternBinderNeuron
        shared = await ctx.get_kv("patterns:edges", None)
        if isinstance(shared, PatternEdgeLog):
            self._edges = shared
            return True

        # Fallback: build our own from MemoryStore memdir
        mem_store = await ctx.get_kv("memory:store", None)
        memdir = str(getattr(mem_store, "base_dir", "") or "")
        if not memdir:
            return False

        self._memdir = memdir
        self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")
        return True

    def _prune_recent(self, now: float) -> None:
        # Deque is small; simple prune is fine
        while self._recent and (now - self._recent[0][0]) > RECENT_WINDOW_SEC:
            self._recent.popleft()

    def _record_concepts(self, now: float, modality: str, tokens: List[str]) -> List[str]:
        concepts: List[str] = []
        for tok in tokens:
            if not _is_concept_candidate(tok):
                continue
            cid = f"concept:{tok}"
            concepts.append(cid)
            self._recent.append((now, cid, modality))
        return concepts

    async def _teach_from_text(self, ctx, *, text: str, channel: str, ts: float) -> None:
        # Look for simple explicit teaching phrases:
        #   "alarm means dangerous"
        #   "sirens are dangerous"
        if not await self._ensure_edges(ctx):
            return

        lowered = (text or "").strip()
        if not lowered:
            return

        for rx in _TEACH_RX:
            m = rx.search(lowered)
            if not m:
                continue

            lemma = str(m.group("lemma") or "").strip().lower()
            if not _is_concept_candidate(lemma):
                continue

            concept_id = f"concept:{lemma}"

            w1 = self._edges.add(
                EDGE_CONCEPT_OUTCOME,
                concept_id,
                OUTCOME_HAZARD,
                TEACH_DELTA,
                role="system",
                channel=channel,
                ts=ts,
                meta={"learned_by": "stress_teach", "pattern": rx.pattern},
            )
            self._edges.add(
                EDGE_OUTCOME_CONCEPT,
                OUTCOME_HAZARD,
                concept_id,
                TEACH_DELTA,
                role="system",
                channel=channel,
                ts=ts,
                meta={"learned_by": "stress_teach", "pattern": rx.pattern},
            )

            self.debug("taught_hazard", concept=concept_id, w=w1, channel=channel)

    async def _hazard_score(self, ctx, concepts: List[str]) -> float:
        if not concepts:
            return 0.0
        if not await self._ensure_edges(ctx):
            return 0.0

        best = 0.0
        for cid in concepts:
            w = float(self._edges.weight(EDGE_CONCEPT_OUTCOME, cid, OUTCOME_HAZARD))
            if w > best:
                best = w
        return best

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call ---
        self.debug("received", topic=event.topic, source=event.source, payload=event.payload)

        now = time.time()

        # ------------------------------
        # Passive decay
        # ------------------------------
        if event.topic == "clock/tick":
            dt = max(0.0, now - float(self._last_tick_ts or now))
            self._last_tick_ts = now

            if self._stress_level > 0.0 and dt > 0.0:
                self._stress_level = max(0.0, self._stress_level - (STRESS_DECAY_PER_SEC * dt))
                await ctx.set_kv("drive:stress", {"level": self._stress_level, "ts": now})

            self._prune_recent(now)
            return []

        # ------------------------------
        # Text teaching + trigger checks
        # ------------------------------
        if event.topic == "reason/request":
            payload = event.payload if isinstance(event.payload, dict) else {}
            text = str(payload.get("text", "") or "").strip()
            channel = str(payload.get("channel", "default") or "default")
            src = str(payload.get("source", "user") or "user")  # "cli" or "mic" usually originate earlier

            if text:
                # 1) Learn from explicit teaching phrases
                # Only learn from user-originated text (prevents self-reinforcing loops)
                if src not in ("internal", "system", "assistant"):
                    await self._teach_from_text(ctx, text=text, channel=channel, ts=event.timestamp)

                # 2) Record concepts for short-term association
                toks = _tokenize(text)
                concepts = self._record_concepts(now, "text", toks)

                # 3) Check hazard score and react
                score = await self._hazard_score(ctx, concepts)
                if score >= TRIGGER_W:
                    self._stress_level = min(MAX_STRESS, max(self._stress_level, min(1.0, score)))
                    await ctx.set_kv("drive:stress", {"level": self._stress_level, "ts": now, "trigger": "text"})

                    # Clamp babble briefly (AttentionController consumes pause_s)
                    pause_s = float(1.0 + (self._stress_level * 3.0))
                    adjust = Event(
                        topic="curiosity/adjust",
                        payload={
                            "boost": 0.0,
                            "pause_s": pause_s,
                            "reason": "stress_hazard",
                            "text": text[:120],
                            "ts": now,
                        },
                        source=self.name,
                        correlation_id=event.correlation_id,
                    )

                    out: List[Event] = [adjust]

                    # Optional cross-trigger: if hazard was heard from mic, ensure vision is on
                    if src == "mic":
                        out.append(
                            Event(
                                topic="control/vision",
                                payload={"action": "on"},
                                source=self.name,
                                correlation_id=event.correlation_id,
                                meta={"kind": "stress_prime"},
                            )
                        )

                    self.debug("hazard_trigger", score=score, stress=self._stress_level, pause_s=pause_s, src=src)
                    return out

            return []

        # ------------------------------
        # Vision trigger checks (multimodal)
        # ------------------------------
        if event.topic == "percept/vision":
            payload = event.payload if isinstance(event.payload, dict) else {}
            desc = str(payload.get("description", "") or "").strip()
            objs = payload.get("objects", []) or []
            if not isinstance(objs, list):
                objs = [str(objs)]

            tokens: List[str] = []
            if desc:
                tokens.extend(_tokenize(desc))
            for o in objs[:12]:
                tokens.extend(_tokenize(str(o)))

            concepts = self._record_concepts(now, "vision", tokens)

            score = await self._hazard_score(ctx, concepts)
            if score >= TRIGGER_W:
                self._stress_level = min(MAX_STRESS, max(self._stress_level, min(1.0, score)))
                await ctx.set_kv("drive:stress", {"level": self._stress_level, "ts": now, "trigger": "vision"})

                pause_s = float(1.0 + (self._stress_level * 3.0))
                adjust = Event(
                    topic="curiosity/adjust",
                    payload={
                        "boost": 0.0,
                        "pause_s": pause_s,
                        "reason": "stress_hazard_vision",
                        "text": (desc or "vision")[:120],
                        "ts": now,
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                )

                self.debug("hazard_trigger_vision", score=score, stress=self._stress_level, pause_s=pause_s)
                return [adjust]

            return []

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "clock/tick",
            "reason/request",
            "percept/vision",  # safe even if vision isn't enabled
        ],
        output_topics=[
            "curiosity/adjust",
            "control/vision",
        ],
        priority=6,  # near PatternRecall (6) / after PatternBinder (7)
    )
    yield StressNeuron(cfg)