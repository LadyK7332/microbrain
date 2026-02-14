from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.patterns.lexicon_store import LexiconStore, simple_tokenize
from microbrain.patterns.pattern_edge_log import PatternEdgeLog

NEURON_NAME = Path(__file__).stem

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
    if t.isdigit():
        return False
    return True


class PatternRecallNeuron(BaseNeuron):
    """
    Spreading-activation recall v0:

    - listens to percept/text
    - reads pattern edges (token->concept)
    - ranks top concepts
    - writes a compact bundle to KV + memdir/state/recall_last.json

    This does NOT generate speech. It only prepares "what comes to mind"
    for the reasoner to optionally use.
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._lex: Optional[LexiconStore] = None
        self._edges: Optional[PatternEdgeLog] = None
        self._mem_store = None
        self._memdir: Optional[Path] = None

    async def _ensure_ready(self, ctx) -> bool:
        if self._lex is not None and self._edges is not None and self._mem_store is not None:
            return True

        mem_store = await ctx.get_kv("memory:store", None)
        if mem_store is None:
            return False

        memdir = Path(str(getattr(mem_store, "base_dir", "") or ""))
        if not str(memdir):
            return False

        if self._memdir != memdir:
            self._memdir = memdir
            self._lex = await ctx.get_kv("patterns:lexicon", None)
            self._edges = await ctx.get_kv("patterns:edges", None)

            if self._lex is None:
                self._lex = LexiconStore(memdir)
                await ctx.set_kv("patterns:lexicon", self._lex)

            if self._edges is None:
                self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")
                await ctx.set_kv("patterns:edges", self._edges)

            self._mem_store = mem_store

        return True

    def _semantic_anchor_salience(self, concept_id: str) -> Dict[str, float]:
        # best-effort: look for a concept_anchor row
        try:
            for row in reversed(getattr(self._mem_store, "semantic", []) or []):
                meta = row.get("meta", {}) if isinstance(row, dict) else {}
                if meta.get("kind") == "concept_anchor" and meta.get("concept_id") == concept_id:
                    sal = row.get("salience", {}) or {}
                    return {
                        "score": float(sal.get("score", 0.0) or 0.0),
                        "valence": float(sal.get("valence", 0.0) or 0.0),
                        "satisfaction": float(sal.get("satisfaction", 0.0) or 0.0),
                        "arousal": float(sal.get("arousal", 0.0) or 0.0),
                    }
        except Exception:
            pass
        return {"score": 0.0, "valence": 0.0, "satisfaction": 0.0, "arousal": 0.0}

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, source=event.source, payload=event.payload)

        if event.topic != "reason/request":
            return []

        if not await self._ensure_ready(ctx):
            return []

        payload = event.payload
        if not isinstance(payload, dict):
            return []

        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        role = str(payload.get("source", "user") or "user")
        if role not in ("user", "assistant", "system"):
            role = "user"

        # We only build recall from user input (keeps chatter from polluting recall)
        if role != "user":
            return []

        channel = str(payload.get("channel", "default") or "default")
        ts = float(payload.get("ts", 0.0) or event.timestamp or time.time())

        raw_meta = payload.get("raw_meta", None)
        if not isinstance(raw_meta, dict):
            # some upstream events may put it in meta instead
            raw_meta = event.meta.get("raw_meta", {}) if isinstance(event.meta, dict) else {}
        noun_id = str((raw_meta or {}).get("noun_id", "") or "").strip() or None
        if noun_id and not noun_id.startswith("noun:"):
            noun_id = f"noun:{noun_id}"

        tokens = [t for t in simple_tokenize(text) if _is_concept_candidate(t)]
        if not tokens:
            return []

        # --- Spread v0: token -> concept (1 hop) ---
        concept_scores: Dict[str, float] = {}

        # NOTE: PatternEdgeLog keeps an internal weight map; early-stage size is small.
        W = getattr(self._edges, "_W", {}) or {}
        for tok in tokens:
            token_id = f"token:{tok}"
            for k, w in W.items():
                try:
                    if getattr(k, "edge_type", "") != "token_concept":
                        continue
                    if getattr(k, "src", "") != token_id:
                        continue
                    concept_id = str(getattr(k, "dst", "") or "")
                    if not concept_id:
                        continue
                    concept_scores[concept_id] = concept_scores.get(concept_id, 0.0) + float(w)
                except Exception:
                    continue

            # If we have no edges yet, still allow the literal token concept to appear
            fallback = f"concept:{tok}"
            concept_scores.setdefault(fallback, 0.01)

                # --- Personal boost: noun -> concept edges ---
        if noun_id:
            alpha = float(await ctx.get_kv("patterns:noun_boost_alpha", 0.5) or 0.5)
            for k, w in W.items():
                try:
                    if getattr(k, "edge_type", "") != "noun_concept":
                        continue
                    if getattr(k, "src", "") != noun_id:
                        continue
                    cid = str(getattr(k, "dst", "") or "")
                    if not cid:
                        continue
                    concept_scores[cid] = concept_scores.get(cid, 0.0) + (alpha * float(w))
                except Exception:
                    continue


        # Rank top concepts
        ranked = sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_concepts = []
        for concept_id, score in ranked:
            label = concept_id.split(":", 1)[1] if ":" in concept_id else concept_id
            sal = self._semantic_anchor_salience(concept_id)
            top_concepts.append(
                {
                    "concept_id": concept_id,
                    "label": label,
                    "score": round(float(score), 6),
                    "salience": sal,
                }
            )

        bundle = {
            "query": text,
            "tokens": tokens[:16],
            "top_concepts": top_concepts,
            "channel": channel,
            "noun_id": noun_id,
            "ts": ts,
            "schema_ver": 2,
            "kind": "pattern_recall_bundle",
        }

        await ctx.set_kv("recall:last_bundle", bundle)

        # Write debug state file (NOT a memory journal)
        try:
            state_dir = Path(self._memdir) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "recall_last.json").write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        # Optional event for UIs/introspection (no default consumer required)
        return [Event(topic="memory/recall_context", payload=bundle, source=self.name)]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["reason/request"],
        output_topics=["memory/recall_context"],
        priority=6, # runs after binder(priority=7) and before reasoner(priority=5)
    )
    yield PatternRecallNeuron(cfg)
