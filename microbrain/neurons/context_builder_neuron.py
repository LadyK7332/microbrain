from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, Dict, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

TRIVIAL_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "good", "has", "have", "hello", "hey", "hi", "how", "i", "if", "in", "is", "it",
    "its", "just", "me", "my", "nice", "of", "ok", "okay", "or", "our", "so", "some",
    "something", "stuff", "that", "the", "there", "thing", "this", "to", "was", "we",
    "what", "when", "where", "who", "why", "yeah", "yes", "you", "your",
}

DIRECT_NAMES = {"demi", "hazard", "microbrain", "mb"}


class ContextBuilderNeuron(BaseNeuron):
    """
    Build a lightweight, explicit context packet before heavier reasoning.

    Listens on:
        - context/request

    Emits:
        - context/built
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        payload = event.payload
        if isinstance(payload, str):
            payload = {"text": payload}
        if not isinstance(payload, dict):
            return []

        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        source = str(payload.get("source", "user") or "user")
        channel = str(payload.get("channel", "default") or "default")
        raw_meta = dict(payload.get("raw_meta", {}) or {})

        lowered = text.lower()
        tokens = [tok for tok in lowered.replace("?", " ").replace("!", " ").replace(",", " ").replace(".", " ").split() if tok]
        meaningful_tokens = [
            tok for tok in tokens
            if tok in DIRECT_NAMES or (len(tok) >= 4 and tok not in TRIVIAL_TOKENS)
        ]
        greeting_words = {"hi", "hello", "hey", "yo", "morning", "afternoon", "evening"}

        memory = await ctx.get_kv("memory:store", None)
        recent: List[Dict[str, Any]] = []
        associations: List[Dict[str, Any]] = []
        if memory is not None:
            try:
                if hasattr(memory, "last_episodic"):
                    recent = list(memory.last_episodic(5) or [])
            except Exception as exc:
                self.debug("recent_lookup_failed", error=repr(exc))
            try:
                semantic_query = " ".join(meaningful_tokens).strip()
                if semantic_query and hasattr(memory, "search_semantic"):
                    raw_assoc = list(memory.search_semantic(semantic_query, k=5) or [])
                    associations = self._filter_associations(raw_assoc, meaningful_tokens)
            except Exception as exc:
                self.debug("semantic_lookup_failed", error=repr(exc))

        boredom = await ctx.get_kv("drive:boredom", {})
        social_interaction = await ctx.get_kv("drive:social_interaction", {})
        social_experimentation = await ctx.get_kv("drive:social_experimentation", {})
        thought_momentum = await ctx.get_kv("thought:momentum", {})
        scene_expectation_delta = await ctx.get_kv("scene:expectation:last_delta", {})
        scene_expectation = await ctx.get_kv("scene:expectation:last_exp", {})
        unresolved_questions = await ctx.get_kv("question:unresolved:recent", [])
        conversation_scene = await ctx.get_kv("conversation:scene", {})
        conversation_summary = await ctx.get_kv("conversation:summary", {})
        affect = await ctx.get_kv("affect:last", {})
        relation = await ctx.get_kv("relation:last", {})
        goals_crisis = bool(await ctx.get_kv("goals:crisis_mode", False))
        attention = await ctx.get_kv("attention:controller", None)
        allow_babble = getattr(attention, "allow_babble", False) if attention is not None else False
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)
        last_decision = await ctx.get_kv("context:last_decision", {})

        cues = {
            "is_question": "?" in text or lowered.startswith(("how ", "what ", "why ", "when ", "where ", "who ", "can ", "could ", "would ", "do ", "are ", "is ")),
            "is_greeting": any(tok in greeting_words for tok in tokens) or lowered in {"good morning", "good afternoon", "good evening"},
            "direct_address": any(tok in DIRECT_NAMES for tok in tokens),
            "well_wish": any(phrase in lowered for phrase in ("hope your", "hope you're", "doing well", "good morning", "good evening")),
            "needs_social_reply": any(phrase in lowered for phrase in ("how are you", "are you there", "respond maybe", "you there", "good morning", "good evening")),
            "has_meaningful_tokens": bool(meaningful_tokens),
        }

        top_assoc_score = 0.0
        if associations:
            try:
                top_assoc_score = float(associations[0].get("score", 0.0) or 0.0)
            except Exception:
                top_assoc_score = 0.0

        context = {
            "input": {
                "text": text,
                "source": source,
                "channel": channel,
                "raw_meta": raw_meta,
                "tokens": tokens,
                "meaningful_tokens": meaningful_tokens,
            },
            "recent": recent,
            "associations": associations,
            "association_meta": {
                "count": len(associations),
                "top_score": top_assoc_score,
                "query": " ".join(meaningful_tokens),
            },
            "drives": {
                "boredom": boredom,
                "social_interaction": social_interaction if isinstance(social_interaction, dict) else {},
                "social_experimentation": social_experimentation if isinstance(social_experimentation, dict) else {},
            },
            "thought_momentum": thought_momentum if isinstance(thought_momentum, dict) else {},
            "scene_expectation": {
                "last_exp": scene_expectation if isinstance(scene_expectation, dict) else {},
                "last_delta": scene_expectation_delta if isinstance(scene_expectation_delta, dict) else {},
                "unresolved_questions": unresolved_questions[-8:] if isinstance(unresolved_questions, list) else [],
            },
            "conversation_scene": conversation_scene if isinstance(conversation_scene, dict) else {},
            "conversation_summary": conversation_summary if isinstance(conversation_summary, dict) else {},
            "affect": affect if isinstance(affect, dict) else {},
            "relation": relation if isinstance(relation, dict) else {},
            "constraints": {
                "crisis_mode": goals_crisis,
                "allow_babble": bool(allow_babble),
            },
            "hrm": {
                "last_idx": hrm_last_idx,
            },
            "cues": cues,
            "last_decision": last_decision if isinstance(last_decision, dict) else {},
        }

        self.debug(
            "context_built",
            text_preview=text[:60],
            recent_n=len(recent),
            assoc_n=len(associations),
            assoc_top=round(top_assoc_score, 3),
            meaningful_tokens=meaningful_tokens,
            boredom=(boredom or {}).get("level", 0.0),
            social=(social_interaction or {}).get("level", 0.0) if isinstance(social_interaction, dict) else 0.0,
            social_experiment=(social_experimentation or {}).get("pressure", 0.0) if isinstance(social_experimentation, dict) else 0.0,
            momentum=(thought_momentum or {}).get("pressure", 0.0) if isinstance(thought_momentum, dict) else 0.0,
            momentum_intent=(thought_momentum or {}).get("dominant_intent", "") if isinstance(thought_momentum, dict) else "",
            scene_delta=(scene_expectation_delta or {}).get("magnitude", 0.0) if isinstance(scene_expectation_delta, dict) else 0.0,
            unresolved_q=len(unresolved_questions) if isinstance(unresolved_questions, list) else 0,
            conversation_topic=(conversation_summary or {}).get("topic", "") if isinstance(conversation_summary, dict) else "",
            conversation_turns=((conversation_scene or {}).get("state", {}) or {}).get("turn_count", 0) if isinstance(conversation_scene, dict) else 0,
            crisis_mode=goals_crisis,
            cues=cues,
        )

        return [
            Event(
                topic="context/built",
                payload={
                    "context": context,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"contextual": True, "stage": "built"},
            )
        ]

    def _filter_associations(self, raw_assoc: List[Dict[str, Any]], meaningful_tokens: List[str]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        meaningful_set = set(meaningful_tokens)
        for item in raw_assoc:
            if not isinstance(item, dict):
                continue
            score = 0.0
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            label = str(item.get("label", "") or "").strip().lower()
            text = str(item.get("text", "") or "").strip().lower()
            concept_id = str(item.get("concept_id", "") or "").strip().lower()
            candidate_blob = " ".join(x for x in [label, text, concept_id] if x)

            overlap = 0
            if meaningful_set:
                overlap = sum(1 for tok in meaningful_set if tok and tok in candidate_blob)

            if label in TRIVIAL_TOKENS and overlap == 0:
                continue
            if not meaningful_tokens:
                continue
            if score < 0.18 and overlap < 2:
                continue
            if score < 0.12:
                continue

            enriched = dict(item)
            enriched["score"] = score
            enriched["token_overlap"] = overlap
            filtered.append(enriched)

        filtered.sort(key=lambda x: (float(x.get("score", 0.0) or 0.0), int(x.get("token_overlap", 0) or 0)), reverse=True)
        return filtered[:3]



def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["context/request"],
        output_topics=["context/built"],
        priority=15,
    )
    yield ContextBuilderNeuron(cfg)
