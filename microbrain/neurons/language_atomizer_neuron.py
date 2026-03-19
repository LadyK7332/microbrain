from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.language_scaffold import ParsedText, TokenAtom, parse_text
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

_SKIP_REL_LEMMAS = {"be", "do", "have"}


class LanguageAtomizerNeuron(BaseNeuron):
    """
    Convert raw percept/text into language atoms MB can build on later.

    This neuron does NOT decide meaning or action. It only:
      - parses text with the language scaffold
      - derives lightweight atom candidates (nouns / verbs / entities / relations)
      - emits structured language events for downstream binders / memory systems
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "percept/text":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {"text": event.payload}
        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        channel = str(payload.get("channel", "default") or "default")
        source = str(payload.get("source", event.source or "unknown") or "unknown")

        try:
            parsed = parse_text(text)
        except Exception as exc:
            await ctx.log_warn(
                f"[{self.name}] parse_text failed",
                topic=event.topic,
                error=repr(exc),
                text_preview=text[:120],
            )
            return []

        atomized = self._build_atomized_payload(parsed, text=text, channel=channel, source=source)

        await ctx.set_kv("language:last_parse", atomized)
        await ctx.set_kv("language:last_atomized", atomized.get("atom_candidates", {}))

        await ctx.log_debug(
            f"[{self.name}] Atomized text",
            noun_count=len(atomized["atom_candidates"]["nouns"]),
            entity_count=len(atomized["atom_candidates"]["entities"]),
            relation_count=len(atomized["atom_candidates"]["relations"]),
            text_preview=text[:100],
        )

        return [
            Event(
                topic="language/parsed",
                payload=atomized,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "language_parse",
                    "channel": channel,
                    "source": source,
                },
            ),
            Event(
                topic="language/atom_candidates",
                payload={
                    "text": text,
                    "channel": channel,
                    "source": source,
                    "atom_candidates": atomized["atom_candidates"],
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "language_atom_candidates",
                    "channel": channel,
                    "source": source,
                },
            ),
        ]

    def _build_atomized_payload(self, parsed: ParsedText, *, text: str, channel: str, source: str) -> Dict[str, Any]:
        noun_chunks = [str(chunk) for chunk in parsed.noun_chunks]
        entities = [dict(ent) for ent in parsed.entities]
        tokens = [self._token_to_dict(tok) for tok in parsed.tokens]

        atom_candidates = {
            "nouns": self._noun_candidates(parsed.tokens),
            "verbs": self._verb_candidates(parsed.tokens),
            "entities": self._entity_candidates(parsed.entities),
            "relations": self._relation_candidates(parsed.tokens),
            "noun_chunks": noun_chunks,
        }

        return {
            "schema": "language.parsed.v1",
            "text": text,
            "channel": channel,
            "source": source,
            "sentences": [str(s) for s in parsed.sentences],
            "noun_chunks": noun_chunks,
            "entities": entities,
            "tokens": tokens,
            "atom_candidates": atom_candidates,
        }

    def _token_to_dict(self, tok: TokenAtom) -> Dict[str, Any]:
        return {
            "text": tok.text,
            "lemma": tok.lemma,
            "pos": tok.pos,
            "dep": tok.dep,
            "is_stop": bool(tok.is_stop),
            "is_alpha": bool(tok.is_alpha),
        }

    def _noun_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tok in tokens:
            if tok.pos not in ("NOUN", "PROPN"):
                continue
            lemma = (tok.lemma or tok.text or "").strip().lower()
            if not lemma:
                continue
            key = (lemma, tok.pos)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": tok.text,
                    "lemma": lemma,
                    "pos": tok.pos,
                    "dep": tok.dep,
                }
            )
        return out

    def _verb_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tok in tokens:
            if tok.pos not in ("VERB", "AUX"):
                continue
            lemma = (tok.lemma or tok.text or "").strip().lower()
            if not lemma:
                continue
            key = (lemma, tok.pos)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": tok.text,
                    "lemma": lemma,
                    "pos": tok.pos,
                    "dep": tok.dep,
                }
            )
        return out

    def _entity_candidates(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for ent in entities:
            text = str(ent.get("text", "") or "").strip()
            label = str(ent.get("label", "") or "").strip()
            if not text:
                continue
            key = (text.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": text, "label": label})
        return out

    def _relation_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        current_subjects = [tok for tok in tokens if tok.dep in ("nsubj", "nsubjpass") and tok.pos in ("NOUN", "PROPN", "PRON")]
        current_objects = [tok for tok in tokens if tok.dep in ("dobj", "obj", "pobj", "attr", "oprd") and tok.pos in ("NOUN", "PROPN", "PRON")]
        relations = [tok for tok in tokens if tok.pos in ("VERB", "AUX", "ADP")]

        seen: set[tuple[str, str, str]] = set()
        for rel in relations:
            rel_lemma = (rel.lemma or rel.text or "").strip().lower()
            if not rel_lemma or rel_lemma in _SKIP_REL_LEMMAS:
                continue
            for subj in current_subjects:
                subj_lemma = (subj.lemma or subj.text or "").strip().lower()
                if not subj_lemma:
                    continue
                if not current_objects:
                    key = (subj_lemma, rel_lemma, "")
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "subject": subj_lemma,
                            "relation": rel_lemma,
                            "object": "",
                            "confidence": 0.35,
                        }
                    )
                    continue
                for obj in current_objects:
                    obj_lemma = (obj.lemma or obj.text or "").strip().lower()
                    if not obj_lemma or obj_lemma == subj_lemma:
                        continue
                    key = (subj_lemma, rel_lemma, obj_lemma)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "subject": subj_lemma,
                            "relation": rel_lemma,
                            "object": obj_lemma,
                            "confidence": 0.6 if rel.pos == "VERB" else 0.5,
                        }
                    )
        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text"],
        output_topics=["language/parsed", "language/atom_candidates"],
        priority=6,
    )
    yield LanguageAtomizerNeuron(cfg)
