from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from microbrain.language_scaffold import ParsedText, TokenAtom, parse_text
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

_SKIP_REL_LEMMAS = {"be", "do", "have"}
SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
LISTENER_PRONOUNS = {"you", "your", "yours", "yourself"}
GROUP_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whom"}
AUX_QUESTION_WORDS = {"can", "could", "would", "will", "do", "does", "did", "is", "are", "was", "were", "should"}
MODALS = {"can", "could", "would", "will", "should", "may", "might", "must"}
NEGATIONS = {"not", "n't", "no", "never"}
TIME_MODIFIERS = {"now", "soon", "later", "then", "today", "tomorrow", "tonight", "eventually", "currently", "already", "next"}
INTENSIFIERS = {"very", "really", "quite", "too", "so", "extremely", "super", "highly"}
NEED_LEMMAS = {"need", "require", "want", "must", "should"}
PREFERENCE_LEMMAS = {"like", "love", "prefer", "enjoy", "want"}
REQUEST_LEMMAS = {"patch", "fix", "update", "make", "create", "generate", "write", "explain", "review", "check", "look"}
CHARGE_ACTIONS = {"charge", "recharge", "plug"}
REST_ACTIONS = {"sleep", "rest", "pause"}
FOOD_ACTIONS = {"eat", "feed"}
MAINT_ACTIONS = {"maintain", "repair", "fix", "service"}


class LanguageAtomizerNeuron(BaseNeuron):
    """
    Convert raw percept/text into language atoms MB can build on later.

    This neuron does NOT decide final meaning or action. It only:
      - parses text with the language scaffold
      - assigns word tool roles, not just parts of speech
      - derives lightweight relation/thought templates
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

        atomized = self._build_atomized_payload(
            parsed,
            text=text,
            channel=channel,
            source=source,
            correlation_id=event.correlation_id,
        )

        await ctx.set_kv("language:last_parse", atomized)
        await ctx.set_kv("language:last_atomized", atomized.get("atom_candidates", {}))

        await ctx.log_debug(
            f"[{self.name}] Atomized text",
            noun_count=len(atomized["atom_candidates"]["nouns"]),
            entity_count=len(atomized["atom_candidates"]["entities"]),
            relation_count=len(atomized["atom_candidates"]["relations"]),
            word_role_count=len(atomized["atom_candidates"]["word_roles"]),
            thought_template_count=len(atomized["atom_candidates"]["thought_templates"]),
            clause_candidate_count=len(atomized["atom_candidates"].get("clause_frames", [])),
            learning_frame_count=len(atomized["atom_candidates"].get("learning_frames", [])),
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
            Event(
                topic="language/thought_templates",
                payload={
                    "text": text,
                    "channel": channel,
                    "source": source,
                    "word_roles": atomized["atom_candidates"]["word_roles"],
                    "thought_templates": atomized["atom_candidates"]["thought_templates"],
                    "learning_frames": atomized["atom_candidates"].get("learning_frames", []),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "language_thought_templates",
                    "channel": channel,
                    "source": source,
                },
            ),
        ]

    def _build_atomized_payload(
        self,
        parsed: ParsedText,
        *,
        text: str,
        channel: str,
        source: str,
        correlation_id: str,
    ) -> Dict[str, Any]:
        noun_chunks = [str(chunk) for chunk in parsed.noun_chunks]
        entities = [dict(ent) for ent in parsed.entities]
        tokens = [self._token_to_dict(tok) for tok in parsed.tokens]
        word_roles = self._word_roles(parsed.tokens)
        thought_templates = self._thought_templates(parsed.tokens, text=text, source=source)

        atom_candidates = {
            "nouns": self._noun_candidates(parsed.tokens),
            "verbs": self._verb_candidates(parsed.tokens),
            "entities": self._entity_candidates(parsed.entities),
            "relations": self._relation_candidates(parsed.tokens),
            "noun_chunks": noun_chunks,
            "word_roles": word_roles,
            "thought_templates": thought_templates,
            "role_candidates": [dict(item) for item in parsed.role_candidates],
            "phrase_chunks": [dict(item) for item in parsed.phrase_chunks],
            "clause_frames": [dict(item) for item in parsed.clause_candidates],
            "learning_frames": [dict(item) for item in parsed.learning_frames],
            "best_clause": dict(parsed.best_clause),
        }

        return {
            "schema": "language.parsed.v3",
            "text": text,
            "channel": channel,
            "source": source,
            "correlation_id": correlation_id,
            "sentences": [str(s) for s in parsed.sentences],
            "noun_chunks": noun_chunks,
            "entities": entities,
            "tokens": tokens,
            "phrase_chunks": [dict(item) for item in parsed.phrase_chunks],
            "role_candidates": [dict(item) for item in parsed.role_candidates],
            "clause_candidates": [dict(item) for item in parsed.clause_candidates],
            "learning_frames": [dict(item) for item in parsed.learning_frames],
            "best_clause": dict(parsed.best_clause),
            "atom_candidates": atom_candidates,
        }

    def _token_to_dict(self, tok: TokenAtom) -> Dict[str, Any]:
        return {
            "idx": int(getattr(tok, "idx", 0) or 0),
            "text": tok.text,
            "lemma": tok.lemma,
            "norm": getattr(tok, "norm", "") or self._lemma(tok),
            "pos": tok.pos,
            "tag": getattr(tok, "tag", "") or "",
            "dep": tok.dep,
            "head_idx": int(getattr(tok, "head_idx", -1) or -1),
            "head_text": getattr(tok, "head_text", "") or "",
            "head_lemma": getattr(tok, "head_lemma", "") or "",
            "ent_type": getattr(tok, "ent_type", "") or "",
            "is_stop": bool(tok.is_stop),
            "is_alpha": bool(tok.is_alpha),
        }

    def _noun_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tok in tokens:
            if tok.pos not in ("NOUN", "PROPN", "PRON"):
                continue
            lemma = self._lemma(tok)
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
                    "tool_role": self._tool_role(tok, tokens),
                }
            )
        return out

    def _verb_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tok in tokens:
            if tok.pos not in ("VERB", "AUX"):
                continue
            lemma = self._lemma(tok)
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
                    "tool_role": self._tool_role(tok, tokens),
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
            out.append({"text": text, "label": label, "tool_role": "named_entity_anchor"})
        return out

    def _relation_candidates(self, tokens: List[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        current_subjects = [tok for tok in tokens if tok.dep in ("nsubj", "nsubjpass") and tok.pos in ("NOUN", "PROPN", "PRON")]
        current_objects = [tok for tok in tokens if tok.dep in ("dobj", "obj", "pobj", "attr", "oprd") and tok.pos in ("NOUN", "PROPN", "PRON", "ADJ")]
        relations = [tok for tok in tokens if tok.pos in ("VERB", "AUX", "ADP")]

        seen: set[tuple[str, str, str]] = set()
        for rel in relations:
            rel_lemma = self._lemma(rel)
            if not rel_lemma or rel_lemma in _SKIP_REL_LEMMAS:
                continue
            for subj in current_subjects:
                subj_lemma = self._lemma(subj)
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
                            "relation_role": self._relation_role(rel_lemma),
                            "confidence": 0.35,
                        }
                    )
                    continue
                for obj in current_objects:
                    obj_lemma = self._lemma(obj)
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
                            "relation_role": self._relation_role(rel_lemma),
                            "confidence": 0.6 if rel.pos == "VERB" else 0.5,
                        }
                    )
        return out

    def _word_roles(self, tokens: Sequence[TokenAtom]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tok in tokens:
            lemma = self._lemma(tok)
            text = str(tok.text or "").strip()
            if not text:
                continue
            role = self._tool_role(tok, tokens)
            out.append(
                {
                    "idx": int(getattr(tok, "idx", 0) or 0),
                    "text": text,
                    "lemma": lemma,
                    "pos": tok.pos,
                    "dep": tok.dep,
                    "part_of_speech": self._pos_category(tok),
                    "tool_role": role,
                    "thought_use": self._thought_use(role),
                    "head_idx": int(getattr(tok, "head_idx", -1) or -1),
                    "head_lemma": str(getattr(tok, "head_lemma", "") or "").lower(),
                    "confidence": 0.78 if tok.pos else 0.45,
                }
            )
        return out

    def _thought_templates(self, tokens: Sequence[TokenAtom], *, text: str, source: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        lows = [self._lemma(tok) or str(tok.text or "").lower() for tok in tokens]
        words = [str(tok.text or "").strip().lower() for tok in tokens]
        is_question = text.strip().endswith("?") or (words[:1] and words[0] in QUESTION_WORDS | AUX_QUESTION_WORDS)
        seen: set[tuple[str, str]] = set()

        def add(pattern_type: str, canonical: str, slots: Dict[str, Any], *, confidence: float = 0.66, evidence: Optional[List[int]] = None) -> None:
            canon = " ".join(str(canonical or "").split()).strip()
            if not canon:
                return
            key = (pattern_type, canon.lower())
            if key in seen:
                return
            seen.add(key)
            out.append(
                {
                    "pattern_type": pattern_type,
                    "canonical": canon,
                    "slots": {k: v for k, v in dict(slots or {}).items() if v not in (None, "", [])},
                    "confidence": max(0.0, min(1.0, float(confidence))),
                    "evidence_token_indices": list(evidence or []),
                    "source": source,
                }
            )

        # Generic question focus.
        if is_question:
            focus = [w for w in words if w not in QUESTION_WORDS and w not in AUX_QUESTION_WORDS and w not in {"a", "an", "the", "to"}]
            if focus:
                add(
                    "question_about",
                    "question " + " ".join(focus[:4]),
                    {"question_word": words[0] if words else "question", "focus": " ".join(focus[:4])},
                    confidence=0.62,
                    evidence=list(range(len(words))),
                )

        # Need/action relation: "I need to charge soon" / "Do you need to charge?"
        for i, lemma in enumerate(lows):
            if lemma not in NEED_LEMMAS:
                continue
            subject = self._nearest_subject(tokens, i) or ""
            action_idx = self._next_action_index(tokens, i + 1)
            action = lows[action_idx] if action_idx is not None else ""
            urgency = self._first_after(words, TIME_MODIFIERS, start=(action_idx or i) + 1)
            need_type = self._need_type_for_action(action)
            pattern_type = "query_need_action" if is_question else "need_action"
            canonical = " ".join(p for p in [subject or "someone", "need", "to", action, urgency] if p)
            add(
                pattern_type,
                canonical,
                {
                    "subject": subject,
                    "subject_ref": self._reference_scope(subject),
                    "relation": "need",
                    "action": action,
                    "need_type": need_type,
                    "urgency": urgency,
                    "is_question": is_question,
                },
                confidence=0.78 if action else 0.58,
                evidence=[idx for idx in [i, action_idx] if idx is not None],
            )

        # Direct action request: "can you patch it", "please fix that".
        if words and (words[0] in AUX_QUESTION_WORDS or "please" in words or any(lemma in REQUEST_LEMMAS for lemma in lows)):
            action_idx = self._request_action_index(tokens)
            if action_idx is not None:
                action = lows[action_idx]
                target = self._object_after(tokens, action_idx + 1)
                add(
                    "request_action",
                    " ".join(p for p in ["request", action, target] if p),
                    {
                        "actor": "listener" if "you" in words else "assistant",
                        "action": action,
                        "target": target,
                        "permission_shape": "can_you" if words[:1] and words[0] in AUX_QUESTION_WORDS else "please_or_imperative",
                        "is_question": is_question,
                    },
                    confidence=0.76,
                    evidence=[action_idx],
                )

        # Preference / desire action: "we like to visit old friends".
        for i, lemma in enumerate(lows):
            if lemma not in PREFERENCE_LEMMAS:
                continue
            subject = self._nearest_subject(tokens, i) or ""
            action_idx = self._next_action_index(tokens, i + 1)
            action = lows[action_idx] if action_idx is not None else ""
            obj = self._object_after(tokens, (action_idx or i) + 1)
            if action:
                add(
                    "preference_action",
                    " ".join(p for p in [subject or "someone", lemma, action, obj] if p),
                    {"subject": subject, "preference": lemma, "action": action, "object": obj},
                    confidence=0.70,
                    evidence=[idx for idx in [i, action_idx] if idx is not None],
                )

        # Attribute assertion from copula.
        for i, lemma in enumerate(lows):
            if lemma not in {"be", "is", "are", "was", "were"} and words[i] not in {"is", "are", "was", "were"}:
                continue
            subject = " ".join(words[:i]).strip()
            attribute = " ".join(words[i + 1 :]).strip(" .?")
            if subject and attribute:
                add(
                    "assert_attribute",
                    f"{subject} {words[i]} {attribute}",
                    {"subject": subject, "copula": words[i], "attribute": attribute},
                    confidence=0.72,
                    evidence=list(range(len(words))),
                )
                break

        # Generic subject/action/object fallback. This is intentionally weaker
        # and is skipped when a stronger need/request/preference template already
        # captured the thought.
        specific_template_present = any(
            t.get("pattern_type") in {"need_action", "query_need_action", "request_action", "preference_action"}
            for t in out
        )
        if not specific_template_present:
            subjects = [tok for tok in tokens if tok.dep in ("nsubj", "nsubjpass")]
            verbs = [tok for tok in tokens if tok.pos == "VERB" and self._lemma(tok) not in NEED_LEMMAS | PREFERENCE_LEMMAS]
            objects = [tok for tok in tokens if tok.dep in ("dobj", "obj", "pobj", "attr", "oprd")]
            if subjects and verbs:
                subj = self._lemma(subjects[0])
                verb = self._lemma(verbs[0])
                obj = self._lemma(objects[0]) if objects else self._object_after(tokens, int(getattr(verbs[0], "idx", 0)) + 1)
                if subj and verb:
                    add(
                        "action_relation",
                        " ".join(p for p in [subj, verb, obj] if p),
                        {"subject": subj, "action": verb, "object": obj},
                        confidence=0.58 if obj else 0.50,
                        evidence=[int(getattr(subjects[0], "idx", 0)), int(getattr(verbs[0], "idx", 0))],
                    )

        return out

    @staticmethod
    def _lemma(tok: TokenAtom) -> str:
        return str(tok.lemma or tok.text or "").strip().lower()

    @staticmethod
    def _pos_category(tok: TokenAtom) -> str:
        pos = str(tok.pos or "").upper()
        return {
            "NOUN": "noun",
            "PROPN": "noun",
            "PRON": "pronoun",
            "ADJ": "adjective",
            "ADV": "adverb",
            "VERB": "verb",
            "AUX": "verb",
            "ADP": "preposition",
            "CCONJ": "conjunction",
            "SCONJ": "conjunction",
            "INTJ": "interjection",
            "DET": "determiner",
            "PART": "particle",
            "PUNCT": "punctuation",
        }.get(pos, pos.lower() or "unknown")

    def _tool_role(self, tok: TokenAtom, tokens: Sequence[TokenAtom]) -> str:
        lemma = self._lemma(tok)
        pos = str(tok.pos or "").upper()
        dep = str(tok.dep or "").lower()
        if lemma in NEGATIONS or dep == "neg":
            return "negation_gate"
        if lemma in QUESTION_WORDS:
            return "question_focus_operator"
        if lemma in MODALS:
            return "capability_or_permission_operator"
        if pos in {"NOUN", "PROPN"}:
            if dep in {"nsubj", "nsubjpass"}:
                return "entity_actor"
            if dep in {"dobj", "obj", "pobj", "attr", "oprd"}:
                return "entity_target"
            return "entity_anchor"
        if pos == "PRON":
            if lemma in SELF_PRONOUNS:
                return "speaker_self_reference"
            if lemma in LISTENER_PRONOUNS:
                return "listener_reference"
            if lemma in GROUP_PRONOUNS:
                return "group_self_reference"
            return "entity_reference_pointer"
        if pos == "ADJ":
            return "attribute_modifier"
        if pos == "ADV":
            if lemma in TIME_MODIFIERS:
                return "time_or_urgency_modifier"
            if lemma in INTENSIFIERS:
                return "intensity_modifier"
            return "manner_modifier"
        if pos in {"VERB", "AUX"}:
            if lemma in NEED_LEMMAS:
                return "need_or_drive_relation"
            if lemma in PREFERENCE_LEMMAS:
                return "preference_relation"
            if lemma in _SKIP_REL_LEMMAS:
                return "state_or_identity_relation"
            return "action_or_process"
        if pos == "ADP":
            return "relationship_marker"
        if pos == "PART":
            return "infinitive_or_particle_marker"
        if pos in {"CCONJ", "SCONJ"}:
            return "structure_connector"
        if pos == "INTJ":
            return "discourse_emotion_marker"
        if pos == "DET":
            return "deixis_or_determiner"
        if pos == "PUNCT":
            return "boundary_marker"
        return "unknown_word_tool"

    @staticmethod
    def _thought_use(tool_role: str) -> str:
        role = str(tool_role or "")
        if "reference" in role or role.startswith("entity"):
            return "binds who_or_what"
        if "relation" in role or "action" in role:
            return "binds change_or_state"
        if "modifier" in role:
            return "changes intensity_time_or_quality"
        if "connector" in role:
            return "joins structures"
        if "question" in role:
            return "requests missing slot"
        if "negation" in role:
            return "inverts_or_blocks relation"
        return "supports parse"

    @staticmethod
    def _relation_role(lemma: str) -> str:
        if lemma in NEED_LEMMAS:
            return "need_or_drive"
        if lemma in PREFERENCE_LEMMAS:
            return "preference"
        if lemma in _SKIP_REL_LEMMAS:
            return "state_identity"
        return "action_relation"

    def _nearest_subject(self, tokens: Sequence[TokenAtom], rel_idx: int) -> str:
        before = list(tokens[: max(0, rel_idx)])
        for tok in reversed(before):
            if tok.dep in ("nsubj", "nsubjpass") or tok.pos in ("PRON", "NOUN", "PROPN"):
                lemma = self._lemma(tok)
                if lemma:
                    return lemma
        return ""

    def _next_action_index(self, tokens: Sequence[TokenAtom], start: int) -> Optional[int]:
        for idx in range(max(0, start), len(tokens)):
            tok = tokens[idx]
            lemma = self._lemma(tok)
            if lemma == "to":
                continue
            if tok.pos == "VERB" or lemma in REQUEST_LEMMAS | CHARGE_ACTIONS | REST_ACTIONS | FOOD_ACTIONS | MAINT_ACTIONS:
                return idx
        return None

    def _request_action_index(self, tokens: Sequence[TokenAtom]) -> Optional[int]:
        for idx, tok in enumerate(tokens):
            lemma = self._lemma(tok)
            if lemma in {"please", "you"} or lemma in AUX_QUESTION_WORDS:
                continue
            if tok.pos == "VERB" or lemma in REQUEST_LEMMAS:
                return idx
        return self._next_action_index(tokens, 0)

    def _object_after(self, tokens: Sequence[TokenAtom], start: int) -> str:
        collected: List[str] = []
        for tok in tokens[max(0, start) :]:
            lemma = self._lemma(tok)
            if not lemma or lemma in {"to", "please"}:
                continue
            if tok.pos in ("NOUN", "PROPN", "PRON", "ADJ") or tok.dep in ("dobj", "obj", "pobj", "attr", "oprd"):
                collected.append(lemma)
                if len(collected) >= 4:
                    break
            elif collected:
                break
        return " ".join(collected).strip()

    @staticmethod
    def _first_after(words: Sequence[str], allowed: set[str], *, start: int = 0) -> str:
        for word in list(words or [])[max(0, start) :]:
            if word in allowed:
                return word
        return ""

    @staticmethod
    def _reference_scope(subject: str) -> str:
        s = str(subject or "").strip().lower()
        if s in SELF_PRONOUNS:
            return "speaker_self"
        if s in LISTENER_PRONOUNS:
            return "listener_reference"
        if s in GROUP_PRONOUNS:
            return "speaker_group"
        return "entity_reference" if s else "unknown_reference"

    @staticmethod
    def _need_type_for_action(action: str) -> str:
        a = str(action or "").strip().lower()
        if a in CHARGE_ACTIONS:
            return "power_recovery"
        if a in REST_ACTIONS:
            return "rest_or_sleep"
        if a in FOOD_ACTIONS or a in {"food", "eat"}:
            return "hunger_or_fuel"
        if a in MAINT_ACTIONS:
            return "maintenance"
        return "unspecified_need"


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text"],
        output_topics=["language/parsed", "language/atom_candidates", "language/thought_templates"],
        priority=6,
    )
    yield LanguageAtomizerNeuron(cfg)
