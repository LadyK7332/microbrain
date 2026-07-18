from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import spacy


@dataclass
class TokenAtom:
    text: str
    lemma: str
    pos: str
    dep: str
    is_stop: bool
    is_alpha: bool
    idx: int = 0
    norm: str = ""
    tag: str = ""
    shape: str = ""
    head_idx: int = -1
    head_text: str = ""
    head_lemma: str = ""
    ent_type: str = ""


@dataclass
class ParsedText:
    text: str
    sentences: list[str]
    tokens: list[TokenAtom]
    noun_chunks: list[str]
    entities: list[dict[str, Any]]


_nlp = None
_nlp_load_failed = False
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whom"}
SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
LISTENER_PRONOUNS = {"you", "your", "yours", "yourself"}
GROUP_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
PRONOUNS = SELF_PRONOUNS | LISTENER_PRONOUNS | GROUP_PRONOUNS | {"he", "she", "it", "they", "them", "his", "her", "their"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
MODALS = {"can", "could", "would", "will", "should", "may", "might", "must", "do", "does", "did", "is", "are", "was", "were"}
PREPOSITIONS = {"to", "from", "in", "on", "at", "with", "without", "for", "about", "of", "into", "over", "under", "between"}
CONNECTORS = {"and", "or", "but", "because", "if", "then", "while", "although"}
TIME_MODIFIERS = {"now", "soon", "later", "then", "today", "tomorrow", "tonight", "eventually", "currently", "already", "next"}
INTENSIFIERS = {"very", "really", "quite", "too", "so", "extremely", "super", "highly"}
NEED_WORDS = {"need", "needs", "needed", "require", "requires", "want", "wants", "must", "should"}
PREFERENCE_WORDS = {"like", "likes", "love", "loves", "prefer", "prefers", "enjoy", "enjoys", "want", "wants"}
ACTION_WORDS = {"answer", "ask", "build", "charge", "create", "do", "explain", "fix", "generate", "get", "give", "go", "learn", "look", "make", "parse", "patch", "read", "remember", "say", "show", "sleep", "speak", "store", "tell", "think", "update", "use", "visit", "write"}
ATTRIBUTE_WORDS = {"old", "new", "good", "bad", "fast", "slow", "low", "high", "stable", "rough", "strong", "weak"}
DISCOURSE_MARKERS = {"well", "oh", "ah", "hey", "hello", "hi", "please", "thanks", "thank"}


def _get_nlp():
    """Load spaCy lazily so importing the scaffold never kills MB startup."""

    global _nlp, _nlp_load_failed
    if _nlp is None and not _nlp_load_failed:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp_load_failed = True
            return None
    return _nlp


def _norm_token(text: str) -> str:
    return str(text or "").strip().lower()


def _fallback_pos(token: str) -> tuple[str, str]:
    t = _norm_token(token)
    if not t:
        return "X", ""
    if not any(ch.isalnum() for ch in t):
        return "PUNCT", "punct"
    if t in PRONOUNS:
        return "PRON", "nsubj"
    if t in DETERMINERS:
        return "DET", "det"
    if t in QUESTION_WORDS:
        return "PRON", "advmod"
    if t in MODALS:
        return "AUX", "aux"
    if t in PREPOSITIONS:
        return "ADP", "prep"
    if t in CONNECTORS:
        return "CCONJ", "cc"
    if t in DISCOURSE_MARKERS:
        return "INTJ", "intj"
    if t in TIME_MODIFIERS or t in INTENSIFIERS or t.endswith("ly"):
        return "ADV", "advmod"
    if t in ATTRIBUTE_WORDS:
        return "ADJ", "amod"
    if t in NEED_WORDS or t in PREFERENCE_WORDS or t in ACTION_WORDS or t.endswith(("ing", "ed")):
        return "VERB", "ROOT"
    return "NOUN", "dep"


def _fallback_parse(text: str) -> ParsedText:
    raw_tokens = TOKEN_RE.findall(str(text or ""))
    token_atoms: list[TokenAtom] = []
    for idx, raw in enumerate(raw_tokens):
        pos, dep = _fallback_pos(raw)
        norm = _norm_token(raw)
        token_atoms.append(
            TokenAtom(
                text=raw,
                lemma=norm,
                pos=pos,
                dep=dep,
                is_stop=norm in DETERMINERS | PREPOSITIONS | CONNECTORS | MODALS,
                is_alpha=raw.isalpha(),
                idx=idx,
                norm=norm,
                tag=pos,
                shape="",
                head_idx=-1,
                head_text="",
                head_lemma="",
                ent_type="",
            )
        )
    sentence = str(text or "").strip()
    noun_chunks: list[str] = []
    for idx, tok in enumerate(token_atoms):
        if tok.pos in {"NOUN", "PROPN", "PRON"}:
            prev = token_atoms[idx - 1].text if idx > 0 and token_atoms[idx - 1].pos in {"DET", "ADJ"} else ""
            noun_chunks.append(" ".join(p for p in [prev, tok.text] if p).strip())
    return ParsedText(text=str(text or ""), sentences=[sentence] if sentence else [], tokens=token_atoms, noun_chunks=noun_chunks, entities=[])


def parse_text(text: str) -> ParsedText:
    nlp = _get_nlp()
    if nlp is None:
        return _fallback_parse(text)
    doc = nlp(text)

    tokens = [
        TokenAtom(
            text=t.text,
            lemma=t.lemma_,
            pos=t.pos_,
            dep=t.dep_,
            is_stop=bool(t.is_stop),
            is_alpha=bool(t.is_alpha),
            idx=int(t.i),
            norm=_norm_token(t.lemma_ or t.text),
            tag=t.tag_,
            shape=t.shape_,
            head_idx=int(t.head.i) if t.head is not None else -1,
            head_text=t.head.text if t.head is not None else "",
            head_lemma=t.head.lemma_ if t.head is not None else "",
            ent_type=t.ent_type_,
        )
        for t in doc
    ]

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]
    sentences = [sent.text.strip() for sent in doc.sents]

    return ParsedText(
        text=text,
        sentences=sentences,
        tokens=tokens,
        noun_chunks=noun_chunks,
        entities=entities,
    )
