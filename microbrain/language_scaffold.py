from __future__ import annotations

import spacy
from dataclasses import dataclass
from typing import Any

@dataclass
class TokenAtom:
    text: str
    lemma: str
    pos: str
    dep: str
    is_stop: bool
    is_alpha: bool

@dataclass
class ParsedText:
    text: str
    sentences: list[str]
    tokens: list[TokenAtom]
    noun_chunks: list[str]
    entities: list[dict[str, Any]]

_nlp = spacy.load("en_core_web_sm")

def parse_text(text: str) -> ParsedText:
    doc = _nlp(text)

    tokens = [
        TokenAtom(
            text=t.text,
            lemma=t.lemma_,
            pos=t.pos_,
            dep=t.dep_,
            is_stop=bool(t.is_stop),
            is_alpha=bool(t.is_alpha),
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