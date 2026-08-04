from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Sequence

try:  # spaCy is optional; the machine-native fallback is the baseline parser.
    import spacy  # type: ignore
except Exception:  # pragma: no cover - depends on local installation
    spacy = None


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
    # Structure-first additions.  These remain candidates, not truth.
    phrase_chunks: list[dict[str, Any]] = field(default_factory=list)
    role_candidates: list[dict[str, Any]] = field(default_factory=list)
    clause_candidates: list[dict[str, Any]] = field(default_factory=list)
    best_clause: dict[str, Any] = field(default_factory=dict)


_nlp = None
_nlp_load_failed = False
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whom", "whose"}
SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
LISTENER_PRONOUNS = {"you", "your", "yours", "yourself"}
GROUP_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
THIRD_PRONOUNS = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
}
PRONOUNS = SELF_PRONOUNS | LISTENER_PRONOUNS | GROUP_PRONOUNS | THIRD_PRONOUNS
POSSESSIVE_PRONOUNS = {"my", "your", "his", "her", "its", "our", "their", "whose"}
OBJECT_PRONOUNS = {"me", "you", "him", "her", "it", "us", "them"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every"}
COPULAS = {"am", "is", "are", "was", "were", "be", "being", "been"}
AUXILIARIES = COPULAS | {"have", "has", "had", "do", "does", "did"}
MODALS = {"can", "could", "would", "will", "should", "may", "might", "must", "shall"}
PREPOSITIONS = {
    "to", "from", "in", "on", "at", "with", "without", "for", "about", "of", "into", "onto", "over",
    "under", "between", "through", "across", "around", "near", "beside", "behind", "before", "after", "by",
}
CONNECTORS = {"and", "or", "but", "because", "if", "then", "while", "although", "though", "unless", "until"}
NEGATIONS = {"not", "n't", "no", "never"}
TIME_MODIFIERS = {
    "now", "soon", "later", "then", "today", "tomorrow", "tonight", "yesterday", "eventually", "currently",
    "already", "next", "earlier", "recently", "always", "often", "sometimes", "rarely",
}
INTENSIFIERS = {"very", "really", "quite", "too", "so", "extremely", "super", "highly", "more", "less"}
NEED_WORDS = {"need", "needs", "needed", "require", "requires", "required", "want", "wants", "wanted"}
PREFERENCE_WORDS = {"like", "likes", "liked", "love", "loves", "prefer", "prefers", "enjoy", "enjoys"}
DISCOURSE_MARKERS = {"well", "oh", "ah", "hey", "hello", "hi", "please", "thanks", "thank", "okay", "ok"}
ATTRIBUTE_WORDS = {
    "old", "new", "good", "bad", "fast", "slow", "low", "high", "stable", "rough", "strong", "weak", "safe",
    "unsafe", "warm", "cold", "hot", "quiet", "loud", "bright", "dark", "near", "far", "happy", "sad", "angry",
    "tired", "hungry", "full", "small", "large", "big", "little", "red", "blue", "green", "soft", "hard", "furry",
}
ACTION_WORDS = {
    "answer", "ask", "become", "break", "bring", "build", "call", "carry", "charge", "check", "choose", "close",
    "come", "create", "cut", "dance", "do", "drive", "eat", "explain", "feel", "find", "fix", "follow", "generate",
    "get", "give", "go", "grab", "hear", "hold", "keep", "know", "learn", "leave", "look", "make", "measure",
    "move", "open", "parse", "patch", "put", "read", "remember", "repeat", "reply", "run", "say", "see", "show",
    "sing", "sleep", "speak", "start", "stop", "store", "take", "tell", "think", "touch", "turn", "update", "use",
    "visit", "wait", "walk", "want", "watch", "work", "write",
} | NEED_WORDS | PREFERENCE_WORDS
PERCEPTION_VERBS = {"see", "watch", "hear", "notice", "observe"}
LITERAL_CONTENT_VERBS = {"say", "repeat", "speak"}
CLOSED_CLASS_WORDS = (
    PRONOUNS | DETERMINERS | QUESTION_WORDS | MODALS | AUXILIARIES | PREPOSITIONS | CONNECTORS | NEGATIONS | DISCOURSE_MARKERS
)

IRREGULAR_LEMMAS = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "did": "do", "does": "do", "done": "do",
    "has": "have", "had": "have",
    "went": "go", "gone": "go",
    "came": "come",
    "saw": "see", "seen": "see",
    "heard": "hear",
    "ate": "eat", "eaten": "eat",
    "gave": "give", "given": "give",
    "made": "make",
    "took": "take", "taken": "take",
    "wrote": "write", "written": "write",
    "ran": "run",
    "said": "say",
    "told": "tell",
    "thought": "think",
    "brought": "bring",
    "found": "find",
    "left": "leave",
    "kept": "keep",
    "became": "become",
}

ROLE_TO_POS = {
    "noun": "NOUN",
    "proper_noun": "PROPN",
    "pronoun": "PRON",
    "verb": "VERB",
    "auxiliary": "AUX",
    "adjective": "ADJ",
    "adverb": "ADV",
    "determiner": "DET",
    "preposition": "ADP",
    "conjunction": "CCONJ",
    "particle": "PART",
    "punctuation": "PUNCT",
    "interjection": "INTJ",
}


# ---------------------------------------------------------------------------
# spaCy optional path
# ---------------------------------------------------------------------------

def _get_nlp():
    """Load spaCy lazily; MB's structure parser remains usable without it."""

    global _nlp, _nlp_load_failed
    if spacy is None:
        _nlp_load_failed = True
        return None
    if _nlp is None and not _nlp_load_failed:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp_load_failed = True
            return None
    return _nlp


# ---------------------------------------------------------------------------
# Token / morphology helpers
# ---------------------------------------------------------------------------

def _norm_token(text: str) -> str:
    return str(text or "").strip().lower()


def _simple_lemma(token: str) -> str:
    t = _norm_token(token)
    if not t:
        return ""
    if t in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[t]
    if t in ACTION_WORDS:
        return t

    # Only collapse morphology when the resulting stem is already a known
    # action. Unknown words stay intact: "snazzled" remains "snazzled" rather
    # than inventing the fake lemma "snazzl".
    if t.endswith("ies") and len(t) > 4:
        root = t[:-3] + "y"
        if root in ACTION_WORDS:
            return root
    if t.endswith("ing") and len(t) > 5:
        root = t[:-3]
        if len(root) >= 3 and root[-1:] == root[-2:-1]:
            root = root[:-1]
        if root in ACTION_WORDS:
            return root
        if root + "e" in ACTION_WORDS:
            return root + "e"
    if t.endswith("ed") and len(t) > 4:
        root = t[:-2]
        if root in ACTION_WORDS:
            return root
        if root + "e" in ACTION_WORDS:
            return root + "e"
    if t.endswith("s") and len(t) > 3 and t[:-1] in ACTION_WORDS:
        return t[:-1]
    return t


def _is_word(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9']+", str(text or "")))


def _score_add(scores: dict[str, float], role: str, amount: float) -> None:
    scores[role] = min(0.99, max(0.0, float(scores.get(role, 0.0))) + float(amount))


def _role_score(candidate: dict[str, Any], role: str) -> float:
    for item in candidate.get("candidates", []) if isinstance(candidate, dict) else []:
        if str(item.get("role", "") or "") == role:
            return float(item.get("score", 0.0) or 0.0)
    return 0.0


def _candidate_map(role_candidates: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(c.get("idx", -1)): c for c in role_candidates if isinstance(c, dict)}


def _infer_role_candidates_from_words(words: Sequence[str], *, spacy_tokens: Sequence[TokenAtom] | None = None) -> list[dict[str, Any]]:
    norms = [_norm_token(w) for w in words]
    out: list[dict[str, Any]] = []

    for idx, raw in enumerate(words):
        t = norms[idx]
        prev = norms[idx - 1] if idx > 0 else ""
        nxt = norms[idx + 1] if idx + 1 < len(norms) else ""
        prev2 = norms[idx - 2] if idx > 1 else ""
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        def add(role: str, amount: float, reason: str) -> None:
            _score_add(scores, role, amount)
            evidence.setdefault(role, []).append(reason)

        if not _is_word(raw):
            add("punctuation", 0.99, "punctuation")
        elif t in PRONOUNS:
            add("pronoun", 0.96, "closed_class_pronoun")
            if t in POSSESSIVE_PRONOUNS:
                add("determiner", 0.42, "possessive_can_modify_noun")
        elif t in DETERMINERS:
            add("determiner", 0.98, "closed_class_determiner")
        elif t in QUESTION_WORDS:
            add("pronoun", 0.80, "question_operator")
            add("adverb", 0.58, "question_modifier_possible")
        elif t in MODALS:
            add("auxiliary", 0.97, "modal_auxiliary")
        elif t in AUXILIARIES:
            add("auxiliary", 0.92, "auxiliary_or_copula")
            if t in COPULAS:
                add("verb", 0.45, "copula_relation")
        elif t in PREPOSITIONS:
            add("preposition", 0.96, "closed_class_preposition")
        elif t in CONNECTORS:
            add("conjunction", 0.95, "closed_class_connector")
        elif t in NEGATIONS:
            add("adverb", 0.92, "negation")
        elif t in DISCOURSE_MARKERS:
            add("interjection", 0.78, "discourse_marker")
        else:
            # Unknown content words start ambiguous. Context then does the work.
            add("noun", 0.30, "open_class_default")
            add("verb", 0.16, "open_class_possible_action")
            add("adjective", 0.10, "open_class_possible_attribute")

            lemma = _simple_lemma(t)
            if lemma in ACTION_WORDS or t in ACTION_WORDS:
                add("verb", 0.62, "known_action_lexeme")
            if t in ATTRIBUTE_WORDS:
                add("adjective", 0.66, "known_attribute_lexeme")
            if t in TIME_MODIFIERS or t in INTENSIFIERS:
                add("adverb", 0.62, "known_modifier_lexeme")
            if t.endswith("ly"):
                add("adverb", 0.54, "morphology_ly")
            if t.endswith(("tion", "ness", "ment", "ity", "ship", "ism")):
                add("noun", 0.48, "noun_morphology")
            if t.endswith(("ous", "ful", "less", "ive", "able", "ible", "al", "ic")):
                add("adjective", 0.46, "adjective_morphology")
            if t.endswith(("ing", "ed")):
                add("verb", 0.42, "verb_morphology")
                add("adjective", 0.14, "participle_can_modify")

        # English structural anchors. These are intentionally stronger than
        # unknown-word dictionary guesses. Closed-class words keep their job;
        # word-order inference is for content words, not for turning "of" into
        # a verb because a determiner happens to follow it.
        if _is_word(raw) and t not in CLOSED_CLASS_WORDS:
            if prev in DETERMINERS or prev in POSSESSIVE_PRONOUNS:
                next_is_content = bool(nxt) and nxt not in CLOSED_CLASS_WORDS and _is_word(words[idx + 1] if idx + 1 < len(words) else "")
                next_is_action = nxt in ACTION_WORDS or _simple_lemma(nxt) in ACTION_WORDS or nxt.endswith(("ing", "ed"))
                if next_is_content and not next_is_action:
                    # "the fluffy fox" / "the red car": first open-class word
                    # is more likely a modifier than the phrase head.
                    add("adjective", 0.52, "modifier_inside_noun_phrase")
                    add("noun", 0.18, "noun_modifier_possible")
                else:
                    add("noun", 0.42, "after_determiner")
            if idx > 0 and prev not in CLOSED_CLASS_WORDS and (
                prev in ATTRIBUTE_WORDS or prev.endswith(("ous", "ful", "less", "ive", "able", "ible", "al", "ic", "y"))
            ):
                if idx > 1 and norms[idx - 2] in DETERMINERS | POSSESSIVE_PRONOUNS:
                    add("noun", 0.38, "head_after_noun_phrase_modifier")
            if prev == "to" and prev2 not in {"go", "come", "walk", "drive", "move"}:
                add("verb", 0.46, "after_infinitive_to")
            if prev in MODALS or (prev in AUXILIARIES and prev not in COPULAS):
                add("verb", 0.42, "after_auxiliary")
            if prev in COPULAS:
                add("adjective", 0.34, "copular_complement")
                add("noun", 0.20, "copular_identity_possible")
            if nxt in DETERMINERS or nxt in PRONOUNS:
                # "glorp snazzled the flib" strongly suggests the middle word
                # is the predicate even if it is unknown.
                if prev and prev not in DETERMINERS | PREPOSITIONS | CONNECTORS | MODALS | AUXILIARIES:
                    add("verb", 0.36, "between_entity_and_noun_phrase")
            if nxt in ACTION_WORDS or _simple_lemma(nxt) in ACTION_WORDS:
                add("noun", 0.24, "before_known_predicate")
            if prev in ACTION_WORDS or _simple_lemma(prev) in ACTION_WORDS:
                add("noun", 0.22, "after_predicate_object_position")
            if idx == 0 and nxt in DETERMINERS and (t in ACTION_WORDS or _simple_lemma(t) in ACTION_WORDS):
                add("verb", 0.25, "imperative_start")

            # Perception-verb ambiguity: "I saw her duck." can mean an object
            # or an embedded action. Preserve both instead of forcing one.
            later_words = [j for j in range(idx + 1, len(words)) if _is_word(words[j])]
            if not later_words and prev in OBJECT_PRONOUNS | POSSESSIVE_PRONOUNS:
                prior_lemmas = {_simple_lemma(x) for x in norms[:idx]}
                if prior_lemmas & PERCEPTION_VERBS:
                    add("verb", 0.30, "perception_embedded_action_possible")
                    add("noun", 0.18, "possessive_object_possible")

        # When spaCy exists, treat it as strong evidence but retain alternatives.
        if spacy_tokens is not None and idx < len(spacy_tokens):
            sp = spacy_tokens[idx]
            pos = str(sp.pos or "").upper()
            role = {
                "NOUN": "noun", "PROPN": "proper_noun", "PRON": "pronoun", "VERB": "verb", "AUX": "auxiliary",
                "ADJ": "adjective", "ADV": "adverb", "DET": "determiner", "ADP": "preposition", "CCONJ": "conjunction",
                "SCONJ": "conjunction", "PART": "particle", "PUNCT": "punctuation", "INTJ": "interjection",
            }.get(pos)
            if role:
                add(role, 0.72, "spacy_pos")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_role = ranked[0][0] if ranked else "unknown"
        best_score = ranked[0][1] if ranked else 0.0
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = max(0.0, min(0.99, best_score - (0.20 * second_score)))
        out.append(
            {
                "idx": idx,
                "text": str(raw),
                "norm": t,
                "lemma": _simple_lemma(t),
                "best_role": best_role,
                "confidence": round(confidence, 4),
                "candidates": [
                    {
                        "role": role,
                        "score": round(score, 4),
                        "evidence": evidence.get(role, [])[:5],
                    }
                    for role, score in ranked[:4]
                ],
            }
        )
    return out


def _fallback_pos_from_candidate(candidate: dict[str, Any]) -> tuple[str, str]:
    role = str(candidate.get("best_role", "unknown") or "unknown")
    pos = ROLE_TO_POS.get(role, "X")
    dep = {
        "determiner": "det",
        "preposition": "prep",
        "conjunction": "cc",
        "adverb": "advmod",
        "adjective": "amod",
        "auxiliary": "aux",
        "interjection": "intj",
        "punctuation": "punct",
    }.get(role, "dep")
    return pos, dep


def _split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in SENTENCE_RE.finditer(str(text or "")) if m.group(0).strip()]


# ---------------------------------------------------------------------------
# Phrase and clause structure
# ---------------------------------------------------------------------------

def _best_role(role_map: dict[int, dict[str, Any]], idx: int) -> str:
    return str((role_map.get(idx) or {}).get("best_role", "unknown") or "unknown")


def _word_indices(tokens: Sequence[TokenAtom]) -> list[int]:
    return [idx for idx, tok in enumerate(tokens) if _is_word(tok.text)]


def _head_of(indices: Sequence[int], tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]]) -> str:
    for idx in reversed(list(indices)):
        role = _best_role(role_map, idx)
        if role in {"noun", "proper_noun", "pronoun"}:
            return _simple_lemma(tokens[idx].text)
    for idx in reversed(list(indices)):
        if _is_word(tokens[idx].text):
            return _simple_lemma(tokens[idx].text)
    return ""


def _surface(indices: Sequence[int], tokens: Sequence[TokenAtom], *, drop_determiners: bool = False) -> str:
    pieces: list[str] = []
    for idx in indices:
        if idx < 0 or idx >= len(tokens):
            continue
        raw = str(tokens[idx].text or "").strip()
        norm = _norm_token(raw)
        if not raw or not _is_word(raw):
            continue
        if drop_determiners and norm in DETERMINERS:
            continue
        pieces.append(raw)
    return " ".join(pieces).strip()


def _noun_phrase_ending_at(end_idx: int, tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]]) -> list[int]:
    if end_idx < 0:
        return []
    out: list[int] = []
    idx = end_idx
    allowed = {"noun", "proper_noun", "pronoun", "adjective", "determiner"}
    while idx >= 0:
        if not _is_word(tokens[idx].text):
            break
        role = _best_role(role_map, idx)
        if role not in allowed and _norm_token(tokens[idx].text) not in POSSESSIVE_PRONOUNS:
            break
        out.append(idx)
        idx -= 1
    return list(reversed(out))


def _noun_phrase_starting_at(start_idx: int, tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]]) -> list[int]:
    out: list[int] = []
    idx = max(0, start_idx)
    allowed = {"noun", "proper_noun", "pronoun", "adjective", "determiner"}
    while idx < len(tokens):
        if not _is_word(tokens[idx].text):
            break
        norm = _norm_token(tokens[idx].text)
        role = _best_role(role_map, idx)
        if role not in allowed and norm not in POSSESSIVE_PRONOUNS:
            break
        out.append(idx)
        idx += 1
    return out


def _build_phrase_chunks(tokens: Sequence[TokenAtom], role_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    role_map = _candidate_map(role_candidates)
    out: list[dict[str, Any]] = []
    idx = 0
    used: set[tuple[str, int, int]] = set()

    def add(kind: str, indices: Sequence[int], confidence: float) -> None:
        if not indices:
            return
        key = (kind, min(indices), max(indices))
        if key in used:
            return
        used.add(key)
        out.append(
            {
                "kind": kind,
                "text": _surface(indices, tokens),
                "start": min(indices),
                "end": max(indices),
                "head": _head_of(indices, tokens, role_map),
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "token_indices": list(indices),
            }
        )

    while idx < len(tokens):
        tok = tokens[idx]
        if not _is_word(tok.text):
            idx += 1
            continue
        role = _best_role(role_map, idx)
        norm = _norm_token(tok.text)

        if role in {"determiner", "adjective", "noun", "proper_noun", "pronoun"} or norm in POSSESSIVE_PRONOUNS:
            phrase = _noun_phrase_starting_at(idx, tokens, role_map)
            if phrase:
                conf = sum(float((role_map.get(i) or {}).get("confidence", 0.4) or 0.4) for i in phrase) / len(phrase)
                add("noun_phrase", phrase, conf)
                idx = phrase[-1] + 1
                continue

        if role in {"auxiliary", "verb", "adverb", "particle"} or norm in NEGATIONS:
            group: list[int] = []
            j = idx
            while j < len(tokens) and _is_word(tokens[j].text):
                r = _best_role(role_map, j)
                n = _norm_token(tokens[j].text)
                if r not in {"auxiliary", "verb", "adverb", "particle"} and n not in NEGATIONS:
                    break
                group.append(j)
                j += 1
            add("verb_group", group, 0.72)
            idx = j
            continue

        if role == "preposition":
            group = [idx]
            np = _noun_phrase_starting_at(idx + 1, tokens, role_map)
            group.extend(np)
            add("prepositional_phrase", group, 0.70)
            idx = (group[-1] + 1) if group else idx + 1
            continue

        idx += 1

    return out


def _verb_candidates(tokens: Sequence[TokenAtom], role_candidates: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    role_map = _candidate_map(role_candidates)
    out: list[tuple[int, float]] = []
    for idx, tok in enumerate(tokens):
        if not _is_word(tok.text):
            continue
        norm = _norm_token(tok.text)
        lemma = _simple_lemma(norm)
        score = _role_score(role_map.get(idx, {}), "verb")
        aux_score = _role_score(role_map.get(idx, {}), "auxiliary")
        if lemma in ACTION_WORDS:
            score = max(score, 0.82)
        if norm in AUXILIARIES | MODALS:
            score = max(score, aux_score * 0.45)
        if score >= 0.34:
            out.append((idx, score))
    return sorted(out, key=lambda item: item[1], reverse=True)


def _first_main_verb_after(start_idx: int, tokens: Sequence[TokenAtom], role_candidates: Sequence[dict[str, Any]]) -> tuple[int, float] | None:
    cands = _verb_candidates(tokens, role_candidates)
    role_map = _candidate_map(role_candidates)
    positional: list[tuple[int, float]] = []
    for idx, score in cands:
        if idx <= start_idx:
            continue
        norm = _norm_token(tokens[idx].text)
        if norm in MODALS:
            continue
        if norm in AUXILIARIES and norm not in ACTION_WORDS - COPULAS:
            continue
        # Slightly prefer earlier plausible predicates after a subject/aux.
        positional.append((idx, score + max(0.0, 0.08 - 0.01 * max(0, idx - start_idx - 1))))
    if positional:
        return max(positional, key=lambda item: item[1])

    # Last-resort structural inference: an unknown word between an entity-ish
    # subject and a determiner/noun phrase is probably the predicate.
    for idx in range(start_idx + 1, len(tokens)):
        if not _is_word(tokens[idx].text):
            continue
        if _role_score(role_map.get(idx, {}), "verb") >= 0.25:
            return idx, _role_score(role_map.get(idx, {}), "verb")
    return None


def _subject_np_before_verb(verb_idx: int, tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]]) -> list[int]:
    end = verb_idx - 1
    while end >= 0 and _is_word(tokens[end].text):
        norm = _norm_token(tokens[end].text)
        role = _best_role(role_map, end)
        if norm in AUXILIARIES | MODALS | NEGATIONS or role in {"auxiliary", "adverb", "particle"}:
            end -= 1
            continue
        break
    return _noun_phrase_ending_at(end, tokens, role_map)


def _next_np_after(start_idx: int, tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]]) -> list[int]:
    idx = max(0, start_idx)
    while idx < len(tokens):
        if not _is_word(tokens[idx].text):
            idx += 1
            continue
        norm = _norm_token(tokens[idx].text)
        role = _best_role(role_map, idx)
        if norm in {"please"} or role in {"adverb", "particle"} or norm in NEGATIONS:
            idx += 1
            continue
        if role == "preposition":
            return []
        if role in {"determiner", "adjective", "noun", "proper_noun", "pronoun"} or norm in POSSESSIVE_PRONOUNS:
            return _noun_phrase_starting_at(idx, tokens, role_map)
        idx += 1
    return []


def _prepositional_adjuncts(tokens: Sequence[TokenAtom], role_map: dict[int, dict[str, Any]], *, start_idx: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = max(0, start_idx)
    while idx < len(tokens):
        if not _is_word(tokens[idx].text):
            idx += 1
            continue
        if _best_role(role_map, idx) != "preposition":
            idx += 1
            continue
        relation = _norm_token(tokens[idx].text)
        np = _noun_phrase_starting_at(idx + 1, tokens, role_map)
        if np:
            out.append(
                {
                    "relation": relation,
                    "target": _head_of(np, tokens, role_map),
                    "target_text": _surface(np, tokens),
                    "token_indices": [idx] + np,
                    "confidence": 0.70,
                }
            )
            idx = np[-1] + 1
        else:
            idx += 1
    return out


def _clause_base(text: str, clause_type: str, confidence: float) -> dict[str, Any]:
    return {
        "clause_type": clause_type,
        "surface": str(text or "").strip(),
        "voice": "active",
        "subject": "",
        "subject_text": "",
        "subject_implied": False,
        "action": "",
        "action_surface": "",
        "object": "",
        "object_text": "",
        "complement": "",
        "agent": "",
        "patient": "",
        "query_target": "",
        "negated": False,
        "adjuncts": [],
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "evidence_token_indices": [],
    }


def _build_clause_candidates(text: str, tokens: Sequence[TokenAtom], role_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    role_map = _candidate_map(role_candidates)
    word_idxs = _word_indices(tokens)
    if not word_idxs:
        return []
    norms = [_norm_token(tok.text) for tok in tokens]
    first_idx = word_idxs[0]
    first = norms[first_idx]
    last_word_idx = word_idxs[-1]
    is_question = str(text or "").strip().endswith("?") or first in QUESTION_WORDS | MODALS | AUXILIARIES
    negated = any(norms[i] in NEGATIONS for i in word_idxs)
    out: list[dict[str, Any]] = []

    def finalize(frame: dict[str, Any], *, evidence: Iterable[int]) -> dict[str, Any]:
        frame["negated"] = bool(negated)
        frame["evidence_token_indices"] = sorted({int(i) for i in evidence if i is not None and int(i) >= 0})
        frame["adjuncts"] = list(frame.get("adjuncts", []) or [])
        return frame

    # Existential frame: "there is a fox in the yard". "There" is a
    # structural placeholder, not the entity being described.
    if first == "there":
        cop_idx = next((i for i in word_idxs[1:] if norms[i] in COPULAS), -1)
        if cop_idx >= 0:
            entity_np = _next_np_after(cop_idx + 1, tokens, role_map)
            if entity_np:
                entity = _head_of(entity_np, tokens, role_map)
                frame = _clause_base(text, "existential", 0.84)
                frame.update(
                    {
                        "action": "exist",
                        "action_surface": tokens[cop_idx].text,
                        "object": entity,
                        "object_text": _surface(entity_np, tokens),
                        "entity": entity,
                        "adjuncts": _prepositional_adjuncts(tokens, role_map, start_idx=entity_np[-1] + 1),
                    }
                )
                out.append(finalize(frame, evidence=[first_idx, cop_idx] + entity_np))

    # Passive voice: "the door was opened by Haz". Normalize semantic roles so
    # memory can compare it with "Haz opened the door".
    by_idx = next((i for i in word_idxs if norms[i] == "by"), -1)
    aux_idx = next((i for i in word_idxs if norms[i] in COPULAS), -1)
    if by_idx > 0 and aux_idx >= 0 and aux_idx < by_idx - 1:
        verb_pick = _first_main_verb_after(aux_idx, tokens, role_candidates)
        if verb_pick and verb_pick[0] < by_idx:
            verb_idx, verb_score = verb_pick
            patient_np = _noun_phrase_ending_at(aux_idx - 1, tokens, role_map)
            agent_np = _next_np_after(by_idx + 1, tokens, role_map)
            if patient_np and agent_np:
                frame = _clause_base(text, "passive", 0.80 + min(0.12, verb_score * 0.10))
                frame.update(
                    {
                        "voice": "passive",
                        "subject": _head_of(patient_np, tokens, role_map),
                        "subject_text": _surface(patient_np, tokens),
                        "action": _simple_lemma(tokens[verb_idx].text),
                        "action_surface": tokens[verb_idx].text,
                        "object": _head_of(patient_np, tokens, role_map),
                        "object_text": _surface(patient_np, tokens),
                        "agent": _head_of(agent_np, tokens, role_map),
                        "patient": _head_of(patient_np, tokens, role_map),
                    }
                )
                out.append(finalize(frame, evidence=patient_np + [aux_idx, verb_idx, by_idx] + agent_np))

    # WH question normalization.
    if first in QUESTION_WORDS:
        query_target = {
            "what": "object", "who": "subject", "whom": "object", "whose": "owner", "where": "location",
            "when": "time", "why": "cause", "how": "manner", "which": "selection",
        }.get(first, "unknown")
        aux_question_idx = next((i for i in word_idxs[1:] if norms[i] in AUXILIARIES | MODALS), -1)
        start = aux_question_idx if aux_question_idx >= 0 else first_idx
        verb_pick = _first_main_verb_after(start, tokens, role_candidates)
        if verb_pick:
            verb_idx, verb_score = verb_pick
            subject_np = _subject_np_before_verb(verb_idx, tokens, role_map)
            # Avoid swallowing the leading WH operator into subject.
            subject_np = [i for i in subject_np if i != first_idx and norms[i] not in AUXILIARIES | MODALS]
            object_np = _next_np_after(verb_idx + 1, tokens, role_map)
            frame = _clause_base(text, "question", 0.72 + min(0.16, verb_score * 0.12))
            frame.update(
                {
                    "subject": _head_of(subject_np, tokens, role_map),
                    "subject_text": _surface(subject_np, tokens),
                    "action": _simple_lemma(tokens[verb_idx].text),
                    "action_surface": tokens[verb_idx].text,
                    "object": _head_of(object_np, tokens, role_map),
                    "object_text": _surface(object_np, tokens),
                    "query_target": query_target,
                    "question_word": first,
                    "adjuncts": _prepositional_adjuncts(tokens, role_map, start_idx=verb_idx + 1),
                }
            )
            out.append(finalize(frame, evidence=[first_idx, verb_idx] + subject_np + object_np))

    # Yes/no / modal questions: "Can you patch it?"
    if first in MODALS | AUXILIARIES and first not in QUESTION_WORDS:
        verb_pick = _first_main_verb_after(first_idx, tokens, role_candidates)
        if verb_pick:
            verb_idx, verb_score = verb_pick
            subject_np = _subject_np_before_verb(verb_idx, tokens, role_map)
            subject_np = [i for i in subject_np if norms[i] not in MODALS | AUXILIARIES]
            object_np = _next_np_after(verb_idx + 1, tokens, role_map)
            frame = _clause_base(text, "question", 0.70 + min(0.18, verb_score * 0.14))
            frame.update(
                {
                    "subject": _head_of(subject_np, tokens, role_map),
                    "subject_text": _surface(subject_np, tokens),
                    "action": _simple_lemma(tokens[verb_idx].text),
                    "action_surface": tokens[verb_idx].text,
                    "object": _head_of(object_np, tokens, role_map),
                    "object_text": _surface(object_np, tokens),
                    "query_target": "truth_value",
                    "auxiliary": first,
                    "adjuncts": _prepositional_adjuncts(tokens, role_map, start_idx=verb_idx + 1),
                }
            )
            out.append(finalize(frame, evidence=[first_idx, verb_idx] + subject_np + object_np))

    # Imperative. The actor is structurally implied rather than hallucinated
    # from memory: English command form defaults to the listener.
    first_verb_score = _role_score(role_map.get(first_idx, {}), "verb")
    first_lemma = _simple_lemma(tokens[first_idx].text)
    if not is_question and (first_lemma in ACTION_WORDS or first_verb_score >= 0.62):
        frame = _clause_base(text, "imperative", 0.80 if first_lemma in ACTION_WORDS else 0.66)
        object_np = _next_np_after(first_idx + 1, tokens, role_map)
        object_text = _surface(object_np, tokens)
        object_head = _head_of(object_np, tokens, role_map)
        if first_lemma in LITERAL_CONTENT_VERBS:
            payload_indices = [i for i in word_idxs if i > first_idx and norms[i] != "please"]
            object_text = _surface(payload_indices, tokens)
            object_head = object_text.lower()
        frame.update(
            {
                "subject": "you",
                "subject_text": "you",
                "subject_implied": True,
                "action": first_lemma,
                "action_surface": tokens[first_idx].text,
                "object": object_head,
                "object_text": object_text,
                "adjuncts": _prepositional_adjuncts(tokens, role_map, start_idx=first_idx + 1),
            }
        )
        out.append(finalize(frame, evidence=[first_idx] + object_np))

    # Copular declarative: "the sky is blue". Skip if the copula is merely
    # supporting a passive/progressive predicate.
    if not is_question and first != "there":
        for cop_idx in word_idxs:
            if norms[cop_idx] not in COPULAS:
                continue
            verb_after = _first_main_verb_after(cop_idx, tokens, role_candidates)
            if verb_after and verb_after[0] == cop_idx + 1 and (
                _norm_token(tokens[verb_after[0]].text).endswith(("ing", "ed")) or by_idx > verb_after[0]
            ):
                continue
            subject_np = _noun_phrase_ending_at(cop_idx - 1, tokens, role_map)
            complement_indices = [i for i in word_idxs if i > cop_idx and norms[i] not in {"please"}]
            if subject_np and complement_indices:
                frame = _clause_base(text, "copular", 0.80)
                comp_head = _head_of(complement_indices, tokens, role_map)
                comp_role = _best_role(role_map, complement_indices[-1])
                frame.update(
                    {
                        "subject": _head_of(subject_np, tokens, role_map),
                        "subject_text": _surface(subject_np, tokens),
                        "action": "be",
                        "action_surface": tokens[cop_idx].text,
                        "complement": _surface(complement_indices, tokens),
                        "attribute": _surface(complement_indices, tokens) if comp_role == "adjective" else "",
                        "identity": comp_head if comp_role in {"noun", "proper_noun", "pronoun"} else "",
                    }
                )
                out.append(finalize(frame, evidence=subject_np + [cop_idx] + complement_indices))
            break

    # Standard declarative SVO. The predicate is chosen after the best
    # entity-like span, allowing an unknown middle word to become a verb based
    # on English word order.
    if not is_question:
        verb_cands = _verb_candidates(tokens, role_candidates)
        for verb_idx, verb_score in verb_cands[:3]:
            if verb_idx == first_idx and first_lemma in ACTION_WORDS:
                continue  # imperative already models this more accurately
            norm = norms[verb_idx]
            if norm in MODALS or norm in COPULAS:
                continue
            subject_np = _subject_np_before_verb(verb_idx, tokens, role_map)
            if not subject_np:
                continue
            object_np = _next_np_after(verb_idx + 1, tokens, role_map)
            frame = _clause_base(text, "declarative", 0.55 + min(0.30, verb_score * 0.30))
            frame.update(
                {
                    "subject": _head_of(subject_np, tokens, role_map),
                    "subject_text": _surface(subject_np, tokens),
                    "action": _simple_lemma(tokens[verb_idx].text),
                    "action_surface": tokens[verb_idx].text,
                    "object": _head_of(object_np, tokens, role_map),
                    "object_text": _surface(object_np, tokens),
                    "adjuncts": _prepositional_adjuncts(tokens, role_map, start_idx=(object_np[-1] + 1 if object_np else verb_idx + 1)),
                }
            )
            out.append(finalize(frame, evidence=subject_np + [verb_idx] + object_np))

    # Explicit ambiguity shelf for perception constructions such as
    # "I saw her duck."  Both interpretations are useful candidates until
    # context supplies discriminating evidence.
    if len(word_idxs) >= 4:
        main_verb_idx = next((i for i in word_idxs if _simple_lemma(tokens[i].text) in PERCEPTION_VERBS), -1)
        if main_verb_idx >= 0:
            after = [i for i in word_idxs if i > main_verb_idx]
            if len(after) == 2 and norms[after[0]] in OBJECT_PRONOUNS | POSSESSIVE_PRONOUNS:
                ambiguous_idx = after[1]
                noun_score = _role_score(role_map.get(ambiguous_idx, {}), "noun")
                verb_score = _role_score(role_map.get(ambiguous_idx, {}), "verb")
                if noun_score >= 0.30 and verb_score >= 0.25:
                    subject_np = _noun_phrase_ending_at(main_verb_idx - 1, tokens, role_map)
                    common = {
                        "subject": _head_of(subject_np, tokens, role_map),
                        "subject_text": _surface(subject_np, tokens),
                        "action": _simple_lemma(tokens[main_verb_idx].text),
                        "action_surface": tokens[main_verb_idx].text,
                    }
                    object_frame = _clause_base(text, "declarative_ambiguous", 0.55)
                    object_frame.update(common)
                    object_frame.update(
                        {
                            "object": _simple_lemma(tokens[ambiguous_idx].text),
                            "object_text": f"{tokens[after[0]].text} {tokens[ambiguous_idx].text}",
                            "ambiguity": "possessive_noun",
                        }
                    )
                    out.append(finalize(object_frame, evidence=subject_np + [main_verb_idx] + after))

                    action_frame = _clause_base(text, "declarative_ambiguous", 0.54)
                    action_frame.update(common)
                    action_frame.update(
                        {
                            "object": _simple_lemma(tokens[after[0]].text),
                            "object_text": tokens[after[0]].text,
                            "embedded_action": _simple_lemma(tokens[ambiguous_idx].text),
                            "ambiguity": "object_plus_embedded_action",
                        }
                    )
                    out.append(finalize(action_frame, evidence=subject_np + [main_verb_idx] + after))

    if not out:
        frame = _clause_base(text, "fragment", 0.36)
        np = _noun_phrase_starting_at(first_idx, tokens, role_map)
        frame.update(
            {
                "subject": _head_of(np, tokens, role_map),
                "subject_text": _surface(np, tokens),
            }
        )
        out.append(finalize(frame, evidence=np))

    # De-duplicate semantically identical parses, then rank. A small complexity
    # penalty keeps the simplest adequate interpretation on top.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for frame in out:
        key = (
            frame.get("clause_type"), frame.get("voice"), frame.get("subject"), frame.get("action"), frame.get("object"),
            frame.get("complement"), frame.get("query_target"), frame.get("embedded_action", ""), frame.get("ambiguity", ""),
        )
        prior = unique.get(key)
        if prior is None or float(frame.get("confidence", 0.0) or 0.0) > float(prior.get("confidence", 0.0) or 0.0):
            unique[key] = frame

    ranked = sorted(unique.values(), key=lambda f: float(f.get("confidence", 0.0) or 0.0), reverse=True)
    return ranked[:6]


def _reinforce_roles_from_clause(
    role_candidates: list[dict[str, Any]],
    tokens: Sequence[TokenAtom],
    best_clause: dict[str, Any],
) -> None:
    """Feed a coherent clause interpretation back into token-role confidence.

    This is the structure-first step: once English word order strongly suggests
    "dax" is the subject and "snorp" is the predicate, those positions become
    evidence about the previously unknown words.
    """
    if not best_clause:
        return
    evidence = {int(i) for i in best_clause.get("evidence_token_indices", []) if isinstance(i, int) or str(i).isdigit()}

    def boost(idx: int, role: str, amount: float, reason: str) -> None:
        if idx < 0 or idx >= len(role_candidates):
            return
        candidate = role_candidates[idx]
        items = [dict(x) for x in list(candidate.get("candidates", []) or []) if isinstance(x, dict)]
        found = False
        for item in items:
            if str(item.get("role", "") or "") != role:
                continue
            item["score"] = round(min(0.99, float(item.get("score", 0.0) or 0.0) + amount), 4)
            ev = list(item.get("evidence", []) or [])
            if reason not in ev:
                ev.append(reason)
            item["evidence"] = ev[:6]
            found = True
            break
        if not found:
            items.append({"role": role, "score": round(min(0.99, amount), 4), "evidence": [reason]})
        items.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        candidate["candidates"] = items[:4]
        candidate["best_role"] = str(items[0].get("role", "unknown") or "unknown")
        best = float(items[0].get("score", 0.0) or 0.0)
        second = float(items[1].get("score", 0.0) or 0.0) if len(items) > 1 else 0.0
        candidate["confidence"] = round(max(0.0, min(0.99, best - 0.20 * second)), 4)

    action_surface = _norm_token(best_clause.get("action_surface", ""))
    if action_surface:
        action_idx = next((i for i, tok in enumerate(tokens) if i in evidence and _norm_token(tok.text) == action_surface), -1)
        boost(action_idx, "verb", 0.36, "best_clause_predicate")

    for field_name, role in (("subject", "noun"), ("agent", "noun"), ("object", "noun"), ("patient", "noun")):
        value = _norm_token(best_clause.get(field_name, ""))
        if not value:
            continue
        matches = [i for i, tok in enumerate(tokens) if i in evidence and _simple_lemma(tok.text) == value]
        if matches:
            idx = matches[-1] if field_name in {"subject", "patient"} else matches[0]
            # Do not erase a real pronoun role just because it fills an entity slot.
            if _norm_token(tokens[idx].text) in PRONOUNS:
                boost(idx, "pronoun", 0.28, f"best_clause_{field_name}")
            else:
                boost(idx, role, 0.34, f"best_clause_{field_name}")


def _apply_best_clause_dependencies(tokens: Sequence[TokenAtom], best_clause: dict[str, Any]) -> None:
    if not best_clause:
        return
    evidence = set(int(i) for i in best_clause.get("evidence_token_indices", []) if isinstance(i, int) or str(i).isdigit())
    action_surface = _norm_token(best_clause.get("action_surface", ""))
    action_idx = next((i for i, tok in enumerate(tokens) if _norm_token(tok.text) == action_surface and i in evidence), -1)
    subj_head = _norm_token(best_clause.get("subject", ""))
    obj_head = _norm_token(best_clause.get("object", ""))
    for i, tok in enumerate(tokens):
        lemma = _simple_lemma(tok.text)
        if i == action_idx:
            tok.dep = "ROOT"
            tok.head_idx = i
            tok.head_text = tok.text
            tok.head_lemma = tok.lemma
        elif subj_head and lemma == subj_head and i in evidence:
            tok.dep = "nsubjpass" if best_clause.get("voice") == "passive" else "nsubj"
            tok.head_idx = action_idx
            if 0 <= action_idx < len(tokens):
                tok.head_text = tokens[action_idx].text
                tok.head_lemma = tokens[action_idx].lemma
        elif obj_head and lemma == obj_head and i in evidence:
            tok.dep = "obj"
            tok.head_idx = action_idx
            if 0 <= action_idx < len(tokens):
                tok.head_text = tokens[action_idx].text
                tok.head_lemma = tokens[action_idx].lemma


# ---------------------------------------------------------------------------
# Public parse API
# ---------------------------------------------------------------------------

def _fallback_parse(text: str) -> ParsedText:
    raw_tokens = TOKEN_RE.findall(str(text or ""))
    role_candidates = _infer_role_candidates_from_words(raw_tokens)
    token_atoms: list[TokenAtom] = []
    for idx, raw in enumerate(raw_tokens):
        candidate = role_candidates[idx] if idx < len(role_candidates) else {}
        pos, dep = _fallback_pos_from_candidate(candidate)
        norm = _norm_token(raw)
        token_atoms.append(
            TokenAtom(
                text=raw,
                lemma=_simple_lemma(raw),
                pos=pos,
                dep=dep,
                is_stop=norm in DETERMINERS | PREPOSITIONS | CONNECTORS | MODALS | AUXILIARIES,
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

    phrase_chunks = _build_phrase_chunks(token_atoms, role_candidates)
    clause_candidates = _build_clause_candidates(text, token_atoms, role_candidates)
    best_clause = dict(clause_candidates[0]) if clause_candidates else {}

    # One bounded feedback pass lets coherent sentence structure refine unknown
    # word roles without creating a self-reinforcing parse loop.
    _reinforce_roles_from_clause(role_candidates, token_atoms, best_clause)
    for idx, atom in enumerate(token_atoms):
        if idx < len(role_candidates):
            pos, fallback_dep = _fallback_pos_from_candidate(role_candidates[idx])
            atom.pos = pos
            if atom.dep == "dep":
                atom.dep = fallback_dep
    phrase_chunks = _build_phrase_chunks(token_atoms, role_candidates)
    clause_candidates = _build_clause_candidates(text, token_atoms, role_candidates)
    best_clause = dict(clause_candidates[0]) if clause_candidates else {}
    _apply_best_clause_dependencies(token_atoms, best_clause)
    noun_chunks = [str(p.get("text", "") or "") for p in phrase_chunks if p.get("kind") == "noun_phrase" and p.get("text")]
    return ParsedText(
        text=str(text or ""),
        sentences=_split_sentences(text),
        tokens=token_atoms,
        noun_chunks=noun_chunks,
        entities=[],
        phrase_chunks=phrase_chunks,
        role_candidates=role_candidates,
        clause_candidates=clause_candidates,
        best_clause=best_clause,
    )


def parse_text(text: str) -> ParsedText:
    nlp = _get_nlp()
    if nlp is None:
        return _fallback_parse(text)

    doc = nlp(text)
    tokens = [
        TokenAtom(
            text=t.text,
            lemma=_simple_lemma(t.lemma_ or t.text),
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
            head_lemma=_simple_lemma(t.head.lemma_ if t.head is not None else ""),
            ent_type=t.ent_type_,
        )
        for t in doc
    ]
    role_candidates = _infer_role_candidates_from_words([t.text for t in tokens], spacy_tokens=tokens)
    phrase_chunks = _build_phrase_chunks(tokens, role_candidates)
    clause_candidates = _build_clause_candidates(text, tokens, role_candidates)
    best_clause = dict(clause_candidates[0]) if clause_candidates else {}

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    # Keep any useful heuristic chunk that the dependency model did not emit.
    seen_chunks = {c.lower() for c in noun_chunks}
    for chunk in phrase_chunks:
        if chunk.get("kind") != "noun_phrase":
            continue
        value = str(chunk.get("text", "") or "").strip()
        if value and value.lower() not in seen_chunks:
            noun_chunks.append(value)
            seen_chunks.add(value.lower())

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    sentences = [sent.text.strip() for sent in doc.sents]

    return ParsedText(
        text=text,
        sentences=sentences,
        tokens=tokens,
        noun_chunks=noun_chunks,
        entities=entities,
        phrase_chunks=phrase_chunks,
        role_candidates=role_candidates,
        clause_candidates=clause_candidates,
        best_clause=best_clause,
    )


def analyze_english_structure(text: str) -> dict[str, Any]:
    """Return the reusable structure bundle used by input and reading memory.

    The result is deliberately probabilistic: role and clause candidates are
    hypotheses that downstream cognition may accept, compare, or investigate.
    Reading chunks may contain several sentences, so sentence-local structures
    are also returned rather than pretending the whole paragraph is one clause.
    """

    parsed = parse_text(text)
    sentence_structures: list[dict[str, Any]] = []
    sentences = list(parsed.sentences) or ([str(text or "").strip()] if str(text or "").strip() else [])

    if len(sentences) <= 1:
        sentence_structures.append(
            {
                "sentence_index": 0,
                "text": sentences[0] if sentences else str(text or ""),
                "phrase_chunks": [dict(x) for x in parsed.phrase_chunks],
                "role_candidates": [dict(x) for x in parsed.role_candidates],
                "clause_candidates": [dict(x) for x in parsed.clause_candidates],
                "best_clause": dict(parsed.best_clause),
            }
        )
    else:
        for sentence_index, sentence in enumerate(sentences):
            sub = parse_text(sentence)
            sentence_structures.append(
                {
                    "sentence_index": sentence_index,
                    "text": sentence,
                    "phrase_chunks": [dict(x) for x in sub.phrase_chunks],
                    "role_candidates": [dict(x) for x in sub.role_candidates],
                    "clause_candidates": [dict(x) for x in sub.clause_candidates],
                    "best_clause": dict(sub.best_clause),
                }
            )

    all_clauses = [
        dict(frame)
        for sentence in sentence_structures
        for frame in list(sentence.get("clause_candidates", []) or [])
        if isinstance(frame, dict)
    ]
    best_clause = dict(sentence_structures[0].get("best_clause", {}) or {}) if sentence_structures else {}
    return {
        "schema": "language.structure.v1",
        "text": str(text or ""),
        "sentences": sentences,
        "phrase_chunks": [dict(x) for x in parsed.phrase_chunks],
        "role_candidates": [dict(x) for x in parsed.role_candidates],
        "clause_candidates": all_clauses or [dict(x) for x in parsed.clause_candidates],
        "best_clause": best_clause or dict(parsed.best_clause),
        "sentence_structures": sentence_structures,
    }

