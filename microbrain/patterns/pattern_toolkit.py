from __future__ import annotations

"""Reusable, non-speaking pattern analyses for MicroBrain.

The toolkit is intentionally pure and cheap.  It accepts a built context packet
and returns serializable observations.  It does not decide truth, write memory,
or emit speech; higher organs (notably the hypothesis engine) decide how much
weight to place on the observations.
"""

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Rolling context limits used by the lightweight pattern pass.
NEAR_TURN_LIMIT = 8
COOCCURRENCE_TURN_LIMIT = 12
COOCCURRENCE_RESULT_LIMIT = 8
MEMORY_SOURCE_LIMIT = 8
MEMORY_MATCH_LIMIT = 6
PATTERN_EVIDENCE_LIMIT = 12

# Default novelty when no prior turn exists.
NOVELTY_WITHOUT_HISTORY = 0.50

# Statement classification confidence defaults.
STATEMENT_KIND_CONFIDENCE = {
    "question": 0.98,
    "request": 0.92,
    "correction": 0.90,
    "disagreement": 0.86,
    "agreement": 0.84,
    "closure": 0.84,
    "personal_state": 0.82,
    "status_update": 0.76,
    "claim": 0.72,
    "greeting": 0.95,
    "minimal_statement": 0.52,
    "statement": 0.60,
}

# Continuity and contradiction fusion weights.
CONTINUITY_NEAR_TURN_WEIGHT = 0.68
CONTINUITY_SUMMARY_WEIGHT = 0.32
CONTRADICTION_CORRECTION_BASE = 0.42
CONTRADICTION_NEGATION_BASE = 0.28
CONTRADICTION_SIMILARITY_WEIGHT = 0.58
CONTRADICTION_PATTERN_MIN = 0.20

# Sequence pattern scoring.
RESPONSE_SEQUENCE_BASE = 0.62
RESPONSE_SEQUENCE_CONTINUITY_WEIGHT = 0.25
THREAD_SEQUENCE_MIN_CONTINUITY = 0.24
THREAD_SEQUENCE_BASE = 0.48
THREAD_SEQUENCE_CONTINUITY_WEIGHT = 0.45

# Consequence/risk scoring.
RISK_TERM_WEIGHT = 0.34

# Expected conversational response demand by statement kind.
RESPONSE_EXPECTATION_BASE = {
    "question": 0.94,
    "request": 0.90,
    "correction": 0.82,
    "disagreement": 0.74,
    "agreement": 0.55,
    "greeting": 0.88,
    "personal_state": 0.72,
    "status_update": 0.64,
    "claim": 0.57,
    "statement": 0.46,
    "minimal_statement": 0.28,
    "closure": 0.20,
}
DEFAULT_RESPONSE_EXPECTATION = 0.42
RESPONSE_EXPECTATION_CONTINUITY_WEIGHT = 0.20
RESPONSE_EXPECTATION_KIND_CONFIDENCE_WEIGHT = 0.08
RESPONSE_EXPECTATION_RISK_WEIGHT = 0.12
DIRECT_ADDRESS_EXPECTATION_BONUS = 0.15
SOCIAL_REPLY_EXPECTATION_BONUS = 0.12
NO_MEANINGFUL_TOKEN_PENALTY = 0.18
VERY_SHORT_TEXT_PENALTY = 0.12
CLOSURE_CONTINUITY_PENALTY = 0.08

# Uncertainty shaping.
AMBIGUOUS_STATEMENT_UNCERTAINTY = 0.18
LOW_INFORMATION_UNCERTAINTY = 0.18
LOW_CONTINUITY_THRESHOLD = 0.10
LOW_CONTINUITY_UNCERTAINTY = 0.10
CONTRADICTION_UNCERTAINTY_WEIGHT = 0.22
MEMORY_SUPPORT_UNCERTAINTY_REDUCTION = 0.10

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "how", "i", "if", "in", "is", "it",
    "its", "me", "mine", "my", "of", "on", "or", "our", "ours", "she", "so",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your", "yours",
}

_NEGATION = {"no", "not", "never", "none", "nothing", "isn't", "wasn't", "won't", "can't", "cannot", "don't", "doesn't", "didn't"}
_CORRECTION_MARKERS = (
    "actually", "rather", "not quite", "i mean", "that's not", "that is not",
    "correction", "instead", "more accurately", "to clarify",
)
_AGREEMENT_MARKERS = (
    "yes", "yeah", "yep", "exactly", "right", "correct", "true", "agreed",
    "that makes sense", "pretty much",
)
_DISAGREEMENT_MARKERS = (
    "no", "wrong", "i disagree", "not really", "i don't think", "doesn't fit",
    "that isn't", "that is not",
)
_CLOSURE_MARKERS = (
    "bye", "goodbye", "good night", "goodnight", "that's all", "thats all",
    "done for now", "talk later", "see you later", "ty", "thanks", "thank you",
)
_STATUS_MARKERS = (
    "still", "finished", "done", "working", "running", "stuck", "currently",
    "right now", "today", "just", "progress", "started", "stopped", "loaded",
    "ingesting", "reading", "building", "testing", "failed", "passed",
)
_PERSONAL_STATE_MARKERS = (
    "i feel", "i'm feeling", "i am feeling", "i'm tired", "i am tired", "i'm okay",
    "i am okay", "i'm fine", "i am fine", "long day", "rough day", "good day",
    "i'm worried", "i am worried", "i'm excited", "i am excited",
)
_REQUEST_PREFIXES = (
    "please ", "can you ", "could you ", "would you ", "will you ", "help me ",
    "show me ", "tell me ", "give me ", "make ", "build ", "add ", "remove ",
    "check ", "look at ", "find ", "explain ", "rewrite ", "list ",
)
_CLAIM_VERBS = (
    " is ", " are ", " means ", " becomes ", " creates ", " allows ", " causes ",
    " should ", " would ", " could ", " = ", " -> ", " leads to ", " results in ",
)
_RISK_TERMS = {
    "danger", "dangerous", "emergency", "hurt", "injury", "injured", "kill",
    "medical", "medicine", "legal", "law", "money", "financial", "fire", "smoke",
    "password", "credential", "weapon", "unsafe", "overheat", "failure", "crash",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text or "").lower()))


def tokenize(text: str, *, meaningful: bool = False) -> List[str]:
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    if not meaningful:
        return tokens
    return [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _contains_phrase(normalized: str, phrases: Sequence[str]) -> bool:
    if not normalized:
        return False
    padded = f" {normalized} "
    for phrase in phrases:
        p = normalize_text(phrase)
        if p and (padded.startswith(f" {p} ") or f" {p} " in padded):
            return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _memory_text(item: Mapping[str, Any]) -> str:
    text = str(item.get("text", "") or item.get("anchor_text", "") or "").strip()
    if text:
        return text
    refs = item.get("refs", [])
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, Mapping):
                value = str(ref.get("value", "") or "").strip()
            else:
                value = str(ref or "").strip()
            if value:
                return value
    return ""


def _evidence_source(item: Mapping[str, Any]) -> str:
    meta = item.get("meta", {}) if isinstance(item.get("meta", {}), Mapping) else {}
    return str(
        item.get("source", "")
        or meta.get("source", "")
        or meta.get("memory_source", "")
        or meta.get("role", "")
        or item.get("tier", "")
        or "unknown"
    )


class PatternToolkit:
    """Analyze a context frame without owning belief or action selection."""

    def analyze(
        self,
        context: Mapping[str, Any],
        *,
        memory_evidence: Sequence[Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        text = str(input_block.get("text", "") or "").strip()
        normalized = normalize_text(text)
        tokens = tokenize(text)
        meaningful_tokens = tokenize(text, meaningful=True)
        cues = context.get("cues", {}) if isinstance(context.get("cues", {}), Mapping) else {}
        scene = context.get("conversation_scene", {}) if isinstance(context.get("conversation_scene", {}), Mapping) else {}
        summary = context.get("conversation_summary", {}) if isinstance(context.get("conversation_summary", {}), Mapping) else {}

        turns = self._near_turns(scene, current_text=text, current_correlation_id=str(input_block.get("correlation_id", "") or ""))
        prior_turns = turns[-NEAR_TURN_LIMIT:]
        prior_user_turns = [turn for turn in prior_turns if str(turn.get("role", "")) == "user"]
        prior_assistant_turns = [turn for turn in prior_turns if str(turn.get("role", "")) == "assistant"]

        kind, kind_confidence, kind_evidence = self._classify_statement(text, normalized, cues)
        continuity = self._continuity_score(
            meaningful_tokens=meaningful_tokens,
            prior_turns=prior_turns,
            summary=summary,
        )
        novelty = clamp(1.0 - continuity if prior_turns else NOVELTY_WITHOUT_HISTORY)
        contradiction = self._contradiction_score(
            normalized=normalized,
            meaningful_tokens=meaningful_tokens,
            prior_turns=prior_turns,
            kind=kind,
        )
        cooccurrence = self._cooccurrence(meaningful_tokens, prior_turns)
        sequence = self._sequence_pattern(kind, continuity, prior_user_turns, prior_assistant_turns)
        memory_summary = self._memory_patterns(meaningful_tokens, memory_evidence or [])
        risk = self._risk_score(meaningful_tokens)
        response_expectation = self._response_expectation(
            kind=kind,
            kind_confidence=kind_confidence,
            continuity=continuity,
            cues=cues,
            meaningful_count=len(meaningful_tokens),
            risk=risk,
            text=text,
        )
        uncertainty = self._uncertainty(
            kind=kind,
            kind_confidence=kind_confidence,
            continuity=continuity,
            contradiction=contradiction,
            meaningful_count=len(meaningful_tokens),
            memory_summary=memory_summary,
        )

        patterns: List[Dict[str, Any]] = [
            self._pattern(
                "statement_kind",
                kind_confidence,
                kind_evidence,
                {"kind": kind},
            ),
            self._pattern(
                "conversation_continuity",
                continuity,
                self._continuity_evidence(meaningful_tokens, prior_turns, summary),
                {"near_turn_count": len(prior_turns)},
            ),
            self._pattern(
                "novelty",
                novelty,
                ["current_vs_near_window"],
                {"continuity_inverse": round(1.0 - novelty, 4)},
            ),
        ]

        if contradiction >= CONTRADICTION_PATTERN_MIN:
            patterns.append(
                self._pattern(
                    "contradiction_candidate",
                    contradiction,
                    ["negation_or_correction", "similar_prior_turn"],
                    {},
                )
            )
        if cooccurrence:
            patterns.append(
                self._pattern(
                    "cooccurrence",
                    cooccurrence[0][1],
                    [f"token:{cooccurrence[0][0]}"],
                    {"tokens": [token for token, _ in cooccurrence[:6]]},
                )
            )
        if sequence:
            patterns.append(sequence)
        if memory_summary["match_count"]:
            patterns.append(
                self._pattern(
                    "memory_recurrence",
                    memory_summary["top_similarity"],
                    [f"memory_matches:{memory_summary['match_count']}"],
                    memory_summary,
                )
            )
        if risk > 0.0:
            patterns.append(
                self._pattern(
                    "consequence_salience",
                    risk,
                    [f"risk_terms:{','.join(memory_summary.get('risk_terms', []) or self._risk_terms(meaningful_tokens))}"],
                    {"risk_terms": self._risk_terms(meaningful_tokens)},
                )
            )

        return {
            "schema_ver": "pattern.analysis.v1",
            "text": text,
            "normalized": normalized,
            "tokens": tokens[:48],
            "meaningful_tokens": meaningful_tokens[:24],
            "statement_kind": kind,
            "statement_kind_confidence": round(kind_confidence, 4),
            "continuity": round(continuity, 4),
            "novelty": round(novelty, 4),
            "contradiction": round(contradiction, 4),
            "risk": round(risk, 4),
            "response_expectation": round(response_expectation, 4),
            "uncertainty": round(uncertainty, 4),
            "near_turn_count": len(prior_turns),
            "memory": memory_summary,
            "patterns": patterns,
        }

    def _near_turns(
        self,
        scene: Mapping[str, Any],
        *,
        current_text: str,
        current_correlation_id: str,
    ) -> List[Dict[str, Any]]:
        raw_turns = list(scene.get("turns", []) or [])
        turns = [dict(turn) for turn in raw_turns if isinstance(turn, Mapping)]
        current_norm = normalize_text(current_text)
        # conversation.scene normally receives percept/text before context/request is
        # processed. Remove exactly that current turn so the near window does not
        # claim perfect continuity with itself.
        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            same_corr = bool(current_correlation_id) and str(turn.get("correlation_id", "") or "") == current_correlation_id
            same_text = current_norm and normalize_text(str(turn.get("text", "") or "")) == current_norm
            if str(turn.get("role", "")) == "user" and (same_corr or same_text):
                del turns[index]
                break
        return turns

    def _classify_statement(
        self,
        text: str,
        normalized: str,
        cues: Mapping[str, Any],
    ) -> Tuple[str, float, List[str]]:
        lowered = str(text or "").strip().lower()
        if cues.get("is_question") or text.rstrip().endswith("?"):
            return "question", STATEMENT_KIND_CONFIDENCE["question"], ["question_form"]
        if lowered.startswith(_REQUEST_PREFIXES):
            return "request", STATEMENT_KIND_CONFIDENCE["request"], ["request_prefix"]
        if _contains_phrase(normalized, _CORRECTION_MARKERS):
            return "correction", STATEMENT_KIND_CONFIDENCE["correction"], ["correction_marker"]
        if _contains_phrase(normalized, _DISAGREEMENT_MARKERS):
            return "disagreement", STATEMENT_KIND_CONFIDENCE["disagreement"], ["disagreement_marker"]
        if _contains_phrase(normalized, _AGREEMENT_MARKERS):
            return "agreement", STATEMENT_KIND_CONFIDENCE["agreement"], ["agreement_marker"]
        if _contains_phrase(normalized, _CLOSURE_MARKERS):
            return "closure", STATEMENT_KIND_CONFIDENCE["closure"], ["closure_marker"]
        if _contains_phrase(normalized, _PERSONAL_STATE_MARKERS):
            return "personal_state", STATEMENT_KIND_CONFIDENCE["personal_state"], ["personal_state_marker"]
        if _contains_phrase(normalized, _STATUS_MARKERS):
            return "status_update", STATEMENT_KIND_CONFIDENCE["status_update"], ["status_marker"]
        padded = f" {lowered} "
        if any(verb in padded for verb in _CLAIM_VERBS):
            return "claim", STATEMENT_KIND_CONFIDENCE["claim"], ["declarative_relation"]
        if cues.get("is_greeting"):
            return "greeting", STATEMENT_KIND_CONFIDENCE["greeting"], ["greeting_cue"]
        if len(tokenize(text, meaningful=True)) <= 1:
            return "minimal_statement", STATEMENT_KIND_CONFIDENCE["minimal_statement"], ["low_information"]
        return "statement", STATEMENT_KIND_CONFIDENCE["statement"], ["declarative_default"]

    def _continuity_score(
        self,
        *,
        meaningful_tokens: Sequence[str],
        prior_turns: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> float:
        if not meaningful_tokens:
            return 0.0
        best_turn = 0.0
        for turn in prior_turns[-NEAR_TURN_LIMIT:]:
            turn_tokens = turn.get("tokens", [])
            if not isinstance(turn_tokens, list) or not turn_tokens:
                turn_tokens = tokenize(str(turn.get("text", "") or ""), meaningful=True)
            best_turn = max(best_turn, jaccard(meaningful_tokens, [str(token) for token in turn_tokens]))

        active = list(summary.get("active_objects", []) or []) + list(summary.get("active_threads", []) or [])
        summary_overlap = jaccard(meaningful_tokens, [str(token) for token in active]) if active else 0.0
        return clamp((CONTINUITY_NEAR_TURN_WEIGHT * best_turn) + (CONTINUITY_SUMMARY_WEIGHT * summary_overlap))

    def _continuity_evidence(
        self,
        meaningful_tokens: Sequence[str],
        prior_turns: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> List[str]:
        evidence: List[str] = []
        active = {str(token) for token in list(summary.get("active_objects", []) or []) + list(summary.get("active_threads", []) or [])}
        overlap = [token for token in meaningful_tokens if token in active]
        if overlap:
            evidence.append(f"active_overlap:{','.join(overlap[:6])}")
        if prior_turns:
            evidence.append(f"near_turns:{len(prior_turns)}")
        if not evidence:
            evidence.append("no_local_overlap")
        return evidence

    def _contradiction_score(
        self,
        *,
        normalized: str,
        meaningful_tokens: Sequence[str],
        prior_turns: Sequence[Mapping[str, Any]],
        kind: str,
    ) -> float:
        has_negation = any(token in _NEGATION for token in tokenize(normalized))
        correction = kind in {"correction", "disagreement"}
        if not has_negation and not correction:
            return 0.0
        best_similarity = 0.0
        for turn in prior_turns[-NEAR_TURN_LIMIT:]:
            prior_tokens = tokenize(str(turn.get("text", "") or ""), meaningful=True)
            best_similarity = max(best_similarity, jaccard(meaningful_tokens, prior_tokens))
        base = CONTRADICTION_CORRECTION_BASE if correction else CONTRADICTION_NEGATION_BASE
        return clamp(base + (CONTRADICTION_SIMILARITY_WEIGHT * best_similarity))

    def _cooccurrence(
        self,
        meaningful_tokens: Sequence[str],
        prior_turns: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[str, float]]:
        if not meaningful_tokens or not prior_turns:
            return []
        counter: Counter[str] = Counter()
        for turn in prior_turns[-COOCCURRENCE_TURN_LIMIT:]:
            for token in tokenize(str(turn.get("text", "") or ""), meaningful=True):
                if token in meaningful_tokens:
                    counter[token] += 1
        denom = max(1.0, float(len(prior_turns[-COOCCURRENCE_TURN_LIMIT:])))
        return [(token, clamp(count / denom)) for token, count in counter.most_common(COOCCURRENCE_RESULT_LIMIT)]

    def _sequence_pattern(
        self,
        kind: str,
        continuity: float,
        prior_user_turns: Sequence[Mapping[str, Any]],
        prior_assistant_turns: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any] | None:
        if not prior_user_turns and not prior_assistant_turns:
            return None
        if kind in {"agreement", "disagreement", "correction"} and prior_assistant_turns:
            return self._pattern(
                "response_to_prior_assistant",
                clamp(RESPONSE_SEQUENCE_BASE + (RESPONSE_SEQUENCE_CONTINUITY_WEIGHT * continuity)),
                [f"kind:{kind}", "prior_assistant_turn"],
                {},
            )
        if continuity >= THREAD_SEQUENCE_MIN_CONTINUITY:
            return self._pattern(
                "thread_continuation",
                clamp(THREAD_SEQUENCE_BASE + (THREAD_SEQUENCE_CONTINUITY_WEIGHT * continuity)),
                ["near_window_overlap"],
                {},
            )
        return None

    def _memory_patterns(
        self,
        meaningful_tokens: Sequence[str],
        memory_evidence: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        matches: List[Tuple[float, str, str]] = []
        sources: Set[str] = set()
        for item in memory_evidence:
            if not isinstance(item, Mapping):
                continue
            text = _memory_text(item)
            if not text:
                continue
            similarity = jaccard(meaningful_tokens, tokenize(text, meaningful=True))
            explicit_score = _safe_float(item.get("score", 0.0), 0.0)
            similarity = max(similarity, min(1.0, explicit_score))
            if similarity <= 0.0:
                continue
            source = _evidence_source(item)
            sources.add(source)
            matches.append((similarity, text[:180], source))
        matches.sort(key=lambda row: row[0], reverse=True)
        return {
            "match_count": len(matches),
            "top_similarity": round(matches[0][0], 4) if matches else 0.0,
            "independent_source_count": len(sources),
            "sources": sorted(sources)[:MEMORY_SOURCE_LIMIT],
            "top_matches": [
                {"score": round(score, 4), "text": text, "source": source}
                for score, text, source in matches[:MEMORY_MATCH_LIMIT]
            ],
        }

    def _risk_terms(self, meaningful_tokens: Sequence[str]) -> List[str]:
        return sorted({token for token in meaningful_tokens if token in _RISK_TERMS})

    def _risk_score(self, meaningful_tokens: Sequence[str]) -> float:
        terms = self._risk_terms(meaningful_tokens)
        return clamp(RISK_TERM_WEIGHT * len(terms))

    def _response_expectation(
        self,
        *,
        kind: str,
        kind_confidence: float,
        continuity: float,
        cues: Mapping[str, Any],
        meaningful_count: int,
        risk: float,
        text: str,
    ) -> float:
        base = RESPONSE_EXPECTATION_BASE.get(kind, DEFAULT_RESPONSE_EXPECTATION)
        score = (
            base
            + (RESPONSE_EXPECTATION_CONTINUITY_WEIGHT * continuity)
            + (RESPONSE_EXPECTATION_KIND_CONFIDENCE_WEIGHT * kind_confidence)
            + (RESPONSE_EXPECTATION_RISK_WEIGHT * risk)
        )
        if cues.get("direct_address"):
            score += DIRECT_ADDRESS_EXPECTATION_BONUS
        if cues.get("needs_social_reply") or cues.get("well_wish"):
            score += SOCIAL_REPLY_EXPECTATION_BONUS
        if meaningful_count == 0:
            score -= NO_MEANINGFUL_TOKEN_PENALTY
        if len(str(text or "").strip()) <= 2:
            score -= VERY_SHORT_TEXT_PENALTY
        if kind == "closure":
            score -= CLOSURE_CONTINUITY_PENALTY * continuity
        return clamp(score)

    def _uncertainty(
        self,
        *,
        kind: str,
        kind_confidence: float,
        continuity: float,
        contradiction: float,
        meaningful_count: int,
        memory_summary: Mapping[str, Any],
    ) -> float:
        uncertainty = 1.0 - kind_confidence
        if kind in {"statement", "minimal_statement"}:
            uncertainty += AMBIGUOUS_STATEMENT_UNCERTAINTY
        if meaningful_count <= 1:
            uncertainty += LOW_INFORMATION_UNCERTAINTY
        if continuity < LOW_CONTINUITY_THRESHOLD:
            uncertainty += LOW_CONTINUITY_UNCERTAINTY
        uncertainty += CONTRADICTION_UNCERTAINTY_WEIGHT * contradiction
        if memory_summary.get("match_count", 0):
            uncertainty -= MEMORY_SUPPORT_UNCERTAINTY_REDUCTION * _safe_float(
                memory_summary.get("top_similarity", 0.0),
                0.0,
            )
        return clamp(uncertainty)

    def _pattern(
        self,
        pattern_type: str,
        confidence: float,
        evidence: Sequence[str],
        details: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": pattern_type,
            "confidence": round(clamp(confidence), 4),
            "evidence": [str(item) for item in evidence if str(item or "")][:PATTERN_EVIDENCE_LIMIT],
            "details": dict(details),
        }
