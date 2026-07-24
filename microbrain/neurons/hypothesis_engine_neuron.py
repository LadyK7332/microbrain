from __future__ import annotations

"""Ephemeral interpretation, prediction, and response-or-silence selection.

This neuron is deliberately not a language generator.  It receives the already
built cognitive context, asks the shared pattern toolkit what structures are
present, selectively checks memory, and publishes temporary hypotheses plus
candidate actions.  Desire/release arbitration remains responsible for whether
anything reaches speech.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.patterns.pattern_toolkit import PatternToolkit, clamp, normalize_text

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Lifetime and working-window limits for temporary hypotheses.
# TTL unit: seconds. Counts are item limits.
HYPOTHESIS_TTL_SECONDS = 120.0
NEAR_CONVERSATION_TURNS = 6
HYPOTHESIS_HISTORY_LIMIT = 16
QUERY_TERM_LIMIT = 12
ACTIVE_THREAD_LIMIT = 8
DEEP_ACTIVE_THREAD_LIMIT = 4
WORKING_EVIDENCE_LIMIT = 12
EVIDENCE_TRACE_LIMIT = 18
DIRECT_EVIDENCE_REF_LIMIT = 6
PATTERN_REF_LIMIT = 6
EVIDENCE_SUMMARY_LIMIT = 10

# Deep-memory escalation thresholds. Range: 0.0-1.0 unless noted.
DEEP_CLAIM_UNCERTAINTY_MIN = 0.32
DEEP_UNCERTAINTY_MIN = 0.58
DEEP_CONTRADICTION_MIN = 0.38
DEEP_NOVELTY_MIN = 0.82
DEEP_RISK_MIN = 0.30
DEEP_COMPLEX_INPUT_CHARS = 360
WORKING_MEMORY_SCORE_MIN = 0.42
DEEP_EPISODIC_LIMIT = 12
DEEP_SEMANTIC_LIMIT = 8
DEEP_MEM_CELL_LIMIT = 10
DEEP_EVIDENCE_LIMIT = 18

# Interpretation thresholds and output limits.
CONTINUITY_INTERPRETATION_MIN = 0.18
CONTINUITY_INTERPRETATION_BASE = 0.46
CONTINUITY_INTERPRETATION_WEIGHT = 0.48
CONTRADICTION_INTERPRETATION_MIN = 0.28
AMBIGUITY_INTERPRETATION_MIN = 0.56
MAX_INTERPRETATIONS = 4
MAX_ACTION_CANDIDATES = 6
MAX_EVIDENCE_REFS_PER_ACTION = 6
MAX_PATTERN_REFS_PER_ACTION = 6

# Action-candidate score profiles. Each row defines a base score plus weights
# for the current pattern dimensions. Changing these alters the mind's default
# action preferences, so they are declared together and remain inspectable.
ACTION_SCORE_PROFILES: Dict[tuple[str, str], Dict[str, float]] = {
    ("question", "answer"): {"base": 0.72, "expectation": 0.22},
    ("question", "clarify"): {"base": 0.35, "uncertainty": 0.55},
    ("request", "act_or_reply"): {"base": 0.70, "expectation": 0.24},
    ("request", "clarify"): {"base": 0.28, "uncertainty": 0.55},
    ("correction", "revise_context"): {"base": 0.72, "expectation": 0.18},
    ("correction", "acknowledge_revision"): {"base": 0.58, "expectation": 0.24},
    ("correction", "clarify"): {"base": 0.30, "uncertainty": 0.48},
    ("disagreement", "revise_context"): {"base": 0.72, "expectation": 0.18},
    ("disagreement", "acknowledge_revision"): {"base": 0.58, "expectation": 0.24},
    ("disagreement", "clarify"): {"base": 0.30, "uncertainty": 0.48},
    ("agreement", "continue_thread"): {"base": 0.48, "continuity": 0.30, "expectation": 0.12},
    ("agreement", "acknowledge"): {"base": 0.42, "expectation": 0.26},
    ("personal_state", "acknowledge"): {"base": 0.58, "expectation": 0.30},
    ("personal_state", "ask_followup"): {"base": 0.36, "expectation": 0.24, "continuity": 0.18},
    ("status_update", "acknowledge"): {"base": 0.52, "expectation": 0.28, "continuity": 0.12},
    ("status_update", "continue_thread"): {"base": 0.43, "continuity": 0.32, "expectation": 0.12},
    ("status_update", "ask_followup"): {"base": 0.28, "expectation": 0.22},
    ("claim", "reflect"): {"base": 0.46, "expectation": 0.24, "continuity": 0.18},
    ("claim", "investigate"): {"base": 0.35, "uncertainty": 0.38, "risk": 0.18},
    ("claim", "ask_followup"): {"base": 0.30, "uncertainty": 0.28},
    ("greeting", "acknowledge"): {"base": 0.74, "expectation": 0.18},
    ("closure", "acknowledge"): {"base": 0.26, "expectation": 0.18},
    ("minimal_statement", "acknowledge"): {"base": 0.24, "expectation": 0.28},
    ("minimal_statement", "clarify"): {"base": 0.22, "uncertainty": 0.42},
    ("statement", "continue_thread"): {"base": 0.38, "continuity": 0.34, "expectation": 0.18},
    ("statement", "reflect"): {"base": 0.34, "expectation": 0.26},
    ("statement", "clarify"): {"base": 0.18, "uncertainty": 0.42},
}

# Silence competes as a real action rather than representing missing output.
SILENCE_BASE_BY_KIND = {
    "question": 0.08,
    "request": 0.10,
    "correction": 0.16,
    "disagreement": 0.18,
    "agreement": 0.34,
    "greeting": 0.10,
    "personal_state": 0.20,
    "status_update": 0.28,
    "claim": 0.30,
    "statement": 0.40,
    "minimal_statement": 0.56,
    "closure": 0.72,
}
DEFAULT_SILENCE_BASE = 0.40
SILENCE_EXPECTATION_WEIGHT = 0.20
SILENCE_CONTINUITY_REDUCTION = 0.10
SILENCE_RISK_REDUCTION = 0.18

# Response commitment gates.
RESPONSE_SCORE_MARGIN = 0.04
RESPONSE_DEMAND_MIN = 0.44
RESPONSE_DEMAND_EXPECTATION_WEIGHT = 0.56
RESPONSE_DEMAND_ACTION_WEIGHT = 0.26
RESPONSE_DEMAND_CONTINUITY_WEIGHT = 0.10
RESPONSE_DEMAND_RISK_WEIGHT = 0.08
RESPONSE_DEMAND_SILENCE_PENALTY = 0.24
RECOMMENDED_SILENCE_PENALTY = 0.24

# Outcome-learned action bias. Range is an absolute score delta.
LEARNED_EXACT_WEIGHT = 0.82
LEARNED_GENERAL_WEIGHT = 0.18
LEARNED_BIAS_SCALE = 0.18
LEARNED_BIAS_MIN = -0.18
LEARNED_BIAS_MAX = 0.18

# DDNA modifies action preference without replacing current evidence. The
# resulting score movement is deliberately small and bounded.
DDNA_ACTION_BIAS_MIN = -0.08
DDNA_ACTION_BIAS_MAX = 0.08
DDNA_EXPRESSION_WEIGHT = 0.040
DDNA_RESTRAINT_WEIGHT = 0.030
DDNA_SOCIAL_WEIGHT = 0.030
DDNA_CONTINUITY_WEIGHT = 0.025
DDNA_INQUIRY_WEIGHT = 0.035
DDNA_CURIOSITY_WEIGHT = 0.020
DDNA_CAUTION_WEIGHT = 0.020
DDNA_ACTION_GATE_WEIGHT = 0.030
DDNA_OUTWARD_ACTION_GATE_SCALE = 0.50
DDNA_EXPRESSION_THRESHOLD_WEIGHT = 0.030

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

NEURON_NAME = Path(__file__).stem
HYPOTHESIS_SCHEMA = "hypothesis.obj.v1"
CONTEXT_BUILT_TOPIC = "context/built"
PATTERN_ANALYSIS_TOPIC = "pattern/analysis"
HYPOTHESIS_READY_TOPIC = "hypothesis/ready"
PATTERN_ANALYSIS_KIND = "pattern.analysis"
HYPOTHESIS_OBJECT_KIND = "hypothesis.object"

# Stable action-family classifications used to apply DDNA temperament. The
# weights are tunable above; these semantic memberships are structural.
DDNA_SOCIAL_ACTIONS = {
    "acknowledge",
    "acknowledge_revision",
    "continue_thread",
    "reflect",
}
DDNA_INQUIRY_ACTIONS = {
    "answer",
    "act_or_reply",
    "clarify",
    "ask_followup",
    "investigate",
    "revise_context",
}

_DEEP_RISK_TERMS = {
    "danger", "dangerous", "emergency", "hurt", "injury", "kill", "medical",
    "legal", "money", "financial", "fire", "smoke", "password", "credential",
    "unsafe", "overheat", "failure", "crash",
}


class HypothesisEngineNeuron(BaseNeuron):
    """Build a temporary hypothesis object for every meaningful context frame."""

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._patterns = PatternToolkit()

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != CONTEXT_BUILT_TOPIC:
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        text = str(input_block.get("text", "") or "").strip()
        if not text:
            return []

        correlation_id = str(event.correlation_id or input_block.get("correlation_id", "") or "")
        if correlation_id and correlation_id == str(await ctx.get_kv("hypothesis:last_correlation_id", "") or ""):
            cached = await ctx.get_kv("hypothesis:last", {})
            if isinstance(cached, Mapping) and cached:
                return [self._ready_event(event, context, dict(cached))]

        now = time.time()
        working_evidence = self._working_memory_evidence(context)
        pattern_analysis = self._patterns.analyze(context, memory_evidence=working_evidence)

        deep_reasons = self._deep_pass_reasons(pattern_analysis, text=text)
        deep_evidence: List[Dict[str, Any]] = []
        if deep_reasons:
            deep_evidence = await self._expand_memory(ctx, context, working_evidence)
            if deep_evidence:
                pattern_analysis = self._patterns.analyze(
                    context,
                    memory_evidence=[*working_evidence, *deep_evidence],
                )

        all_evidence = [*working_evidence, *deep_evidence]
        evidence_trace = self._evidence_trace(all_evidence)
        direct_evidence_refs = self._direct_evidence_refs(evidence_trace)
        pattern_refs = self._pattern_refs(pattern_analysis)

        interpretations = self._build_interpretations(pattern_analysis, context)
        action_candidates = self._build_action_candidates(
            pattern_analysis,
            interpretations,
            context,
            evidence_refs=direct_evidence_refs,
            pattern_refs=pattern_refs,
        )
        action_candidates, ddna_tuning = await self._apply_ddna_action_bias(
            ctx,
            pattern_analysis=pattern_analysis,
            candidates=action_candidates,
        )
        action_candidates, learning_bias = await self._apply_learned_action_bias(
            ctx,
            pattern_analysis=pattern_analysis,
            candidates=action_candidates,
        )
        recommended_action, recommended_score = self._recommended_action(action_candidates)
        silence_score = self._score_for(action_candidates, "silence")
        non_silence_best = max(
            (float(item.get("score", 0.0) or 0.0) for item in action_candidates if item.get("action") != "silence"),
            default=0.0,
        )

        response_demand = self._response_demand(
            pattern_analysis=pattern_analysis,
            recommended_action=recommended_action,
            recommended_score=recommended_score,
            silence_score=silence_score,
            non_silence_best=non_silence_best,
        )
        should_respond = (
            recommended_action != "silence"
            and non_silence_best >= silence_score + RESPONSE_SCORE_MARGIN
            and response_demand >= RESPONSE_DEMAND_MIN
        )

        near_turns = self._near_window(context, current_text=text, correlation_id=correlation_id)
        summary = context.get("conversation_summary", {}) if isinstance(context.get("conversation_summary", {}), Mapping) else {}
        query_terms = list(pattern_analysis.get("meaningful_tokens", []) or [])[:QUERY_TERM_LIMIT]
        active_threads = list(summary.get("active_threads", []) or [])[:ACTIVE_THREAD_LIMIT]

        hypothesis_id = self._hypothesis_id(correlation_id=correlation_id, text=text, now=now)
        hypothesis = {
            "schema_ver": HYPOTHESIS_SCHEMA,
            "hypothesis_id": hypothesis_id,
            "kind": HYPOTHESIS_OBJECT_KIND,
            "status": "candidate",
            "created_at": now,
            "expires_at": now
            + float(
                await ctx.get_kv("hypothesis:ttl_s", HYPOTHESIS_TTL_SECONDS)
                or HYPOTHESIS_TTL_SECONDS
            ),
            "correlation_id": correlation_id,
            "trigger": {
                "topic": event.topic,
                "source": str(input_block.get("source", payload.get("source", "user")) or "user"),
                "text": text,
            },
            "working_memory": {
                "near_turns": near_turns,
                "near_turn_count": len(near_turns),
                "rolling_summary": dict(summary),
                "active_threads": active_threads,
                "query_terms": query_terms,
            },
            "pattern_analysis": pattern_analysis,
            "interpretations": interpretations,
            "action_candidates": action_candidates,
            "ddna_tuning": ddna_tuning,
            "learning_bias": learning_bias,
            "recommended_action": recommended_action,
            "recommended_action_score": round(recommended_score, 4),
            "silence_score": round(silence_score, 4),
            "response_demand": round(response_demand, 4),
            "expected_usefulness": round(non_silence_best, 4),
            "should_respond": bool(should_respond),
            "memory_check": {
                "mode": "deep" if deep_reasons else "working",
                "deep_reasons": deep_reasons,
                "working_evidence_count": len(working_evidence),
                "deep_evidence_count": len(deep_evidence),
                "evidence": self._evidence_summary(all_evidence),
                "evidence_trace": evidence_trace,
                "direct_evidence_refs": direct_evidence_refs,
                "pattern_refs": pattern_refs,
            },
            "state": {
                "ephemeral": True,
                "durable_memory": False,
                "tested": False,
                "awaiting_outcome": bool(should_respond),
            },
        }

        await ctx.set_kv("pattern:last_analysis", pattern_analysis)
        await ctx.set_kv("hypothesis:last", hypothesis)
        await ctx.set_kv("hypothesis:last_correlation_id", correlation_id)
        await ctx.set_kv("hypothesis:last_tuning", ddna_tuning)
        await self._append_history(ctx, hypothesis)

        self.debug(
            "hypothesis_built",
            hypothesis_id=hypothesis_id,
            kind=pattern_analysis.get("statement_kind", "statement"),
            continuity=pattern_analysis.get("continuity", 0.0),
            uncertainty=pattern_analysis.get("uncertainty", 0.0),
            memory_mode=hypothesis["memory_check"]["mode"],
            recommended_action=recommended_action,
            response_demand=round(response_demand, 3),
            silence_score=round(silence_score, 3),
            should_respond=should_respond,
        )

        quiet_meta = {
            "kind": HYPOTHESIS_OBJECT_KIND,
            "channel": "thought",
            "store_in_memory": False,
            "reinforcement_eligible": False,
            "self_output_track": False,
            "cognitive_visible": False,
            "ephemeral": True,
        }
        return [
            Event(
                topic=PATTERN_ANALYSIS_TOPIC,
                payload=pattern_analysis,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={**quiet_meta, "kind": PATTERN_ANALYSIS_KIND},
            ),
            self._ready_event(event, context, hypothesis),
        ]

    def _ready_event(self, event: Event, context: Mapping[str, Any], hypothesis: Dict[str, Any]) -> Event:
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        return Event(
            topic=HYPOTHESIS_READY_TOPIC,
            payload={
                "context": dict(context),
                "hypothesis": hypothesis,
                "source": str(input_block.get("source", "user") or "user"),
                "channel": str(input_block.get("channel", "default") or "default"),
                "raw_meta": dict(input_block.get("raw_meta", {}) or {}),
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "hypothesis.ready",
                "stage": "hypothesized",
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "ephemeral": True,
            },
        )

    def _working_memory_evidence(self, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for item in list(context.get("associations", []) or []):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("source", "semantic_association")
            row.setdefault("evidence_kind", "semantic")
            evidence.append(row)
        for item in list(context.get("recent", []) or []):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("source", "recent_episodic")
            row.setdefault("evidence_kind", "episodic")
            evidence.append(row)
        return self._dedupe_evidence(evidence)[:WORKING_EVIDENCE_LIMIT]

    async def _expand_memory(
        self,
        ctx,
        context: Mapping[str, Any],
        working_evidence: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        text = str(input_block.get("text", "") or "").strip()
        summary = context.get("conversation_summary", {}) if isinstance(context.get("conversation_summary", {}), Mapping) else {}
        active_threads = [
            str(item)
            for item in list(summary.get("active_threads", []) or [])[:DEEP_ACTIVE_THREAD_LIMIT]
            if str(item or "")
        ]
        query = " ".join([text, *active_threads]).strip()
        if not query:
            return []

        expanded: List[Dict[str, Any]] = []
        memory = await ctx.get_kv("memory:store", None)
        if memory is not None:
            try:
                if hasattr(memory, "last_episodic"):
                    for item in list(memory.last_episodic(DEEP_EPISODIC_LIMIT) or []):
                        if isinstance(item, Mapping):
                            row = dict(item)
                            row.setdefault("source", "deep_episodic")
                            row.setdefault("evidence_kind", "episodic")
                            expanded.append(row)
            except Exception as exc:
                self.debug("deep_episodic_failed", error=repr(exc))

            working_top = max(
                (float(item.get("score", 0.0) or 0.0) for item in working_evidence if isinstance(item, Mapping)),
                default=0.0,
            )
            if working_top < WORKING_MEMORY_SCORE_MIN:
                try:
                    if hasattr(memory, "search_semantic"):
                        for item in list(memory.search_semantic(query, k=DEEP_SEMANTIC_LIMIT) or []):
                            if isinstance(item, Mapping):
                                row = dict(item)
                                row.setdefault("source", "deep_semantic")
                                row.setdefault("evidence_kind", "semantic")
                                expanded.append(row)
                except Exception as exc:
                    self.debug("deep_semantic_failed", error=repr(exc))

        mem_cells = await ctx.get_kv("memory:mem_cell_store", None)
        if mem_cells is not None and hasattr(mem_cells, "search_text_cells"):
            try:
                for item in list(mem_cells.search_text_cells(query, limit=DEEP_MEM_CELL_LIMIT) or []):
                    if isinstance(item, Mapping):
                        row = dict(item)
                        row.setdefault("source", f"mem_cell:{row.get('tier', 'unknown')}")
                        row.setdefault("evidence_kind", "mem_cell")
                        expanded.append(row)
            except Exception as exc:
                self.debug("deep_mem_cell_failed", error=repr(exc))

        working_norms = {normalize_text(self._evidence_text(item)) for item in working_evidence}
        return [
            item
            for item in self._dedupe_evidence(expanded)
            if normalize_text(self._evidence_text(item)) not in working_norms
        ][:DEEP_EVIDENCE_LIMIT]

    def _deep_pass_reasons(self, analysis: Mapping[str, Any], *, text: str) -> List[str]:
        reasons: List[str] = []
        kind = str(analysis.get("statement_kind", "statement") or "statement")
        uncertainty = float(analysis.get("uncertainty", 0.0) or 0.0)
        contradiction = float(analysis.get("contradiction", 0.0) or 0.0)
        novelty = float(analysis.get("novelty", 0.0) or 0.0)
        risk = float(analysis.get("risk", 0.0) or 0.0)
        tokens = {str(token) for token in list(analysis.get("meaningful_tokens", []) or [])}

        if kind in {"question", "correction", "disagreement"}:
            reasons.append(kind)
        elif kind == "claim" and uncertainty >= DEEP_CLAIM_UNCERTAINTY_MIN:
            reasons.append("uncertain_claim")
        if uncertainty >= DEEP_UNCERTAINTY_MIN:
            reasons.append("high_uncertainty")
        if contradiction >= DEEP_CONTRADICTION_MIN:
            reasons.append("contradiction_candidate")
        if novelty >= DEEP_NOVELTY_MIN and len(tokens) >= 3:
            reasons.append("high_novelty")
        if risk >= DEEP_RISK_MIN or tokens & _DEEP_RISK_TERMS:
            reasons.append("consequence_risk")
        if len(str(text or "")) > DEEP_COMPLEX_INPUT_CHARS:
            reasons.append("complex_input")
        return list(dict.fromkeys(reasons))

    def _build_interpretations(
        self,
        analysis: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        kind = str(analysis.get("statement_kind", "statement") or "statement")
        confidence = float(analysis.get("statement_kind_confidence", 0.5) or 0.5)
        continuity = float(analysis.get("continuity", 0.0) or 0.0)
        uncertainty = float(analysis.get("uncertainty", 0.0) or 0.0)
        interpretations: List[Tuple[str, str, float, List[str]]] = []

        primary = {
            "question": ("information_request", "Participant is asking for an answer or explanation."),
            "request": ("action_request", "Participant is asking MB to perform or prepare an action."),
            "correction": ("context_revision", "Participant is correcting or narrowing the active model."),
            "disagreement": ("belief_challenge", "Participant is challenging a prior claim or interpretation."),
            "agreement": ("shared_alignment", "Participant is affirming the current thread."),
            "greeting": ("social_opening", "Participant is opening or renewing social contact."),
            "personal_state": ("state_disclosure", "Participant is sharing a personal state that may merit acknowledgment."),
            "status_update": ("progress_report", "Participant is reporting a changed or continuing state."),
            "claim": ("proposed_model", "Participant is proposing an observation, relation, or belief."),
            "closure": ("conversation_closure", "Participant may be closing the exchange."),
            "minimal_statement": ("minimal_signal", "Participant supplied a brief signal with limited explicit intent."),
            "statement": ("thread_contribution", "Participant is adding information to the current verbal scene."),
        }.get(kind, ("thread_contribution", "Participant is adding information to the current verbal scene."))
        interpretations.append((primary[0], primary[1], confidence, [f"statement_kind:{kind}"]))

        if continuity >= CONTINUITY_INTERPRETATION_MIN:
            interpretations.append(
                (
                    "conversation_continuation",
                    "The statement likely continues an active thread rather than starting a separate topic.",
                    clamp(CONTINUITY_INTERPRETATION_BASE + (CONTINUITY_INTERPRETATION_WEIGHT * continuity)),
                    ["rolling_window_overlap"],
                )
            )
        if float(analysis.get("contradiction", 0.0) or 0.0) >= CONTRADICTION_INTERPRETATION_MIN:
            interpretations.append(
                (
                    "prior_model_may_be_wrong",
                    "A prior interpretation or claim may need revision before responding.",
                    clamp(float(analysis.get("contradiction", 0.0) or 0.0)),
                    ["contradiction_candidate"],
                )
            )
        if uncertainty >= AMBIGUITY_INTERPRETATION_MIN:
            interpretations.append(
                (
                    "intent_ambiguous",
                    "The intended response or meaning is not yet sufficiently constrained.",
                    uncertainty,
                    ["high_uncertainty"],
                )
            )

        out = [
            {
                "interpretation": label,
                "meaning": meaning,
                "confidence": round(clamp(score), 4),
                "evidence": evidence,
            }
            for label, meaning, score, evidence in interpretations
        ]
        out.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
        return out[:MAX_INTERPRETATIONS]

    def _action_score(
        self,
        kind: str,
        action: str,
        *,
        expectation: float,
        uncertainty: float,
        continuity: float,
        risk: float,
    ) -> float:
        profile = ACTION_SCORE_PROFILES.get((kind, action))
        if profile is None:
            profile = ACTION_SCORE_PROFILES.get(("statement", action), {})
        return clamp(
            float(profile.get("base", 0.0) or 0.0)
            + (float(profile.get("expectation", 0.0) or 0.0) * expectation)
            + (float(profile.get("uncertainty", 0.0) or 0.0) * uncertainty)
            + (float(profile.get("continuity", 0.0) or 0.0) * continuity)
            + (float(profile.get("risk", 0.0) or 0.0) * risk)
        )

    def _build_action_candidates(
        self,
        analysis: Mapping[str, Any],
        interpretations: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        *,
        evidence_refs: Sequence[Mapping[str, Any]] = (),
        pattern_refs: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        kind = str(analysis.get("statement_kind", "statement") or "statement")
        expectation = float(analysis.get("response_expectation", 0.0) or 0.0)
        uncertainty = float(analysis.get("uncertainty", 0.0) or 0.0)
        continuity = float(analysis.get("continuity", 0.0) or 0.0)
        risk = float(analysis.get("risk", 0.0) or 0.0)
        candidates: List[Dict[str, Any]] = []

        def add(action: str, score: float, outcome: str, reason: str, outward: bool = True) -> None:
            candidates.append(
                {
                    "action": action,
                    "score": round(clamp(score), 4),
                    "predicted_outcome": outcome,
                    "reason": reason,
                    "outward": outward,
                    "evidence_refs": [dict(item) for item in evidence_refs][:MAX_EVIDENCE_REFS_PER_ACTION],
                    "pattern_refs": [dict(item) for item in pattern_refs][:MAX_PATTERN_REFS_PER_ACTION],
                }
            )

        if kind == "question":
            add("answer", self._action_score(kind, "answer", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Resolve the requested uncertainty if evidence is adequate.", "explicit_question")
            add("clarify", self._action_score(kind, "clarify", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Reduce ambiguity before committing to an answer.", "question_uncertainty")
        elif kind == "request":
            add("act_or_reply", self._action_score(kind, "act_or_reply", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Advance the requested task or explain the available route.", "explicit_request")
            add("clarify", self._action_score(kind, "clarify", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Constrain the requested action before execution.", "request_uncertainty")
        elif kind in {"correction", "disagreement"}:
            add("revise_context", self._action_score(kind, "revise_context", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Update the active model and avoid repeating the rejected interpretation.", "correction_detected", outward=False)
            add("acknowledge_revision", self._action_score(kind, "acknowledge_revision", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Show that the correction changed the active context.", "social_repair")
            add("clarify", self._action_score(kind, "clarify", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Resolve the exact boundary of the correction.", "remaining_ambiguity")
        elif kind == "agreement":
            add("continue_thread", self._action_score(kind, "continue_thread", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Continue the active line without restarting it.", "shared_alignment")
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Mark alignment without over-expanding the exchange.", "agreement")
        elif kind == "personal_state":
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Recognize the disclosed state and preserve social continuity.", "state_disclosure")
            add("ask_followup", self._action_score(kind, "ask_followup", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Invite more detail if the participant appears to be opening a thread.", "possible_social_opening")
        elif kind == "status_update":
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Confirm that the changed state was incorporated.", "progress_report")
            add("continue_thread", self._action_score(kind, "continue_thread", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Relate the update to the active shared task.", "thread_progress")
            add("ask_followup", self._action_score(kind, "ask_followup", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Request the next useful observation only if it would move the task.", "optional_probe")
        elif kind == "claim":
            add("reflect", self._action_score(kind, "reflect", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Test or extend the proposed model within the current context.", "proposed_model")
            add("investigate", self._action_score(kind, "investigate", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Check memory or evidence before accepting the claim.", "claim_evaluation", outward=False)
            add("ask_followup", self._action_score(kind, "ask_followup", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Seek a discriminating detail if the model remains ambiguous.", "claim_uncertainty")
        elif kind == "greeting":
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Return the social opening and establish contact.", "greeting")
        elif kind == "closure":
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Close cleanly if a reply is socially useful.", "closure")
        elif kind == "minimal_statement":
            add("acknowledge", self._action_score(kind, "acknowledge", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Mark receipt only if the brief signal expects it.", "minimal_signal")
            add("clarify", self._action_score(kind, "clarify", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Ask only when the missing meaning blocks continuity.", "minimal_ambiguity")
        else:
            add("continue_thread", self._action_score("statement", "continue_thread", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Connect the statement to the rolling conversation.", "thread_contribution")
            add("reflect", self._action_score("statement", "reflect", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Offer a useful interpretation if one is supported.", "meaningful_statement")
            add("clarify", self._action_score("statement", "clarify", expectation=expectation, uncertainty=uncertainty, continuity=continuity, risk=risk), "Ask for clarification only when ambiguity is material.", "statement_uncertainty")

        silence = SILENCE_BASE_BY_KIND.get(kind, DEFAULT_SILENCE_BASE)
        silence += SILENCE_EXPECTATION_WEIGHT * (1.0 - expectation)
        silence -= SILENCE_CONTINUITY_REDUCTION * continuity
        if risk > 0.0:
            silence -= SILENCE_RISK_REDUCTION * risk
        add("silence", silence, "Avoid unnecessary interruption or unsupported speech.", "silence_is_valid")

        # Internal actions may be useful, but an outward candidate must win before
        # release. Sorting still keeps the complete decision field inspectable.
        candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return candidates[:MAX_ACTION_CANDIDATES]

    async def _apply_ddna_action_bias(
        self,
        ctx,
        *,
        pattern_analysis: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply bounded temperament pressure to candidate actions.

        DDNA changes the nature of the mind, not the evidence. Pattern analysis
        and memory scores are left intact; only candidate preference receives a
        small, inspectable adjustment.
        """
        raw = await ctx.get_kv("drive:ddna_modulators", {}) or {}
        ddna = raw if isinstance(raw, Mapping) else {}
        expression = float(ddna.get("expression_bias", 1.0) or 1.0)
        restraint = float(ddna.get("restraint_bias", 1.0) or 1.0)
        social = float(ddna.get("social_gain", 1.0) or 1.0)
        continuity_gain = float(ddna.get("continuity_gain", 1.0) or 1.0)
        inquiry = float(ddna.get("inquiry_gain", 1.0) or 1.0)
        curiosity = float(ddna.get("curiosity_gain", 1.0) or 1.0)
        caution = float(ddna.get("caution_gain", 1.0) or 1.0)
        action_gate = float(ddna.get("action_gate_strictness", 1.0) or 1.0)
        expression_threshold = float(ddna.get("expression_threshold_gain", 1.0) or 1.0)

        adjusted: List[Dict[str, Any]] = []
        applied: List[Dict[str, Any]] = []
        for candidate in candidates:
            row = dict(candidate)
            action = str(row.get("action", "") or "")
            base_score = float(row.get("score", 0.0) or 0.0)
            bias = 0.0

            if action == "silence":
                bias += (restraint - 1.0) * DDNA_RESTRAINT_WEIGHT
                bias += (action_gate - 1.0) * DDNA_ACTION_GATE_WEIGHT
                bias += (expression_threshold - 1.0) * DDNA_EXPRESSION_THRESHOLD_WEIGHT
                bias -= (expression - 1.0) * DDNA_EXPRESSION_WEIGHT
            else:
                bias += (expression - 1.0) * DDNA_EXPRESSION_WEIGHT
                bias -= (restraint - 1.0) * DDNA_RESTRAINT_WEIGHT
                bias -= (action_gate - 1.0) * (DDNA_ACTION_GATE_WEIGHT * DDNA_OUTWARD_ACTION_GATE_SCALE)

                if action in DDNA_SOCIAL_ACTIONS:
                    bias += (social - 1.0) * DDNA_SOCIAL_WEIGHT
                    bias += (continuity_gain - 1.0) * DDNA_CONTINUITY_WEIGHT
                if action in DDNA_INQUIRY_ACTIONS:
                    bias += (inquiry - 1.0) * DDNA_INQUIRY_WEIGHT
                    bias += (curiosity - 1.0) * DDNA_CURIOSITY_WEIGHT
                if action in {"clarify", "investigate", "revise_context"}:
                    bias += (caution - 1.0) * DDNA_CAUTION_WEIGHT

            bias = max(DDNA_ACTION_BIAS_MIN, min(DDNA_ACTION_BIAS_MAX, bias))
            row["pre_ddna_score"] = round(base_score, 4)
            row["ddna_bias"] = round(bias, 4)
            row["score"] = round(clamp(base_score + bias), 4)
            adjusted.append(row)
            if abs(bias) > 1e-9:
                applied.append({"action": action, "bias": round(bias, 4)})

        adjusted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        trace = {
            "scope": "hypothesis_action_candidates",
            "statement_kind": str(pattern_analysis.get("statement_kind", "statement") or "statement"),
            "bias_min": DDNA_ACTION_BIAS_MIN,
            "bias_max": DDNA_ACTION_BIAS_MAX,
            "inputs": {
                "expression_bias": round(expression, 6),
                "restraint_bias": round(restraint, 6),
                "social_gain": round(social, 6),
                "continuity_gain": round(continuity_gain, 6),
                "inquiry_gain": round(inquiry, 6),
                "curiosity_gain": round(curiosity, 6),
                "caution_gain": round(caution, 6),
                "action_gate_strictness": round(action_gate, 6),
                "expression_threshold_gain": round(expression_threshold, 6),
            },
            "applied": applied,
        }
        return adjusted[:MAX_ACTION_CANDIDATES], trace


    async def _apply_learned_action_bias(
        self,
        ctx,
        *,
        pattern_analysis: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply bounded outcome-learned bias without replacing current evidence.

        Sparse experience must not dominate a fresh scene.  Exact statement-kind
        buckets count more than global action buckets, confidence grows slowly,
        and the total score movement is capped at +/-0.18.
        """

        raw = await ctx.get_kv("hypothesis:action_learning", {})
        learning = dict(raw) if isinstance(raw, Mapping) else {}
        kind = str(pattern_analysis.get("statement_kind", "statement") or "statement")
        applied: List[Dict[str, Any]] = []
        adjusted: List[Dict[str, Any]] = []

        def signal(bucket: Mapping[str, Any]) -> Tuple[float, float, int]:
            avg = max(-1.0, min(1.0, float(bucket.get("avg_score", 0.0) or 0.0)))
            confidence = max(0.0, min(1.0, float(bucket.get("confidence", 0.0) or 0.0)))
            observations = int(bucket.get("observations", 0) or 0)
            return avg * confidence, confidence, observations

        for candidate in candidates:
            row = dict(candidate)
            action = str(row.get("action", "") or "")
            base_score = float(row.get("score", 0.0) or 0.0)
            exact = learning.get(f"{kind}|{action}", {})
            general = learning.get(f"*|{action}", {})
            exact = exact if isinstance(exact, Mapping) else {}
            general = general if isinstance(general, Mapping) else {}
            exact_signal, exact_conf, exact_n = signal(exact)
            general_signal, general_conf, general_n = signal(general)
            combined = (LEARNED_EXACT_WEIGHT * exact_signal) + (LEARNED_GENERAL_WEIGHT * general_signal)
            learned_bias = max(
                LEARNED_BIAS_MIN,
                min(LEARNED_BIAS_MAX, combined * LEARNED_BIAS_SCALE),
            )
            row["base_score"] = round(base_score, 4)
            row["learned_bias"] = round(learned_bias, 4)
            row["score"] = round(clamp(base_score + learned_bias), 4)
            if exact_n or general_n:
                row["learning_evidence"] = {
                    "exact_observations": exact_n,
                    "exact_confidence": round(exact_conf, 4),
                    "general_observations": general_n,
                    "general_confidence": round(general_conf, 4),
                }
                applied.append({
                    "action": action,
                    "bias": round(learned_bias, 4),
                    "exact_observations": exact_n,
                    "general_observations": general_n,
                })
            adjusted.append(row)

        adjusted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return adjusted[:MAX_ACTION_CANDIDATES], {
            "statement_kind": kind,
            "applied": applied,
            "max_abs_bias": round(max((abs(float(item.get("bias", 0.0) or 0.0)) for item in applied), default=0.0), 4),
        }

    def _recommended_action(self, candidates: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
        outward = [item for item in candidates if bool(item.get("outward", True))]
        if not outward:
            return "silence", 1.0
        best = max(outward, key=lambda item: float(item.get("score", 0.0) or 0.0))
        return str(best.get("action", "silence") or "silence"), float(best.get("score", 0.0) or 0.0)

    def _score_for(self, candidates: Sequence[Mapping[str, Any]], action: str) -> float:
        for item in candidates:
            if str(item.get("action", "") or "") == action:
                return float(item.get("score", 0.0) or 0.0)
        return 0.0

    def _response_demand(
        self,
        *,
        pattern_analysis: Mapping[str, Any],
        recommended_action: str,
        recommended_score: float,
        silence_score: float,
        non_silence_best: float,
    ) -> float:
        expectation = float(pattern_analysis.get("response_expectation", 0.0) or 0.0)
        risk = float(pattern_analysis.get("risk", 0.0) or 0.0)
        continuity = float(pattern_analysis.get("continuity", 0.0) or 0.0)
        demand = (
            (RESPONSE_DEMAND_EXPECTATION_WEIGHT * expectation)
            + (RESPONSE_DEMAND_ACTION_WEIGHT * non_silence_best)
            + (RESPONSE_DEMAND_CONTINUITY_WEIGHT * continuity)
            + (RESPONSE_DEMAND_RISK_WEIGHT * risk)
            - (RESPONSE_DEMAND_SILENCE_PENALTY * silence_score)
        )
        if recommended_action == "silence":
            demand -= RECOMMENDED_SILENCE_PENALTY
        return clamp(demand)

    def _near_window(self, context: Mapping[str, Any], *, current_text: str, correlation_id: str) -> List[Dict[str, Any]]:
        scene = context.get("conversation_scene", {}) if isinstance(context.get("conversation_scene", {}), Mapping) else {}
        turns = [dict(turn) for turn in list(scene.get("turns", []) or []) if isinstance(turn, Mapping)]
        current_norm = normalize_text(current_text)
        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            same_corr = bool(correlation_id) and str(turn.get("correlation_id", "") or "") == correlation_id
            same_text = current_norm and normalize_text(str(turn.get("text", "") or "")) == current_norm
            if str(turn.get("role", "")) == "user" and (same_corr or same_text):
                del turns[index]
                break
        return [
            {
                "ts": float(turn.get("ts", 0.0) or 0.0),
                "role": str(turn.get("role", "") or ""),
                "text": str(turn.get("text", "") or "")[:260],
                "correlation_id": str(turn.get("correlation_id", "") or ""),
            }
            for turn in turns[-NEAR_CONVERSATION_TURNS:]
        ]

    def _evidence_trace(self, evidence: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Preserve exact query participants for later credit assignment."""
        traced: List[Dict[str, Any]] = []
        for item in self._dedupe_evidence(evidence)[:EVIDENCE_TRACE_LIMIT]:
            text = self._evidence_text(item)
            cell_id = str(item.get("cell_id", item.get("id", "")) or "").strip()
            source = str(item.get("source", "") or "unknown")
            tier = str(item.get("tier", "") or "")
            evidence_kind = str(item.get("evidence_kind", item.get("kind", "")) or "memory")
            if not text and not cell_id:
                continue
            traced.append(
                {
                    "cell_id": cell_id,
                    "tier": tier,
                    "kind": evidence_kind,
                    "source": source,
                    "text": text[:220],
                    "retrieval_score": round(float(item.get("score", 0.0) or 0.0), 6),
                    "links_explicit": [
                        str(link or "")
                        for link in list(item.get("links_explicit", []) or [])
                        if str(link or "")
                    ][:12],
                }
            )
        return traced

    def _direct_evidence_refs(self, evidence_trace: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for item in evidence_trace:
            cell_id = str(item.get("cell_id", "") or "").strip()
            if not cell_id:
                continue
            refs.append(
                {
                    "cell_id": cell_id,
                    "tier": str(item.get("tier", "") or ""),
                    "score": round(max(0.0, float(item.get("retrieval_score", 0.0) or 0.0)), 6),
                    "role": "hypothesis_evidence",
                }
            )
        refs.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return refs[:DIRECT_EVIDENCE_REF_LIMIT]

    def _pattern_refs(self, analysis: Mapping[str, Any]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for item in list(analysis.get("patterns", []) or []):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("pattern", item.get("type", "")) or "").strip()
            if not name:
                continue
            refs.append(
                {
                    "pattern": name,
                    "confidence": round(float(item.get("confidence", item.get("score", 0.0)) or 0.0), 6),
                }
            )
        refs.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
        return refs[:PATTERN_REF_LIMIT]

    def _evidence_summary(self, evidence: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in self._dedupe_evidence(evidence)[:EVIDENCE_SUMMARY_LIMIT]:
            text = self._evidence_text(item)
            if not text:
                continue
            out.append(
                {
                    "text": text[:220],
                    "source": str(item.get("source", "") or "unknown"),
                    "kind": str(item.get("evidence_kind", item.get("kind", "")) or "memory"),
                    "score": round(float(item.get("score", 0.0) or 0.0), 4),
                    "cell_id": str(item.get("cell_id", item.get("id", "")) or ""),
                }
            )
        return out

    def _evidence_text(self, item: Mapping[str, Any]) -> str:
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

    def _dedupe_evidence(self, evidence: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            text = self._evidence_text(row)
            norm = normalize_text(text)
            if not norm:
                continue
            score = float(row.get("score", 0.0) or 0.0)
            if norm not in best:
                best[norm] = row
                order.append(norm)
            elif score > float(best[norm].get("score", 0.0) or 0.0):
                best[norm] = row
        return [best[norm] for norm in order]

    async def _append_history(self, ctx, hypothesis: Mapping[str, Any]) -> None:
        history = await ctx.get_kv("hypothesis:history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "hypothesis_id": str(hypothesis.get("hypothesis_id", "") or ""),
                "created_at": float(hypothesis.get("created_at", 0.0) or 0.0),
                "correlation_id": str(hypothesis.get("correlation_id", "") or ""),
                "recommended_action": str(hypothesis.get("recommended_action", "") or ""),
                "response_demand": float(hypothesis.get("response_demand", 0.0) or 0.0),
                "should_respond": bool(hypothesis.get("should_respond", False)),
                "statement_kind": str(
                    (hypothesis.get("pattern_analysis", {}) if isinstance(hypothesis.get("pattern_analysis", {}), Mapping) else {}).get("statement_kind", "")
                    or ""
                ),
            }
        )
        await ctx.set_kv("hypothesis:history", history[-HYPOTHESIS_HISTORY_LIMIT:])

    def _hypothesis_id(self, *, correlation_id: str, text: str, now: float) -> str:
        seed = f"{correlation_id}|{normalize_text(text)}|{now:.6f}".encode("utf-8", errors="ignore")
        return f"hyp:{hashlib.blake2s(seed, digest_size=10).hexdigest()}"


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[CONTEXT_BUILT_TOPIC],
        output_topics=[PATTERN_ANALYSIS_TOPIC, HYPOTHESIS_READY_TOPIC],
        priority=14,
        cooldown_sec=0.0,
    )
    yield HypothesisEngineNeuron(cfg)
