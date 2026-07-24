from __future__ import annotations

"""Observe whether a committed hypothesis action produced the expected result.

The hypothesis engine proposes meaning and likely actions.  This neuron closes
that loop.  It records the chosen action, confirms whether it was actually
executed, inspects the next external observation or explicit trainer feedback,
and updates small action-learning buckets that the hypothesis engine can reuse.

No hypothesis is promoted to truth here.  Missing feedback is explicitly kept
as unobserved rather than being treated as success or failure.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.patterns.pattern_toolkit import jaccard, normalize_text, tokenize

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Time allowed for action execution and later outcome observation.
# Unit: seconds.
OBSERVATION_TTL_SECONDS = 15 * 60.0
EXECUTION_TTL_SECONDS = 60.0
OUTCOME_HISTORY_LIMIT = 48

# Direct execution failures.
ACTION_NOT_EXECUTED_SCORE = -0.72
ACTION_NOT_EXECUTED_RELIABILITY = 0.96
ACTION_TIMEOUT_RELIABILITY = 0.94

# Explicit reinforcement conversion.
REINFORCEMENT_WEIGHT_DIVISOR = 5.0
EXPLICIT_FEEDBACK_RELIABILITY = 1.0
TRAINER_CORRECTION_SCORE = -1.0

# Interaction relief is useful but weaker evidence than participant feedback.
RELIEF_SCORE_BASE = 0.20
RELIEF_SCORE_CAP = 0.55
RELIEF_RELIABILITY = 0.58

# Textual outcome scoring and diagnostic-overlap thresholds.
ACCENT_SCORE_DIVISOR = 10.0
ACCENT_RELIABILITY = 0.98
SILENCE_REJECTED_SCORE = -0.95
SILENCE_REJECTED_RELIABILITY = 0.98
NEGATIVE_FEEDBACK_SCORE = -0.88
NEGATIVE_FEEDBACK_RELIABILITY = 0.94
CORRECTION_OVERLAP_MIN = 0.08
CORRECTION_SCORE = -0.68
CORRECTION_RELIABILITY = 0.82
REPEATED_QUESTION_OVERLAP_MIN = 0.42
REPEATED_QUESTION_SCORE = -0.56
REPEATED_QUESTION_RELIABILITY = 0.82
POSITIVE_FEEDBACK_OVERLAP_MIN = 0.05
POSITIVE_FEEDBACK_SCORE_RELEVANT = 0.78
POSITIVE_FEEDBACK_SCORE_GENERIC = 0.68
POSITIVE_FEEDBACK_RELIABILITY = 0.90
CLOSURE_SCORE = 0.62
CLOSURE_RELIABILITY = 0.82
SILENCE_TOPIC_ADVANCE_MAX_OVERLAP = 0.10
SILENCE_TOLERATED_SCORE = 0.20
SILENCE_TOLERATED_RELIABILITY = 0.38
SILENCE_AMBIGUOUS_RELIABILITY = 0.20
THREAD_CONTINUATION_OVERLAP_MIN = 0.12
FOLLOWUP_SCORE = 0.38
FOLLOWUP_RELIABILITY = 0.58
THREAD_CONTINUATION_SCORE = 0.32
THREAD_CONTINUATION_RELIABILITY = 0.52
UNDIAGNOSTIC_RELIABILITY = 0.22

# Outcome status and learning thresholds.
POSITIVE_OUTCOME_THRESHOLD = 0.24
NEGATIVE_OUTCOME_THRESHOLD = -0.24
GENERAL_ACTION_BUCKET_SHARE = 0.35
LEARNING_CONFIDENCE_OBSERVATIONS = 6.0
REINFORCEMENT_TARGET_SIMILARITY = 0.60
TRAINER_TARGET_SIMILARITY = 0.55

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

NEURON_NAME = Path(__file__).stem
OUTCOME_SCHEMA = "hypothesis.outcome.v1"
ACTION_COMMITTED_TOPIC = "hypothesis/action_committed"
SPEECH_TOPIC = "act/speech"
TEXT_TOPIC = "percept/text"
REINFORCE_TOPIC = "control/reinforce"
TRAINER_CORRECTION_TOPIC = "control/trainer_correction"
INTERACTION_RELIEF_TOPIC = "event/relief/interaction"
CLOCK_TOPIC = "clock/tick"
OUTCOME_TOPIC = "hypothesis/outcome"

_POSITIVE_MARKERS = (
    "yes", "yeah", "yep", "exactly", "right", "correct", "true", "agreed",
    "that makes sense", "makes sense", "good", "great", "perfect", "nice",
    "thanks", "thank you", "ty", "helpful", "works", "worked",
)
_NEGATIVE_MARKERS = (
    "wrong", "no that's not", "no that is not", "not what i meant", "not quite",
    "that doesn't", "that does not", "doesn't answer", "does not answer",
    "you misunderstood", "missed the point", "incorrect", "actually no",
)
_CORRECTION_MARKERS = (
    "actually", "instead", "rather", "to clarify", "i mean", "correction",
)
_CLOSURE_MARKERS = (
    "thanks", "thank you", "ty", "that's all", "thats all", "done for now",
    "goodbye", "bye", "talk later", "see you later",
)
_SILENCE_COMPLAINTS = (
    "hello", "hello?", "are you there", "why no response", "why didn't you respond",
    "why did you not respond", "answer me", "you didn't answer", "you did not answer",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _extract_text(event: Event) -> str:
    payload = event.payload
    if isinstance(payload, Mapping):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return ""


def _raw_meta(event: Event) -> Dict[str, Any]:
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    raw = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), Mapping) else {}
    return {**dict(raw), **dict(event.meta or {})}


def _contains(normalized: str, phrases: Sequence[str]) -> bool:
    padded = f" {normalized} "
    for phrase in phrases:
        needle = normalize_text(phrase)
        if needle and (normalized == needle or f" {needle} " in padded):
            return True
    return False


class HypothesisOutcomeObserverNeuron(BaseNeuron):
    """Close the prediction -> action -> observation -> learning loop."""

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic == ACTION_COMMITTED_TOPIC:
            await self._commit(ctx, event)
            return []

        if event.topic == SPEECH_TOPIC:
            await self._mark_execution(ctx, event)
            return []

        if event.topic == TEXT_TOPIC:
            if not self._is_external_text(event):
                return []
            pending = await self._pending(ctx)
            if not pending or self._same_trigger(pending, event):
                return []
            if not bool(pending.get("action_executed", False)):
                return await self._finalize(
                    ctx,
                    pending,
                    event=event,
                    score=ACTION_NOT_EXECUTED_SCORE,
                    reliability=ACTION_NOT_EXECUTED_RELIABILITY,
                    reason="selected_action_not_executed",
                    evidence=["next_participant_turn_arrived_before_action"],
                )
            score, reliability, reason, evidence = self._score_external_text(pending, event)
            return await self._finalize(
                ctx,
                pending,
                event=event,
                score=score,
                reliability=reliability,
                reason=reason,
                evidence=evidence,
            )

        if event.topic == REINFORCE_TOPIC:
            pending = await self._pending(ctx)
            if not pending or not self._reinforcement_matches(pending, event):
                return []
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            weight = _safe_float(payload.get("weight", payload.get("score", 0.0)), 0.0)
            if weight == 0.0:
                return []
            return await self._finalize(
                ctx,
                pending,
                event=event,
                score=_clamp(weight / REINFORCEMENT_WEIGHT_DIVISOR),
                reliability=EXPLICIT_FEEDBACK_RELIABILITY,
                reason="explicit_reinforcement",
                evidence=[f"reinforcement_weight:{weight:+g}"],
            )

        if event.topic == TRAINER_CORRECTION_TOPIC:
            pending = await self._pending(ctx)
            if not pending or not self._trainer_matches(pending, event):
                return []
            return await self._finalize(
                ctx,
                pending,
                event=event,
                score=TRAINER_CORRECTION_SCORE,
                reliability=EXPLICIT_FEEDBACK_RELIABILITY,
                reason="trainer_correction",
                evidence=["trainer_supplied_alternative"],
            )

        if event.topic == INTERACTION_RELIEF_TOPIC:
            pending = await self._pending(ctx)
            if not pending or not self._correlation_matches(pending, event):
                return []
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            relief = max(0.0, _safe_float(payload.get("delta_pct", 0.0), 0.0))
            if relief <= 0.0:
                return []
            # Interaction relief proves that expression vented a need, but it is
            # weaker evidence about semantic correctness than participant feedback.
            return await self._finalize(
                ctx,
                pending,
                event=event,
                score=min(RELIEF_SCORE_CAP, RELIEF_SCORE_BASE + relief),
                reliability=RELIEF_RELIABILITY,
                reason="interaction_relief_observed",
                evidence=[f"relief_delta:{relief:.4f}"],
            )

        if event.topic == CLOCK_TOPIC:
            pending = await self._pending(ctx)
            if not pending:
                return []
            now = time.time()
            committed_at = _safe_float(pending.get("committed_at", 0.0), 0.0)
            action = str(pending.get("selected_action", "") or "")
            execution_ttl = _safe_float(
                await ctx.get_kv("hypothesis:execution_ttl_s", EXECUTION_TTL_SECONDS),
                EXECUTION_TTL_SECONDS,
            )
            observation_ttl = _safe_float(
                await ctx.get_kv("hypothesis:observation_ttl_s", OBSERVATION_TTL_SECONDS),
                OBSERVATION_TTL_SECONDS,
            )
            age = max(0.0, now - committed_at) if committed_at else 0.0

            if action != "silence" and not bool(pending.get("action_executed", False)) and age >= execution_ttl:
                return await self._finalize(
                    ctx,
                    pending,
                    event=event,
                    score=ACTION_NOT_EXECUTED_SCORE,
                    reliability=ACTION_TIMEOUT_RELIABILITY,
                    reason="action_execution_timeout",
                    evidence=[f"execution_timeout_s:{execution_ttl:.1f}"],
                )

            if age >= observation_ttl:
                return await self._finalize(
                    ctx,
                    pending,
                    event=event,
                    score=0.0,
                    reliability=0.0,
                    reason="expired_without_observation",
                    evidence=[f"observation_timeout_s:{observation_ttl:.1f}"],
                    learn=False,
                )

        return []

    async def _commit(self, ctx, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
        trigger = payload.get("trigger", {}) if isinstance(payload.get("trigger", {}), Mapping) else {}
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        pattern = hypothesis.get("pattern_analysis", {}) if isinstance(hypothesis.get("pattern_analysis", {}), Mapping) else {}
        action = str(trigger.get("recommended_action", hypothesis.get("recommended_action", "silence")) or "silence")
        candidate = self._candidate_for(hypothesis, action)
        now = time.time()
        pending = {
            "schema_ver": "hypothesis.pending_outcome.v1",
            "hypothesis_id": str(hypothesis.get("hypothesis_id", "") or ""),
            "correlation_id": str(event.correlation_id or hypothesis.get("correlation_id", "") or ""),
            "selected_action": action,
            "statement_kind": str(pattern.get("statement_kind", "statement") or "statement"),
            "trigger_text": str(input_block.get("text", "") or "")[:700],
            "trigger_tokens": list(pattern.get("meaningful_tokens", []) or [])[:24],
            "predicted_outcome": str(candidate.get("predicted_outcome", "") or ""),
            "predicted_utility": _safe_float(candidate.get("score", hypothesis.get("recommended_action_score", 0.5)), 0.5),
            "candidate_reason": str(candidate.get("reason", "") or ""),
            "evidence_refs": [dict(item) for item in list(candidate.get("evidence_refs", []) or []) if isinstance(item, Mapping)][:8],
            "pattern_refs": [dict(item) for item in list(candidate.get("pattern_refs", []) or []) if isinstance(item, Mapping)][:8],
            "memory_evidence_trace": [
                dict(item)
                for item in list(
                    (hypothesis.get("memory_check", {}) if isinstance(hypothesis.get("memory_check", {}), Mapping) else {}).get("evidence_trace", [])
                    or []
                )
                if isinstance(item, Mapping)
            ][:18],
            "response_demand": _safe_float(hypothesis.get("response_demand", 0.0), 0.0),
            "should_release": bool(trigger.get("should_release", False)),
            "deliberate_silence": bool(trigger.get("deliberate_silence", action == "silence")),
            "committed_at": now,
            "action_executed": action == "silence",
            "acted_at": now if action == "silence" else 0.0,
            "output_text": "",
            "output_source": "",
            "output_cell_ids": [],
            "awaiting_observation": True,
        }
        await ctx.set_kv("hypothesis:pending_outcome", pending)
        await ctx.set_kv("hypothesis:pending_action", pending)
        await ctx.set_kv("hypothesis:last_committed_action", dict(pending))

    async def _mark_execution(self, ctx, event: Event) -> None:
        if bool((event.meta or {}).get("control", False)):
            return
        text = _extract_text(event)
        if not text:
            return
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        channel = str(payload.get("channel", "default") or "default")
        if channel in {"thought", "internal"}:
            return
        pending = await self._pending(ctx)
        if not pending or not self._correlation_matches(pending, event):
            return
        pending["action_executed"] = True
        pending["acted_at"] = time.time()
        pending["output_text"] = text[:900]
        pending["output_source"] = str(event.source or "")
        pending["output_kind"] = str((event.meta or {}).get("kind", "") or "")
        raw_ids = payload.get("memory_cell_ids", (event.meta or {}).get("memory_cell_ids", []))
        pending["output_cell_ids"] = [str(cell_id or "") for cell_id in list(raw_ids or []) if str(cell_id or "")][:12]
        await ctx.set_kv("hypothesis:pending_outcome", pending)
        await ctx.set_kv("hypothesis:pending_action", pending)

    def _score_external_text(self, pending: Mapping[str, Any], event: Event) -> Tuple[float, float, str, list[str]]:
        text = _extract_text(event)
        normalized = normalize_text(text)
        raw = _raw_meta(event)
        accent = _safe_float(raw.get("accent_value", 0.0), 0.0)
        if accent != 0.0:
            return (
                _clamp(accent / ACCENT_SCORE_DIVISOR),
                ACCENT_RELIABILITY,
                "explicit_textual_accent",
                [f"accent:{accent:+g}"],
            )

        action = str(pending.get("selected_action", "") or "")
        statement_kind = str(pending.get("statement_kind", "statement") or "statement")
        current_tokens = tokenize(text, meaningful=True)
        trigger_tokens = [str(token) for token in list(pending.get("trigger_tokens", []) or [])]
        if not trigger_tokens:
            trigger_tokens = tokenize(str(pending.get("trigger_text", "") or ""), meaningful=True)
        output_tokens = tokenize(str(pending.get("output_text", "") or ""), meaningful=True)
        trigger_overlap = jaccard(current_tokens, trigger_tokens)
        output_overlap = jaccard(current_tokens, output_tokens)
        topical_overlap = max(trigger_overlap, output_overlap)

        if action == "silence" and _contains(normalized, _SILENCE_COMPLAINTS):
            return SILENCE_REJECTED_SCORE, SILENCE_REJECTED_RELIABILITY, "silence_rejected", ["participant_requested_missing_response"]

        if _contains(normalized, _NEGATIVE_MARKERS):
            return NEGATIVE_FEEDBACK_SCORE, NEGATIVE_FEEDBACK_RELIABILITY, "participant_rejected_outcome", ["negative_feedback_marker"]

        if _contains(normalized, _CORRECTION_MARKERS) and topical_overlap >= CORRECTION_OVERLAP_MIN:
            return CORRECTION_SCORE, CORRECTION_RELIABILITY, "participant_corrected_interpretation", [f"topic_overlap:{topical_overlap:.3f}"]

        if text.rstrip().endswith("?") and trigger_overlap >= REPEATED_QUESTION_OVERLAP_MIN:
            return REPEATED_QUESTION_SCORE, REPEATED_QUESTION_RELIABILITY, "question_repeated_after_response", [f"trigger_overlap:{trigger_overlap:.3f}"]

        if _contains(normalized, _POSITIVE_MARKERS):
            score = POSITIVE_FEEDBACK_SCORE_RELEVANT if topical_overlap >= POSITIVE_FEEDBACK_OVERLAP_MIN else POSITIVE_FEEDBACK_SCORE_GENERIC
            return score, POSITIVE_FEEDBACK_RELIABILITY, "participant_affirmed_outcome", [f"topic_overlap:{topical_overlap:.3f}"]

        if _contains(normalized, _CLOSURE_MARKERS):
            return CLOSURE_SCORE, CLOSURE_RELIABILITY, "conversation_closed_cleanly", ["closure_after_action"]

        if action == "silence":
            if statement_kind == "closure" and topical_overlap < SILENCE_TOPIC_ADVANCE_MAX_OVERLAP:
                return SILENCE_TOLERATED_SCORE, SILENCE_TOLERATED_RELIABILITY, "silence_tolerated_then_topic_advanced", [f"topic_overlap:{topical_overlap:.3f}"]
            return 0.0, SILENCE_AMBIGUOUS_RELIABILITY, "silence_outcome_ambiguous", [f"topic_overlap:{topical_overlap:.3f}"]

        if topical_overlap >= THREAD_CONTINUATION_OVERLAP_MIN:
            if text.rstrip().endswith("?"):
                return FOLLOWUP_SCORE, FOLLOWUP_RELIABILITY, "useful_followup_on_same_thread", [f"topic_overlap:{topical_overlap:.3f}"]
            return THREAD_CONTINUATION_SCORE, THREAD_CONTINUATION_RELIABILITY, "thread_continued_after_action", [f"topic_overlap:{topical_overlap:.3f}"]

        return 0.0, UNDIAGNOSTIC_RELIABILITY, "next_turn_not_diagnostic", [f"topic_overlap:{topical_overlap:.3f}"]

    async def _finalize(
        self,
        ctx,
        pending: Mapping[str, Any],
        *,
        event: Event,
        score: float,
        reliability: float,
        reason: str,
        evidence: Sequence[str],
        learn: bool = True,
    ) -> list[Event]:
        now = time.time()
        score = _clamp(score)
        reliability = max(0.0, min(1.0, float(reliability)))
        predicted_utility = max(0.0, min(1.0, _safe_float(pending.get("predicted_utility", 0.5), 0.5)))
        observed_utility = (score + 1.0) / 2.0
        prediction_error = observed_utility - predicted_utility
        status = (
            "positive"
            if score >= POSITIVE_OUTCOME_THRESHOLD
            else ("negative" if score <= NEGATIVE_OUTCOME_THRESHOLD else "neutral")
        )
        outcome_id = self._outcome_id(pending, now, reason)
        observed_text = _extract_text(event)

        outcome = {
            "schema_ver": OUTCOME_SCHEMA,
            "outcome_id": outcome_id,
            "hypothesis_id": str(pending.get("hypothesis_id", "") or ""),
            "correlation_id": str(pending.get("correlation_id", "") or ""),
            "selected_action": str(pending.get("selected_action", "") or ""),
            "statement_kind": str(pending.get("statement_kind", "statement") or "statement"),
            "predicted_outcome": str(pending.get("predicted_outcome", "") or ""),
            "predicted_utility": round(predicted_utility, 4),
            "observed_topic": str(event.topic or ""),
            "observed_text": observed_text[:500],
            "score": round(score, 4),
            "reliability": round(reliability, 4),
            "status": status,
            "reason": reason,
            "evidence": [str(item) for item in evidence][:8],
            "prediction_error": round(prediction_error, 4),
            "prediction_supported": bool(score >= POSITIVE_OUTCOME_THRESHOLD),
            "action_executed": bool(pending.get("action_executed", False)),
            "output_text": str(pending.get("output_text", "") or "")[:500],
            "output_cell_ids": [str(cell_id or "") for cell_id in list(pending.get("output_cell_ids", []) or []) if str(cell_id or "")][:12],
            "evidence_refs": [dict(item) for item in list(pending.get("evidence_refs", []) or []) if isinstance(item, Mapping)][:8],
            "pattern_refs": [dict(item) for item in list(pending.get("pattern_refs", []) or []) if isinstance(item, Mapping)][:8],
            "memory_evidence_trace": [dict(item) for item in list(pending.get("memory_evidence_trace", []) or []) if isinstance(item, Mapping)][:18],
            "committed_at": _safe_float(pending.get("committed_at", 0.0), 0.0),
            "observed_at": now,
            "latency_s": round(max(0.0, now - _safe_float(pending.get("committed_at", now), now)), 4),
            "learning_applied": bool(learn and reliability > 0.0),
            "durable_memory": False,
        }

        if learn and reliability > 0.0:
            await self._update_learning(ctx, outcome)
        await self._append_history(ctx, outcome)
        await self._mark_hypothesis_tested(ctx, outcome)
        await ctx.set_kv("hypothesis:last_outcome", outcome)
        await ctx.set_kv("hypothesis:pending_outcome", {})
        current_pending = await ctx.get_kv("hypothesis:pending_action", {})
        if isinstance(current_pending, Mapping) and str(current_pending.get("hypothesis_id", "") or "") == outcome["hypothesis_id"]:
            await ctx.set_kv("hypothesis:pending_action", {})

        self.debug(
            "hypothesis_outcome",
            hypothesis_id=outcome["hypothesis_id"],
            action=outcome["selected_action"],
            status=status,
            score=outcome["score"],
            reliability=outcome["reliability"],
            prediction_error=outcome["prediction_error"],
            reason=reason,
        )

        meta = {
            "kind": "hypothesis.outcome",
            "channel": "thought",
            "store_in_memory": False,
            "reinforcement_eligible": False,
            "self_output_track": False,
            "cognitive_visible": False,
            "ephemeral": True,
        }
        return [
            Event(
                topic=OUTCOME_TOPIC,
                payload=outcome,
                source=self.name,
                correlation_id=event.correlation_id,
                meta=meta,
            )
        ]

    async def _update_learning(self, ctx, outcome: Mapping[str, Any]) -> None:
        raw = await ctx.get_kv("hypothesis:action_learning", {})
        learning = dict(raw) if isinstance(raw, Mapping) else {}
        kind = str(outcome.get("statement_kind", "statement") or "statement")
        action = str(outcome.get("selected_action", "") or "silence")
        score = _safe_float(outcome.get("score", 0.0), 0.0)
        reliability = max(0.0, min(1.0, _safe_float(outcome.get("reliability", 0.0), 0.0)))
        for key, share in ((f"{kind}|{action}", 1.0), (f"*|{action}", GENERAL_ACTION_BUCKET_SHARE)):
            weight = reliability * share
            if weight <= 0.0:
                continue
            bucket = dict(learning.get(key, {}) or {})
            observations = int(bucket.get("observations", 0) or 0) + 1
            weight_sum = _safe_float(bucket.get("weight_sum", 0.0), 0.0) + weight
            score_sum = _safe_float(bucket.get("score_sum", 0.0), 0.0) + (score * weight)
            bucket.update(
                {
                    "statement_kind": kind if not key.startswith("*") else "*",
                    "action": action,
                    "observations": observations,
                    "weight_sum": round(weight_sum, 6),
                    "score_sum": round(score_sum, 6),
                    "avg_score": round(score_sum / max(1e-9, weight_sum), 6),
                    "positive": int(bucket.get("positive", 0) or 0) + (1 if score >= POSITIVE_OUTCOME_THRESHOLD else 0),
                    "negative": int(bucket.get("negative", 0) or 0) + (1 if score <= NEGATIVE_OUTCOME_THRESHOLD else 0),
                    "neutral": int(bucket.get("neutral", 0) or 0)
                    + (1 if NEGATIVE_OUTCOME_THRESHOLD < score < POSITIVE_OUTCOME_THRESHOLD else 0),
                    "confidence": round(min(1.0, weight_sum / LEARNING_CONFIDENCE_OBSERVATIONS), 6),
                    "last_reason": str(outcome.get("reason", "") or ""),
                    "last_outcome_id": str(outcome.get("outcome_id", "") or ""),
                    "last_at": _safe_float(outcome.get("observed_at", time.time()), time.time()),
                }
            )
            learning[key] = bucket
        await ctx.set_kv("hypothesis:action_learning", learning)

    async def _append_history(self, ctx, outcome: Mapping[str, Any]) -> None:
        history = await ctx.get_kv("hypothesis:outcome_history", [])
        history = list(history) if isinstance(history, list) else []
        history.append(dict(outcome))
        await ctx.set_kv("hypothesis:outcome_history", history[-OUTCOME_HISTORY_LIMIT:])

    async def _mark_hypothesis_tested(self, ctx, outcome: Mapping[str, Any]) -> None:
        last = await ctx.get_kv("hypothesis:last", {})
        if isinstance(last, Mapping) and str(last.get("hypothesis_id", "") or "") == str(outcome.get("hypothesis_id", "") or ""):
            updated = dict(last)
            state = dict(updated.get("state", {}) or {})
            state.update({"tested": True, "awaiting_outcome": False, "last_outcome_id": outcome.get("outcome_id", "")})
            updated["state"] = state
            updated["outcome"] = {
                "status": outcome.get("status"),
                "score": outcome.get("score"),
                "reliability": outcome.get("reliability"),
                "prediction_error": outcome.get("prediction_error"),
                "reason": outcome.get("reason"),
            }
            await ctx.set_kv("hypothesis:last", updated)

        history = await ctx.get_kv("hypothesis:history", [])
        if isinstance(history, list):
            changed = False
            rows = []
            for row in history:
                if isinstance(row, Mapping) and str(row.get("hypothesis_id", "") or "") == str(outcome.get("hypothesis_id", "") or ""):
                    row = dict(row)
                    row.update(
                        {
                            "outcome_status": outcome.get("status"),
                            "outcome_score": outcome.get("score"),
                            "outcome_reliability": outcome.get("reliability"),
                            "prediction_error": outcome.get("prediction_error"),
                        }
                    )
                    changed = True
                rows.append(row)
            if changed:
                await ctx.set_kv("hypothesis:history", rows)

    async def _pending(self, ctx) -> Dict[str, Any]:
        raw = await ctx.get_kv("hypothesis:pending_outcome", {})
        return dict(raw) if isinstance(raw, Mapping) and raw else {}

    def _candidate_for(self, hypothesis: Mapping[str, Any], action: str) -> Dict[str, Any]:
        for item in list(hypothesis.get("action_candidates", []) or []):
            if isinstance(item, Mapping) and str(item.get("action", "") or "") == action:
                return dict(item)
        return {}

    def _is_external_text(self, event: Event) -> bool:
        if bool((event.meta or {}).get("control", False)):
            return False
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        source = str(payload.get("source", event.source or "user") or "user")
        channel = str(payload.get("channel", "default") or "default")
        return source not in {"assistant", "system"} and channel not in {"thought", "internal", "control"}

    def _same_trigger(self, pending: Mapping[str, Any], event: Event) -> bool:
        pending_corr = str(pending.get("correlation_id", "") or "")
        if pending_corr and pending_corr == str(event.correlation_id or ""):
            return True
        return False

    def _correlation_matches(self, pending: Mapping[str, Any], event: Event) -> bool:
        pending_corr = str(pending.get("correlation_id", "") or "")
        event_corr = str(event.correlation_id or "")
        return not pending_corr or not event_corr or pending_corr == event_corr

    def _reinforcement_matches(self, pending: Mapping[str, Any], event: Event) -> bool:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        target = payload.get("target", {}) if isinstance(payload.get("target", {}), Mapping) else {}
        role = str(payload.get("target_role", target.get("role", "")) or "")
        if role and role != "assistant":
            return False
        target_text = str(target.get("text", target.get("utterance", "")) or "").strip()
        output_text = str(pending.get("output_text", "") or "").strip()
        if target_text and output_text:
            return normalize_text(target_text) == normalize_text(output_text) or jaccard(
                tokenize(target_text, meaningful=True),
                tokenize(output_text, meaningful=True),
            ) >= REINFORCEMENT_TARGET_SIMILARITY
        return bool(pending.get("action_executed", False))

    def _trainer_matches(self, pending: Mapping[str, Any], event: Event) -> bool:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        target = payload.get("target", {}) if isinstance(payload.get("target", {}), Mapping) else {}
        target_text = str(target.get("utterance", target.get("text", "")) or "").strip()
        output_text = str(pending.get("output_text", "") or "").strip()
        if target_text and output_text:
            return normalize_text(target_text) == normalize_text(output_text) or jaccard(
                tokenize(target_text, meaningful=True),
                tokenize(output_text, meaningful=True),
            ) >= TRAINER_TARGET_SIMILARITY
        return bool(pending.get("action_executed", False))

    def _outcome_id(self, pending: Mapping[str, Any], now: float, reason: str) -> str:
        seed = f"{pending.get('hypothesis_id', '')}|{pending.get('selected_action', '')}|{reason}|{now:.6f}".encode("utf-8", errors="ignore")
        return f"outcome:{hashlib.blake2s(seed, digest_size=10).hexdigest()}"


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            ACTION_COMMITTED_TOPIC,
            SPEECH_TOPIC,
            TEXT_TOPIC,
            REINFORCE_TOPIC,
            TRAINER_CORRECTION_TOPIC,
            INTERACTION_RELIEF_TOPIC,
            CLOCK_TOPIC,
        ],
        output_topics=[OUTCOME_TOPIC],
        priority=30,
        cooldown_sec=0.0,
    )
    yield HypothesisOutcomeObserverNeuron(cfg)
