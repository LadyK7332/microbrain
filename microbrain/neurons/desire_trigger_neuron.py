from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Baseline pressure before statement-specific evidence is added.
# Range: 0.0-1.0. Higher values make outward release easier overall.
BASE_RELEASE_PRESSURE = 0.12

# Evidence contribution weights and per-source caps. Range: 0.0-1.0.
RESPONSE_DEMAND_PRESSURE_WEIGHT = 0.62
RESPONSE_DEMAND_PRESSURE_CAP = 0.62
USEFULNESS_PRESSURE_WEIGHT = 0.15
USEFULNESS_PRESSURE_CAP = 0.15
CONTINUITY_PRESSURE_WEIGHT = 0.08
CONTINUITY_PRESSURE_CAP = 0.08
RISK_PRESSURE_WEIGHT = 0.10
RISK_PRESSURE_CAP = 0.10

# Statement-kind bonuses. These are organ defaults; DDNA does not replace them.
STATEMENT_KIND_PRESSURE_BONUS = {
    "question": 0.18,
    "request": 0.16,
    "correction": 0.12,
    "disagreement": 0.12,
    "greeting": 0.12,
    "personal_state": 0.06,
    "status_update": 0.06,
}

# Additional state/evidence contributions.
MEMORY_SUPPORT_MIN_SCORE = 0.25
MEMORY_SUPPORT_PRESSURE_WEIGHT = 0.05
MEMORY_SUPPORT_PRESSURE_CAP = 0.05
CLARIFY_UNCERTAINTY_MIN = 0.60
CLARIFY_PRESSURE_BONUS = 0.05
BOREDOM_PRESSURE_WEIGHT = 0.05
BOREDOM_PRESSURE_CAP = 0.05
ALLOW_BABBLE_PRESSURE_BONUS = 0.03
CRISIS_PRESSURE_BONUS = 0.25
SILENCE_PRESSURE_WEIGHT = 0.28
SILENCE_PRESSURE_CAP = 0.28

# Baseline release thresholds by statement kind.
# Range: 0.0-1.0. Higher values make MB quieter/more selective.
DEFAULT_RESPONSE_THRESHOLD = 0.52
RESPONSE_THRESHOLD_BY_KIND = {
    "question": 0.44,
    "request": 0.44,
    "greeting": 0.45,
    "correction": 0.48,
    "disagreement": 0.48,
    "closure": 0.68,
}
CRISIS_RESPONSE_THRESHOLD = 0.35

# DDNA is allowed to alter the nature of response selection, but only within a
# bounded band. Positive restraint terms raise the threshold; positive
# expression/social/inquiry terms lower it.
DDNA_RESTRAINT_BIAS_WEIGHT = 0.035
DDNA_ACTION_GATE_WEIGHT = 0.035
DDNA_EXPRESSION_THRESHOLD_WEIGHT = 0.030
DDNA_EXPRESSION_BIAS_WEIGHT = 0.035
DDNA_EXPRESSION_ACTIVATION_WEIGHT = 0.025
DDNA_SOCIAL_GAIN_WEIGHT = 0.020
DDNA_INQUIRY_GAIN_WEIGHT = 0.020
DDNA_THRESHOLD_OFFSET_MIN = -0.08
DDNA_THRESHOLD_OFFSET_MAX = 0.08
EFFECTIVE_THRESHOLD_MIN = 0.30
EFFECTIVE_THRESHOLD_MAX = 0.75

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

# Bus routes and schema markers are protocol. Changing them requires updating
# every producer/subscriber rather than merely retuning behavior.
NEURON_NAME = Path(__file__).stem
HYPOTHESIS_READY_TOPIC = "hypothesis/ready"
HYPOTHESIS_ACTION_COMMITTED_TOPIC = "hypothesis/action_committed"
RELEASE_REQUEST_TOPIC = "release/request"
ACTION_COMMITTED_KIND = "hypothesis.action_committed"

_SOCIAL_STATEMENT_KINDS = {"agreement", "greeting", "personal_state", "status_update", "closure"}
_INQUIRY_STATEMENT_KINDS = {"question", "request", "correction", "disagreement", "claim"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class DesireTriggerNeuron(BaseNeuron):
    """Release arbitration after pattern and hypothesis formation.

    Silence is now an explicit candidate supplied by the hypothesis engine.  A
    plain participant statement can therefore earn a response route without a
    question mark, while low-value or closing statements can be deliberately
    left quiet.
    """

    async def _resolve_release_tuning(
        self,
        ctx,
        *,
        statement_kind: str,
        crisis_mode: bool,
    ) -> Dict[str, Any]:
        base_threshold = RESPONSE_THRESHOLD_BY_KIND.get(statement_kind, DEFAULT_RESPONSE_THRESHOLD)
        ddna = await ctx.get_kv("drive:ddna_modulators", {}) or {}
        ddna = ddna if isinstance(ddna, Mapping) else {}

        restraint_bias = _safe_float(ddna.get("restraint_bias", 1.0), 1.0)
        action_gate = _safe_float(ddna.get("action_gate_strictness", 1.0), 1.0)
        expression_threshold = _safe_float(ddna.get("expression_threshold_gain", 1.0), 1.0)
        expression_bias = _safe_float(ddna.get("expression_bias", 1.0), 1.0)
        expression_activation = _safe_float(ddna.get("expression_activation_gain", 1.0), 1.0)
        social_gain = _safe_float(ddna.get("social_gain", 1.0), 1.0)
        inquiry_gain = _safe_float(ddna.get("inquiry_gain", 1.0), 1.0)

        restraint_offset = (
            ((restraint_bias - 1.0) * DDNA_RESTRAINT_BIAS_WEIGHT)
            + ((action_gate - 1.0) * DDNA_ACTION_GATE_WEIGHT)
            + ((expression_threshold - 1.0) * DDNA_EXPRESSION_THRESHOLD_WEIGHT)
        )
        expression_offset = (
            ((expression_bias - 1.0) * DDNA_EXPRESSION_BIAS_WEIGHT)
            + ((expression_activation - 1.0) * DDNA_EXPRESSION_ACTIVATION_WEIGHT)
        )
        if statement_kind in _SOCIAL_STATEMENT_KINDS:
            expression_offset += (social_gain - 1.0) * DDNA_SOCIAL_GAIN_WEIGHT
        if statement_kind in _INQUIRY_STATEMENT_KINDS:
            expression_offset += (inquiry_gain - 1.0) * DDNA_INQUIRY_GAIN_WEIGHT

        raw_ddna_offset = restraint_offset - expression_offset
        ddna_offset = _clamp(raw_ddna_offset, DDNA_THRESHOLD_OFFSET_MIN, DDNA_THRESHOLD_OFFSET_MAX)

        # Crisis behavior is a shared safety constraint, not a temperament
        # preference. DDNA is reported but does not raise the crisis gate.
        if crisis_mode:
            effective_threshold = CRISIS_RESPONSE_THRESHOLD
            applied_ddna_offset = 0.0
        else:
            effective_threshold = _clamp(
                base_threshold + ddna_offset,
                EFFECTIVE_THRESHOLD_MIN,
                EFFECTIVE_THRESHOLD_MAX,
            )
            applied_ddna_offset = ddna_offset

        trace = {
            "name": "response_threshold",
            "statement_kind": statement_kind,
            "default": round(base_threshold, 6),
            "ddna_offset_raw": round(raw_ddna_offset, 6),
            "ddna_offset_applied": round(applied_ddna_offset, 6),
            "effective": round(effective_threshold, 6),
            "minimum": EFFECTIVE_THRESHOLD_MIN,
            "maximum": EFFECTIVE_THRESHOLD_MAX,
            "crisis_override": bool(crisis_mode),
            "ddna_inputs": {
                "restraint_bias": round(restraint_bias, 6),
                "action_gate_strictness": round(action_gate, 6),
                "expression_threshold_gain": round(expression_threshold, 6),
                "expression_bias": round(expression_bias, 6),
                "expression_activation_gain": round(expression_activation, 6),
                "social_gain": round(social_gain, 6),
                "inquiry_gain": round(inquiry_gain, 6),
            },
        }
        await ctx.set_kv("hypothesis:release_tuning", trace)
        return trace

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != HYPOTHESIS_READY_TOPIC:
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
        input_block = context.get("input", {}) if isinstance(context.get("input", {}), Mapping) else {}
        text = str(input_block.get("text", "") or "").strip()
        if not text or not hypothesis:
            return []

        source = str(input_block.get("source", payload.get("source", "user")) or "user")
        channel = str(input_block.get("channel", payload.get("channel", "default")) or "default")
        raw_meta = dict(input_block.get("raw_meta", payload.get("raw_meta", {})) or {})

        cues = context.get("cues", {}) if isinstance(context.get("cues", {}), Mapping) else {}
        constraints = context.get("constraints", {}) if isinstance(context.get("constraints", {}), Mapping) else {}
        boredom = ((context.get("drives", {}) or {}).get("boredom", {}) or {})
        assoc_meta = context.get("association_meta", {}) if isinstance(context.get("association_meta", {}), Mapping) else {}
        pattern = hypothesis.get("pattern_analysis", {}) if isinstance(hypothesis.get("pattern_analysis", {}), Mapping) else {}

        response_demand = _safe_float(hypothesis.get("response_demand", 0.0))
        usefulness = _safe_float(hypothesis.get("expected_usefulness", 0.0))
        silence_score = _safe_float(hypothesis.get("silence_score", 0.0))
        recommended_action = str(hypothesis.get("recommended_action", "silence") or "silence")
        hypothesis_should_respond = bool(hypothesis.get("should_respond", False))
        continuity = _safe_float(pattern.get("continuity", 0.0))
        risk = _safe_float(pattern.get("risk", 0.0))
        uncertainty = _safe_float(pattern.get("uncertainty", 0.0))
        statement_kind = str(pattern.get("statement_kind", "statement") or "statement")

        boredom_level = _safe_float(boredom.get("level", 0.0))
        crisis_mode = bool(constraints.get("crisis_mode", False))
        allow_babble = bool(constraints.get("allow_babble", False))
        top_assoc_score = _safe_float(assoc_meta.get("top_score", 0.0))

        pressure = BASE_RELEASE_PRESSURE
        reasons = ["hypothesis_pass"]
        pressure += min(RESPONSE_DEMAND_PRESSURE_CAP, response_demand * RESPONSE_DEMAND_PRESSURE_WEIGHT)
        pressure += min(USEFULNESS_PRESSURE_CAP, usefulness * USEFULNESS_PRESSURE_WEIGHT)
        pressure += min(CONTINUITY_PRESSURE_CAP, continuity * CONTINUITY_PRESSURE_WEIGHT)
        pressure += min(RISK_PRESSURE_CAP, risk * RISK_PRESSURE_WEIGHT)
        reasons.extend(["response_demand", f"action:{recommended_action}"])

        kind_bonus = STATEMENT_KIND_PRESSURE_BONUS.get(statement_kind, 0.0)
        if kind_bonus:
            pressure += kind_bonus
            reason = {
                "correction": "context_revision",
                "disagreement": "context_revision",
                "personal_state": "social_state",
                "status_update": "social_state",
            }.get(statement_kind, statement_kind)
            reasons.append(reason)

        if top_assoc_score >= MEMORY_SUPPORT_MIN_SCORE:
            pressure += min(MEMORY_SUPPORT_PRESSURE_CAP, top_assoc_score * MEMORY_SUPPORT_PRESSURE_WEIGHT)
            reasons.append("memory_support")
        if uncertainty >= CLARIFY_UNCERTAINTY_MIN and recommended_action == "clarify":
            pressure += CLARIFY_PRESSURE_BONUS
            reasons.append("clarification_needed")
        if boredom_level > 0.0:
            pressure += min(BOREDOM_PRESSURE_CAP, boredom_level * BOREDOM_PRESSURE_WEIGHT)
            reasons.append("boredom")
        if allow_babble:
            pressure += ALLOW_BABBLE_PRESSURE_BONUS
            reasons.append("allow_babble")
        if crisis_mode:
            pressure += CRISIS_PRESSURE_BONUS
            reasons.append("crisis_mode")

        # Silence is a real competing action, not merely absence of a route.
        pressure -= min(SILENCE_PRESSURE_CAP, silence_score * SILENCE_PRESSURE_WEIGHT)
        pressure = max(0.0, pressure)

        tuning = await self._resolve_release_tuning(
            ctx,
            statement_kind=statement_kind,
            crisis_mode=crisis_mode,
        )
        threshold = _safe_float(tuning.get("effective", DEFAULT_RESPONSE_THRESHOLD), DEFAULT_RESPONSE_THRESHOLD)

        explicit_silence = recommended_action == "silence" or not hypothesis_should_respond
        should_release = pressure >= threshold and not explicit_silence
        trigger = {
            "pressure": round(pressure, 6),
            "threshold": round(threshold, 6),
            "reasons": reasons,
            "should_release": bool(should_release),
            "kind": self._classify_trigger(
                statement_kind=statement_kind,
                recommended_action=recommended_action,
                crisis_mode=crisis_mode,
            ),
            "recommended_action": recommended_action,
            "response_demand": round(response_demand, 6),
            "expected_usefulness": round(usefulness, 6),
            "silence_score": round(silence_score, 6),
            "deliberate_silence": bool(explicit_silence or not should_release),
            "hypothesis_id": str(hypothesis.get("hypothesis_id", "") or ""),
            "tuning": tuning,
        }

        await ctx.set_kv("desire:last_trigger", trigger)
        await ctx.set_kv(
            "hypothesis:last_release_decision",
            {
                "hypothesis_id": trigger["hypothesis_id"],
                "recommended_action": recommended_action,
                "pressure": trigger["pressure"],
                "threshold": trigger["threshold"],
                "should_release": trigger["should_release"],
                "deliberate_silence": trigger["deliberate_silence"],
            },
        )

        self.debug(
            "release_check",
            pressure=round(pressure, 3),
            threshold=round(threshold, 3),
            release=should_release,
            deliberate_silence=trigger["deliberate_silence"],
            statement_kind=statement_kind,
            recommended_action=recommended_action,
            response_demand=round(response_demand, 3),
            silence_score=round(silence_score, 3),
        )

        action_event = Event(
            topic=HYPOTHESIS_ACTION_COMMITTED_TOPIC,
            payload={
                "context": dict(context),
                "hypothesis": dict(hypothesis),
                "trigger": dict(trigger),
                "source": source,
                "channel": channel,
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": ACTION_COMMITTED_KIND,
                "stage": "committed",
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "ephemeral": True,
                "hypothesis_id": trigger["hypothesis_id"],
                "selected_action": recommended_action,
                "deliberate_silence": trigger["deliberate_silence"],
            },
        )

        if not should_release:
            return [action_event]

        return [
            action_event,
            Event(
                topic=RELEASE_REQUEST_TOPIC,
                payload={
                    "context": dict(context),
                    "hypothesis": dict(hypothesis),
                    "trigger": trigger,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "contextual": True,
                    "stage": "triggered",
                    "kind": trigger["kind"],
                    "hypothesis_id": trigger["hypothesis_id"],
                },
            )
        ]

    def _classify_trigger(self, *, statement_kind: str, recommended_action: str, crisis_mode: bool) -> str:
        if crisis_mode:
            return "crisis"
        if recommended_action:
            return recommended_action
        return statement_kind or "contextual"


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[HYPOTHESIS_READY_TOPIC],
        output_topics=[HYPOTHESIS_ACTION_COMMITTED_TOPIC, RELEASE_REQUEST_TOPIC],
        priority=12,
    )
    yield DesireTriggerNeuron(cfg)
