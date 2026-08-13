from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Mapping


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def clamp_signed(value: float, limit: float = 1.0) -> float:
    return clamp(value, -abs(limit), abs(limit))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class AffectCurveSpec:
    """
    Time-shaped affect/reward curve definition.

    flow_threshold is the input cap over flow_window_s.
    curve_capacity is how much of the state can be meaningfully held before saturation.
    decay_half_life_s is the settling/drain speed.
    repeat_penalty limits repeated praise/correction for the same recent target.
    """

    name: str
    flow_threshold: float = 1.0
    flow_window_s: float = 8.0
    curve_capacity: float = 1.0
    decay_half_life_s: float = 8.0
    repeat_window_s: float = 18.0
    repeat_penalty: float = 0.12
    repeat_floor: float = 0.25


@dataclass
class AffectCurveState:
    level: float = 0.0
    flow: float = 0.0
    saturation: float = 0.0
    overload: float = 0.0
    last_ts: float = 0.0
    last_pulse_ts: float = 0.0
    last_target_key: str = ""
    repeat_count: int = 0
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "AffectCurveState":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            level=safe_float(data.get("level"), 0.0),
            flow=max(0.0, safe_float(data.get("flow"), 0.0)),
            saturation=clamp(safe_float(data.get("saturation"), 0.0)),
            overload=clamp(safe_float(data.get("overload"), 0.0)),
            last_ts=max(0.0, safe_float(data.get("last_ts"), 0.0)),
            last_pulse_ts=max(0.0, safe_float(data.get("last_pulse_ts"), 0.0)),
            last_target_key=str(data.get("last_target_key", "") or ""),
            repeat_count=max(0, int(data.get("repeat_count", 0) or 0)),
            sources=[str(x) for x in data.get("sources", [])[:8]] if isinstance(data.get("sources"), list) else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["level"] = round(float(out["level"]), 4)
        out["flow"] = round(float(out["flow"]), 4)
        out["saturation"] = round(float(out["saturation"]), 4)
        out["overload"] = round(float(out["overload"]), 4)
        return out


@dataclass(frozen=True)
class AffectPulseResult:
    curve_name: str
    signed_input: float
    accepted_input: float
    effective: float
    flow_available: float
    capacity_remaining: float
    saturation: float
    overload: float
    repeat_multiplier: float
    state: AffectCurveState

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve_name": self.curve_name,
            "signed_input": round(self.signed_input, 4),
            "accepted_input": round(self.accepted_input, 4),
            "effective": round(self.effective, 4),
            "flow_available": round(self.flow_available, 4),
            "capacity_remaining": round(self.capacity_remaining, 4),
            "saturation": round(self.saturation, 4),
            "overload": round(self.overload, 4),
            "repeat_multiplier": round(self.repeat_multiplier, 4),
            "state": self.state.to_dict(),
        }


DEFAULT_CURVE_SPECS: Dict[str, AffectCurveSpec] = {
    "user_approval": AffectCurveSpec(
        name="user_approval",
        flow_threshold=1.00,
        flow_window_s=8.0,
        curve_capacity=1.00,
        decay_half_life_s=9.0,
        repeat_window_s=20.0,
        repeat_penalty=0.14,
        repeat_floor=0.25,
    ),
    "user_correction": AffectCurveSpec(
        name="user_correction",
        flow_threshold=1.10,
        flow_window_s=10.0,
        curve_capacity=1.20,
        decay_half_life_s=12.0,
        repeat_window_s=20.0,
        repeat_penalty=0.10,
        repeat_floor=0.35,
    ),
    "arousal": AffectCurveSpec(
        name="arousal",
        flow_threshold=1.35,
        flow_window_s=6.0,
        curve_capacity=1.25,
        decay_half_life_s=5.0,
        repeat_window_s=8.0,
        repeat_penalty=0.08,
        repeat_floor=0.45,
    ),
    "overload": AffectCurveSpec(
        name="overload",
        flow_threshold=0.80,
        flow_window_s=7.0,
        curve_capacity=1.00,
        decay_half_life_s=10.0,
        repeat_window_s=14.0,
        repeat_penalty=0.05,
        repeat_floor=0.60,
    ),
    "task_commitment": AffectCurveSpec(
        name="task_commitment",
        flow_threshold=0.80,
        flow_window_s=14.0,
        curve_capacity=1.35,
        decay_half_life_s=24.0,
        repeat_window_s=30.0,
        repeat_penalty=0.05,
        repeat_floor=0.70,
    ),
}


def curve_spec(name: str, override: Mapping[str, Any] | None = None) -> AffectCurveSpec:
    base = DEFAULT_CURVE_SPECS.get(name, AffectCurveSpec(name=name))
    if not isinstance(override, Mapping):
        return base
    return AffectCurveSpec(
        name=str(override.get("name", base.name) or base.name),
        flow_threshold=max(0.001, safe_float(override.get("flow_threshold"), base.flow_threshold)),
        flow_window_s=max(0.001, safe_float(override.get("flow_window_s"), base.flow_window_s)),
        curve_capacity=max(0.001, safe_float(override.get("curve_capacity"), base.curve_capacity)),
        decay_half_life_s=max(0.001, safe_float(override.get("decay_half_life_s"), base.decay_half_life_s)),
        repeat_window_s=max(0.001, safe_float(override.get("repeat_window_s"), base.repeat_window_s)),
        repeat_penalty=clamp(safe_float(override.get("repeat_penalty"), base.repeat_penalty), 0.0, 1.0),
        repeat_floor=clamp(safe_float(override.get("repeat_floor"), base.repeat_floor), 0.0, 1.0),
    )


def _half_life_decay(value: float, dt_s: float, half_life_s: float) -> float:
    if dt_s <= 0.0 or value == 0.0:
        return value
    return value * math.pow(0.5, dt_s / max(0.001, half_life_s))


def decay_curve_state(
    state: Mapping[str, Any] | AffectCurveState | None,
    *,
    spec: AffectCurveSpec,
    now: float,
) -> AffectCurveState:
    cur = state if isinstance(state, AffectCurveState) else AffectCurveState.from_mapping(state)
    last_ts = cur.last_ts if cur.last_ts > 0.0 else now
    dt_s = max(0.0, now - last_ts)

    level = _half_life_decay(cur.level, dt_s, spec.decay_half_life_s)
    flow_drain_per_s = spec.flow_threshold / max(0.001, spec.flow_window_s)
    flow = max(0.0, cur.flow - (flow_drain_per_s * dt_s))
    overload = _half_life_decay(cur.overload, dt_s, max(0.001, spec.decay_half_life_s * 0.75))
    saturation = clamp(abs(level) / max(0.001, spec.curve_capacity))

    repeat_count = cur.repeat_count
    if cur.last_pulse_ts > 0.0 and (now - cur.last_pulse_ts) > spec.repeat_window_s:
        repeat_count = 0

    return AffectCurveState(
        level=clamp_signed(level, spec.curve_capacity),
        flow=round(flow, 6),
        saturation=saturation,
        overload=clamp(overload),
        last_ts=now,
        last_pulse_ts=cur.last_pulse_ts,
        last_target_key=cur.last_target_key,
        repeat_count=repeat_count,
        sources=list(cur.sources[-8:]),
    )


def decay_curve_map(
    curves: Mapping[str, Any] | None,
    *,
    now: float,
    specs: Mapping[str, AffectCurveSpec] | None = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(curves, Mapping):
        return out
    for name, raw_state in curves.items():
        spec = (specs or {}).get(str(name)) or curve_spec(str(name))
        out[str(name)] = decay_curve_state(raw_state, spec=spec, now=now).to_dict()
    return out


def apply_curve_pulse(
    curves: Mapping[str, Any] | None,
    *,
    name: str,
    signed_amount: float,
    now: float,
    spec: AffectCurveSpec | None = None,
    ddna_gain: float = 1.0,
    target_confidence: float = 1.0,
    novelty: float = 1.0,
    target_key: str = "",
    source: str = "",
) -> tuple[Dict[str, Dict[str, Any]], AffectPulseResult]:
    spec = spec or curve_spec(name)
    signed_amount = clamp_signed(safe_float(signed_amount, 0.0), 1.0)
    sign = -1.0 if signed_amount < 0.0 else 1.0
    incoming = abs(signed_amount) * clamp(safe_float(ddna_gain, 1.0), 0.10, 3.00)

    decayed_map = decay_curve_map(curves, now=now)
    cur = decay_curve_state(decayed_map.get(name), spec=spec, now=now)

    target_key = str(target_key or "")[:180]
    if target_key and target_key == cur.last_target_key and cur.last_pulse_ts > 0.0 and (now - cur.last_pulse_ts) <= spec.repeat_window_s:
        repeat_count = cur.repeat_count + 1
    else:
        repeat_count = 0

    repeat_multiplier = max(spec.repeat_floor, 1.0 - (spec.repeat_penalty * repeat_count))
    effective_novelty = clamp(safe_float(novelty, 1.0), 0.0, 1.0) * repeat_multiplier

    flow_available = clamp((spec.flow_threshold - cur.flow) / max(0.001, spec.flow_threshold))
    capacity_remaining = clamp((spec.curve_capacity - abs(cur.level)) / max(0.001, spec.curve_capacity))
    accepted_input = incoming * flow_available
    effective = accepted_input * capacity_remaining * clamp(safe_float(target_confidence, 1.0), 0.0, 1.0) * effective_novelty

    next_level = clamp_signed(cur.level + (sign * effective), spec.curve_capacity)
    next_flow = max(0.0, cur.flow + incoming)
    saturation = clamp(abs(next_level) / max(0.001, spec.curve_capacity))
    flow_overload = max(0.0, (next_flow - spec.flow_threshold) / max(0.001, spec.flow_threshold))
    capacity_overload = max(0.0, (abs(cur.level) + incoming - spec.curve_capacity) / max(0.001, spec.curve_capacity))
    overload = clamp(max(cur.overload, flow_overload, capacity_overload))

    sources = list(cur.sources[-7:])
    if source:
        sources.append(str(source)[:80])

    next_state = AffectCurveState(
        level=next_level,
        flow=round(next_flow, 6),
        saturation=saturation,
        overload=overload,
        last_ts=now,
        last_pulse_ts=now,
        last_target_key=target_key or cur.last_target_key,
        repeat_count=repeat_count,
        sources=sources,
    )
    decayed_map[name] = next_state.to_dict()
    result = AffectPulseResult(
        curve_name=name,
        signed_input=sign * incoming,
        accepted_input=accepted_input,
        effective=sign * effective,
        flow_available=flow_available,
        capacity_remaining=capacity_remaining,
        saturation=saturation,
        overload=overload,
        repeat_multiplier=repeat_multiplier,
        state=next_state,
    )
    return decayed_map, result


def signed_feedback_curve(
    curves: Mapping[str, Any] | None,
    *,
    signed_strength: float,
    now: float,
    ddna: Mapping[str, Any] | None = None,
    target_key: str = "",
    target_confidence: float = 1.0,
    novelty: float = 1.0,
    source: str = "acc",
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Apply /acc-style signed feedback to affect curves.

    Positive values fill user_approval. Negative values fill user_correction.
    The result exposes deltas that reward/novelty organs can consume, but it
    never directly selects actions or bypasses the governor.
    """

    signed_strength = clamp(safe_float(signed_strength, 0.0), -10.0, 10.0)
    if signed_strength == 0.0:
        return decay_curve_map(curves, now=now), {
            "curve_name": "neutral",
            "effective_reward": 0.0,
            "reward_delta": 0.0,
            "salience_delta": 0.0,
            "novelty_delta": 0.0,
            "boredom_relief_delta": 0.0,
            "satisfaction_delta": 0.0,
            "valence_delta": 0.0,
            "saturation": 0.0,
            "overload": 0.0,
        }

    mods = dict(ddna or {}) if isinstance(ddna, Mapping) else {}
    if signed_strength > 0.0:
        curve_name = "user_approval"
        gain = clamp(
            safe_float(mods.get("reward_gain"), 1.0)
            * safe_float(mods.get("trainer_alignment_gain"), 1.0)
            * safe_float(mods.get("social_reward_gain"), 1.0),
            0.20,
            2.50,
        )
        signed_amount = signed_strength / 10.0
    else:
        curve_name = "user_correction"
        gain = clamp(
            safe_float(mods.get("trainer_alignment_gain"), 1.0)
            * safe_float(mods.get("salience_gain"), 1.0),
            0.20,
            2.50,
        )
        signed_amount = signed_strength / 10.0

    next_curves, pulse = apply_curve_pulse(
        curves,
        name=curve_name,
        signed_amount=signed_amount,
        now=now,
        ddna_gain=gain,
        target_confidence=target_confidence,
        novelty=novelty,
        target_key=target_key,
        source=source,
    )

    eff = pulse.effective
    pos_eff = max(0.0, eff)
    neg_eff = abs(min(0.0, eff))
    raw_mag = abs(signed_strength) / 10.0

    if signed_strength > 0.0:
        deltas = {
            "curve_name": curve_name,
            "effective_reward": round(pos_eff, 4),
            "reward_delta": round(0.62 * pos_eff, 4),
            "salience_delta": round((0.10 * raw_mag * pulse.flow_available) + (0.22 * pos_eff), 4),
            "novelty_delta": round(0.10 * pos_eff * pulse.repeat_multiplier, 4),
            "boredom_relief_delta": round(0.22 * pos_eff, 4),
            "satisfaction_delta": round(0.18 * pos_eff, 4),
            "valence_delta": round(0.42 * pos_eff, 4),
        }
    else:
        deltas = {
            "curve_name": curve_name,
            "effective_reward": round(-neg_eff, 4),
            "reward_delta": round(-0.28 * neg_eff, 4),
            "salience_delta": round((0.16 * raw_mag * pulse.flow_available) + (0.20 * neg_eff), 4),
            "novelty_delta": round(0.05 * neg_eff, 4),
            "boredom_relief_delta": 0.0,
            "satisfaction_delta": 0.0,
            "valence_delta": round(-0.46 * neg_eff, 4),
        }

    deltas.update(
        {
            "saturation": round(pulse.saturation, 4),
            "overload": round(pulse.overload, 4),
            "flow_available": round(pulse.flow_available, 4),
            "capacity_remaining": round(pulse.capacity_remaining, 4),
            "repeat_multiplier": round(pulse.repeat_multiplier, 4),
            "pulse": pulse.to_dict(),
        }
    )
    return next_curves, deltas
