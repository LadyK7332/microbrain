from __future__ import annotations

"""
Affect curve helpers for MB's reward/novelty/hormone field.

This module is deliberately pure-Python and side-effect free.  It does not own
reward, speech, action, or memory.  It only shapes pulses before an existing
organ applies them.
"""

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any, Mapping


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class AffectCurveConfig:
    """Configuration for a single affect field curve.

    flow_threshold:
        How much signed/absolute input can enter during flow_window_s before
        overload starts reducing effectiveness.

    curve_capacity:
        How full the curve can get before new input stops being meaningful.

    decay_half_life_s:
        How long it takes the active level/saturation to halve without refresh.

    overload_half_life_s:
        How long overload takes to settle.

    repeat_dampening:
        How much repeated pulses against the same target lose value per repeat.
    """

    name: str
    flow_threshold: float = 1.0
    flow_window_s: float = 10.0
    curve_capacity: float = 1.0
    decay_half_life_s: float = 18.0
    overload_half_life_s: float = 10.0
    repeat_window_s: float = 6.0
    repeat_dampening: float = 0.18
    min_effective: float = 0.0


@dataclass(frozen=True)
class AffectPulseResult:
    curve: str
    raw_strength: float
    effective_strength: float
    multiplier: float
    flow_available: float
    capacity_remaining: float
    saturation: float
    overload: float
    repeat_damp: float
    level: float
    reason: str
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Keep the nested state but avoid carrying unreadable float noise.
        data["state"] = normalize_curve_state(self.state)
        for key in (
            "raw_strength",
            "effective_strength",
            "multiplier",
            "flow_available",
            "capacity_remaining",
            "saturation",
            "overload",
            "repeat_damp",
            "level",
        ):
            data[key] = round(safe_float(data.get(key), 0.0), 4)
        return data


def _half_life_decay(level: float, dt_s: float, half_life_s: float) -> float:
    level = max(0.0, safe_float(level, 0.0))
    dt_s = max(0.0, safe_float(dt_s, 0.0))
    half_life_s = max(0.001, safe_float(half_life_s, 1.0))
    if level <= 0.0 or dt_s <= 0.0:
        return level
    return level * exp(-log(2.0) * (dt_s / half_life_s))


def normalize_curve_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(state or {})
    return {
        "level": round(clamp(safe_float(data.get("level"), 0.0)), 4),
        "flow_used": round(max(0.0, safe_float(data.get("flow_used"), 0.0)), 4),
        "flow_window_start": round(max(0.0, safe_float(data.get("flow_window_start"), 0.0)), 4),
        "saturation": round(clamp(safe_float(data.get("saturation"), 0.0)), 4),
        "overload": round(clamp(safe_float(data.get("overload"), 0.0)), 4),
        "last_ts": round(max(0.0, safe_float(data.get("last_ts"), 0.0)), 4),
        "last_target": str(data.get("last_target", "") or "")[:160],
        "repeat_count": int(max(0, int(data.get("repeat_count", 0) or 0))),
    }


def decay_curve_state(
    state: Mapping[str, Any] | None,
    config: AffectCurveConfig,
    *,
    now: float,
) -> dict[str, Any]:
    """Return a decayed copy of a curve state."""
    st = normalize_curve_state(state)
    now = max(0.0, safe_float(now, 0.0))
    last_ts = safe_float(st.get("last_ts"), now)
    dt_s = max(0.0, now - last_ts)

    st["level"] = round(clamp(_half_life_decay(st.get("level", 0.0), dt_s, config.decay_half_life_s)), 4)
    st["saturation"] = round(clamp(_half_life_decay(st.get("saturation", 0.0), dt_s, config.decay_half_life_s * 1.35)), 4)
    st["overload"] = round(clamp(_half_life_decay(st.get("overload", 0.0), dt_s, config.overload_half_life_s)), 4)

    # Flow windows are discrete buckets.  Once the window expires, new input can
    # enter again without treating the old bucket as immediate overload.
    window_start = safe_float(st.get("flow_window_start"), 0.0)
    if window_start <= 0.0 or (now - window_start) >= max(0.001, config.flow_window_s):
        st["flow_window_start"] = round(now, 4)
        st["flow_used"] = 0.0
    else:
        st["flow_used"] = round(max(0.0, safe_float(st.get("flow_used"), 0.0)), 4)

    st["last_ts"] = round(now, 4)
    return st


def apply_affect_pulse(
    state: Mapping[str, Any] | None,
    config: AffectCurveConfig,
    *,
    now: float,
    incoming_strength: float,
    target_key: str = "",
    target_confidence: float = 1.0,
    novelty: float = 1.0,
    ddna_gain: float = 1.0,
) -> AffectPulseResult:
    """Apply a pulse to a curve and return the shaped/effective result.

    incoming_strength is expected as 0..1, but is clamped safely.  The result's
    effective_strength is also 0..1 and can be used as the new field input; the
    multiplier can be used to scale legacy deltas.
    """
    raw = clamp(safe_float(incoming_strength, 0.0), 0.0, 1.0)
    now = max(0.0, safe_float(now, 0.0))
    st = decay_curve_state(state, config, now=now)

    flow_threshold = max(0.001, safe_float(config.flow_threshold, 1.0))
    flow_used = max(0.0, safe_float(st.get("flow_used"), 0.0))
    flow_available = clamp((flow_threshold - flow_used) / flow_threshold, 0.0, 1.0)
    if flow_available <= 0.005:
        flow_available = 0.0

    level = clamp(safe_float(st.get("level"), 0.0))
    capacity = max(0.001, safe_float(config.curve_capacity, 1.0))
    capacity_remaining = clamp((capacity - level) / capacity, 0.0, 1.0)
    if capacity_remaining <= 0.005:
        capacity_remaining = 0.0

    target_key = str(target_key or "")[:160]
    last_target = str(st.get("last_target", "") or "")
    repeat_count = int(st.get("repeat_count", 0) or 0)
    last_ts = safe_float(st.get("last_ts"), now)
    if target_key and target_key == last_target and (now - last_ts) <= max(0.0, config.repeat_window_s):
        repeat_count += 1
    elif target_key:
        repeat_count = 0
    repeat_damp = clamp(1.0 - (config.repeat_dampening * min(5, repeat_count)), 0.10, 1.0)

    overload = clamp(safe_float(st.get("overload"), 0.0))
    confidence = clamp(safe_float(target_confidence, 1.0), 0.0, 1.0)
    novelty = clamp(safe_float(novelty, 1.0), 0.0, 1.0)
    ddna_gain = clamp(safe_float(ddna_gain, 1.0), 0.20, 2.00)

    # Flow controls "too much too fast".  Capacity controls "this curve is full".
    # Overload suppresses usefulness until it settles.
    shaped = raw * flow_available * capacity_remaining * repeat_damp * confidence * novelty * ddna_gain * (1.0 - (0.65 * overload))
    effective = clamp(max(config.min_effective, shaped), 0.0, 1.0)
    multiplier = 0.0 if raw <= 0.0 else clamp(effective / raw, 0.0, 1.5)

    st["level"] = round(clamp(level + effective), 4)
    st["saturation"] = round(clamp(st["level"] / capacity), 4)
    st["flow_used"] = round(flow_used + raw, 4)
    # Input beyond the flow threshold becomes temporary overload instead of just
    # disappearing.  This models the baby/too-many-relatives curve.
    excess_flow = max(0.0, (flow_used + raw) - flow_threshold) / flow_threshold
    st["overload"] = round(clamp(overload + excess_flow * 0.45), 4)
    st["last_target"] = target_key
    st["repeat_count"] = repeat_count
    st["last_ts"] = round(now, 4)

    if raw <= 0.0:
        reason = "no_pulse"
    elif st["overload"] > 0.75:
        reason = "overload_limited"
    elif capacity_remaining <= 0.05:
        reason = "capacity_saturated"
    elif flow_available <= 0.05:
        reason = "flow_limited"
    elif repeat_count > 0:
        reason = "repeat_dampened"
    else:
        reason = "accepted"

    return AffectPulseResult(
        curve=config.name,
        raw_strength=raw,
        effective_strength=effective,
        multiplier=multiplier,
        flow_available=flow_available,
        capacity_remaining=capacity_remaining,
        saturation=safe_float(st.get("saturation"), 0.0),
        overload=safe_float(st.get("overload"), 0.0),
        repeat_damp=repeat_damp,
        level=safe_float(st.get("level"), 0.0),
        reason=reason,
        state=st,
    )


def decay_curve_bucket(
    curves: Mapping[str, Any] | None,
    configs: Mapping[str, AffectCurveConfig],
    *,
    now: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    raw = dict(curves or {})
    for name, cfg in configs.items():
        out[name] = decay_curve_state(raw.get(name), cfg, now=now)
    return out


def summarize_curve_bucket(curves: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, state in dict(curves or {}).items():
        st = normalize_curve_state(state if isinstance(state, Mapping) else {})
        out[str(name)] = {
            "level": st["level"],
            "saturation": st["saturation"],
            "overload": st["overload"],
            "flow_used": st["flow_used"],
            "repeat_count": st["repeat_count"],
            "last_target": st["last_target"],
        }
    return out
