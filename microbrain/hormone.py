from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

HORMONE_KEYS = (
    "arousal",
    "inquiry",
    "affiliation",
    "caution",
    "frustration",
    "settling",
    "persistence",
    "continuity",
)

NEED_KEYS = (
    "stimulation",
    "social",
    "coherence",
    "continuity",
    "safety",
    "salience",
    "novelty",
    "maintenance",
)

DDNA_MOD_KEYS = (
    "arousal_gain",
    "inquiry_gain",
    "affiliation_gain",
    "social_gain",
    "caution_gain",
    "frustration_gain",
    "settling_gain",
    "persistence_gain",
    "continuity_gain",
    "novelty_gain",
    "expression_bias",
    "restraint_bias",
    "volatility",
)


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


@dataclass
class HormoneState:
    arousal: float = 0.15
    inquiry: float = 0.10
    affiliation: float = 0.10
    caution: float = 0.20
    frustration: float = 0.05
    settling: float = 0.80
    persistence: float = 0.45
    continuity: float = 0.12

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "HormoneState":
        if not isinstance(data, Mapping):
            return cls()
        kwargs = {k: clamp(safe_float(data.get(k), getattr(cls, k, 0.0))) for k in HORMONE_KEYS if k in data}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, float]:
        return {k: round(clamp(v), 4) for k, v in asdict(self).items()}


DEFAULT_DDNA_MODULATORS: Dict[str, float] = {
    "arousal_gain": 1.00,
    "inquiry_gain": 1.00,
    "affiliation_gain": 1.00,
    "social_gain": 1.00,
    "caution_gain": 1.00,
    "frustration_gain": 1.00,
    "settling_gain": 1.00,
    "persistence_gain": 1.00,
    "continuity_gain": 1.00,
    "novelty_gain": 1.00,
    "expression_bias": 1.00,
    "restraint_bias": 1.00,
    "volatility": 1.00,
}


def derive_ddna_modulators(pdna: Any) -> Dict[str, float]:
    mods = dict(DEFAULT_DDNA_MODULATORS)
    if pdna is None:
        return mods

    warmth = clamp(safe_float(getattr(pdna, "warmth", 0.6), 0.6))
    playfulness = clamp(safe_float(getattr(pdna, "playfulness", 0.5), 0.5))
    flirtation = clamp(safe_float(getattr(pdna, "flirtation", 0.3), 0.3))
    formality = clamp(safe_float(getattr(pdna, "formality", 0.3), 0.3))
    introspection = clamp(safe_float(getattr(pdna, "introspection", 0.6), 0.6))
    safety = clamp(safe_float(getattr(pdna, "safety_orientation", 0.9), 0.9))
    focus = clamp(safe_float(getattr(pdna, "focus", 0.6), 0.6))
    energy = clamp(safe_float(getattr(pdna, "energy", 0.5), 0.5))
    support = clamp(safe_float(getattr(pdna, "support_level", 0.7), 0.7))

    mods["arousal_gain"] = clamp(0.70 + (0.45 * energy) + (0.12 * playfulness), 0.35, 1.75)
    mods["inquiry_gain"] = clamp(0.70 + (0.38 * introspection) + (0.18 * focus) + (0.10 * playfulness), 0.35, 1.75)
    mods["affiliation_gain"] = clamp(0.70 + (0.35 * warmth) + (0.20 * support) + (0.05 * flirtation), 0.35, 1.75)
    mods["social_gain"] = clamp(0.60 + (0.25 * warmth) + (0.22 * support) + (0.08 * flirtation), 0.35, 1.75)
    mods["caution_gain"] = clamp(0.55 + (0.45 * safety) + (0.18 * formality) + (0.10 * focus), 0.35, 1.90)
    mods["frustration_gain"] = clamp(0.70 + (0.18 * energy) + (0.08 * focus) - (0.15 * warmth), 0.25, 1.75)
    mods["settling_gain"] = clamp(0.55 + (0.30 * warmth) + (0.20 * safety) + (0.10 * support), 0.30, 1.75)
    mods["persistence_gain"] = clamp(0.55 + (0.32 * focus) + (0.18 * introspection) + (0.10 * safety), 0.30, 1.90)
    mods["continuity_gain"] = clamp(0.60 + (0.25 * focus) + (0.18 * introspection) + (0.12 * support), 0.30, 1.90)
    mods["novelty_gain"] = clamp(0.55 + (0.28 * playfulness) + (0.18 * energy) - (0.10 * formality), 0.25, 1.75)
    mods["expression_bias"] = clamp(0.55 + (0.25 * warmth) + (0.15 * support) + (0.10 * energy) - (0.15 * formality), 0.25, 1.75)
    mods["restraint_bias"] = clamp(0.55 + (0.35 * safety) + (0.25 * formality) + (0.08 * introspection) - (0.12 * playfulness), 0.25, 1.90)
    mods["volatility"] = clamp(0.45 + (0.28 * energy) + (0.12 * playfulness) - (0.18 * safety), 0.20, 1.60)

    overrides = getattr(pdna, "hormone_overrides", {})
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            if key in mods:
                mods[key] = clamp(safe_float(value, mods[key]), 0.20, 2.00)

    return {k: round(v, 4) for k, v in mods.items()}


def merge_need_maps(*maps: Mapping[str, Any] | None) -> Dict[str, float]:
    out = {k: 0.0 for k in NEED_KEYS}
    for data in maps:
        if not isinstance(data, Mapping):
            continue
        for key in NEED_KEYS:
            if key in data:
                out[key] = clamp(safe_float(data.get(key), out[key]))
    return {k: round(v, 4) for k, v in out.items()}


def compute_base_needs(
    *,
    boredom_level: float,
    stress_level: float,
    salience: float,
    now: float,
    last_user_ts: float,
    last_external_ts: float,
    sleeping: bool,
    charging: bool,
    unresolved_pending: bool,
    pending_age_s: float,
    coherence_hint: float,
) -> Dict[str, float]:
    time_since_user = max(0.0, now - last_user_ts) if last_user_ts > 0 else 1e9
    time_since_external = max(0.0, now - last_external_ts) if last_external_ts > 0 else 1e9

    stimulation = clamp(boredom_level)
    social = clamp(max(0.0, (time_since_user - 12.0) / 120.0))
    coherence = clamp((0.45 * coherence_hint) + (0.22 * stress_level) + (0.18 * salience) + (0.12 if unresolved_pending else 0.0))
    continuity = clamp((0.20 if unresolved_pending else 0.0) + min(0.60, pending_age_s / 90.0) if unresolved_pending else 0.0)
    safety = clamp((0.80 * stress_level) + (0.10 * salience))
    novelty = clamp((0.65 * boredom_level) + (0.18 * min(1.0, time_since_external / 90.0)) + (0.12 * salience))
    maintenance = clamp((0.65 if sleeping and charging else 0.0) + (0.20 if sleeping else 0.0) + (0.08 if charging else 0.0))

    return {
        "stimulation": round(stimulation, 4),
        "social": round(social, 4),
        "coherence": round(coherence, 4),
        "continuity": round(continuity, 4),
        "safety": round(safety, 4),
        "salience": round(clamp(salience), 4),
        "novelty": round(novelty, 4),
        "maintenance": round(maintenance, 4),
    }


def _blend(prev: float, target: float, rate: float) -> float:
    rate = clamp(rate, 0.01, 0.95)
    return clamp(prev + ((target - prev) * rate))


def update_hormone_state(
    prev: Mapping[str, Any] | HormoneState | None,
    *,
    needs: Mapping[str, Any],
    ddna: Mapping[str, Any] | None,
    dt_s: float = 1.0,
    context: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    state = prev if isinstance(prev, HormoneState) else HormoneState.from_mapping(prev)
    mods = dict(DEFAULT_DDNA_MODULATORS)
    if isinstance(ddna, Mapping):
        for key in DDNA_MOD_KEYS:
            if key in ddna:
                mods[key] = clamp(safe_float(ddna.get(key), mods[key]), 0.20, 2.00)

    ctx = dict(context or {})
    step = clamp(max(0.1, safe_float(dt_s, 1.0)) / 3.0, 0.05, 1.50)
    response_rate = clamp((0.16 + (0.18 * mods["volatility"])) * step, 0.05, 0.75)

    n = merge_need_maps(needs)
    blocked = clamp(safe_float(ctx.get("blocked", 0.0), 0.0))
    resolution = clamp(safe_float(ctx.get("resolution", 0.0), 0.0))
    interruption_cost = clamp(safe_float(ctx.get("interruption_cost", 0.0), 0.0))
    direct_address = clamp(safe_float(ctx.get("direct_address", 0.0), 0.0))

    arousal_target = clamp(
        (0.38 * n["stimulation"])
        + (0.22 * n["salience"])
        + (0.16 * n["social"])
        + (0.18 * n["maintenance"])
        + (0.10 * state.inquiry)
        - (0.16 * state.settling)
    )
    inquiry_target = clamp(
        (0.42 * n["coherence"])
        + (0.25 * n["continuity"])
        + (0.18 * n["novelty"])
        + (0.10 * n["salience"])
        + (0.08 * direct_address)
    )
    affiliation_target = clamp(
        (0.44 * n["social"])
        + (0.18 * n["continuity"])
        + (0.10 * direct_address)
        + (0.10 * resolution)
        - (0.10 * n["safety"])
    )
    caution_target = clamp(
        (0.58 * n["safety"])
        + (0.12 * interruption_cost)
        + (0.10 * blocked)
        + (0.08 * (1.0 - state.settling))
    )
    frustration_target = clamp(
        (0.40 * blocked)
        + (0.28 * n["coherence"])
        + (0.20 * n["continuity"])
        + (0.08 * interruption_cost)
        - (0.16 * resolution)
    )
    settling_target = clamp(
        (0.55 * resolution)
        + (0.22 * (1.0 - n["safety"]))
        + (0.16 * (1.0 - n["coherence"]))
        - (0.12 * blocked)
        - (0.10 * n["stimulation"])
    )
    persistence_target = clamp(
        (0.34 * n["continuity"])
        + (0.30 * n["coherence"])
        + (0.14 * n["salience"])
        + (0.10 * direct_address)
        - (0.18 * n["safety"])
        - (0.10 * interruption_cost)
    )
    continuity_target = clamp(
        (0.62 * n["continuity"])
        + (0.20 * n["coherence"])
        + (0.08 * direct_address)
    )

    next_state = HormoneState(
        arousal=_blend(state.arousal, arousal_target * mods["arousal_gain"], response_rate),
        inquiry=_blend(state.inquiry, inquiry_target * mods["inquiry_gain"], response_rate),
        affiliation=_blend(state.affiliation, affiliation_target * mods["affiliation_gain"], response_rate),
        caution=_blend(state.caution, caution_target * mods["caution_gain"], response_rate),
        frustration=_blend(state.frustration, frustration_target * mods["frustration_gain"], response_rate),
        settling=_blend(state.settling, settling_target * mods["settling_gain"], response_rate),
        persistence=_blend(state.persistence, persistence_target * mods["persistence_gain"], response_rate),
        continuity=_blend(state.continuity, continuity_target * mods["continuity_gain"], response_rate),
    )

    # Coupling pass: keep the field feeling like weather, not independent sliders.
    next_state.arousal = clamp(next_state.arousal + (0.08 * next_state.inquiry) + (0.06 * next_state.frustration) - (0.10 * next_state.settling))
    next_state.inquiry = clamp(next_state.inquiry + (0.08 * next_state.continuity) - (0.05 * next_state.caution))
    next_state.affiliation = clamp(next_state.affiliation + (0.05 * direct_address) - (0.06 * next_state.caution))
    next_state.caution = clamp(next_state.caution + (0.08 * next_state.frustration) - (0.04 * next_state.settling))
    next_state.settling = clamp(next_state.settling + (0.05 * resolution) - (0.10 * next_state.frustration))
    next_state.persistence = clamp(next_state.persistence + (0.06 * next_state.continuity) - (0.05 * next_state.caution))

    return next_state.to_dict()


def derive_want_vector(
    hormones: Mapping[str, Any],
    *,
    needs: Mapping[str, Any] | None = None,
    ddna: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    h = HormoneState.from_mapping(hormones)
    n = merge_need_maps(needs)
    mods = dict(DEFAULT_DDNA_MODULATORS)
    if isinstance(ddna, Mapping):
        for key in DDNA_MOD_KEYS:
            if key in ddna:
                mods[key] = clamp(safe_float(ddna.get(key), mods[key]), 0.20, 2.00)

    expression_bias = mods["expression_bias"]
    restraint_bias = mods["restraint_bias"]

    inquire = clamp((0.46 * h.inquiry) + (0.22 * h.continuity) + (0.12 * n["coherence"]))
    connect = clamp((0.40 * h.affiliation) + (0.20 * n["social"]) + (0.12 * h.continuity))
    observe = clamp((0.30 * h.arousal) + (0.24 * n["novelty"]) + (0.16 * n["salience"]))
    settle = clamp((0.42 * h.settling) + (0.18 * n["maintenance"]) - (0.16 * h.frustration))
    withhold = clamp((0.38 * h.caution) + (0.18 * restraint_bias / 2.0) + (0.12 * n["safety"]))
    externalize = clamp(
        ((0.28 * h.inquiry) + (0.24 * h.affiliation) + (0.18 * h.continuity) + (0.10 * n["social"])) * expression_bias
        - ((0.16 * h.caution) + (0.14 * restraint_bias))
    )

    return {
        "inquire": round(inquire, 4),
        "connect": round(connect, 4),
        "observe": round(observe, 4),
        "settle": round(settle, 4),
        "withhold": round(withhold, 4),
        "externalize": round(externalize, 4),
    }
