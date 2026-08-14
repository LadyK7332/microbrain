from __future__ import annotations

from microbrain.affect_curves import (
    AffectCurveConfig,
    apply_affect_pulse,
    decay_curve_state,
    summarize_curve_bucket,
)


def test_repeated_same_target_gets_dampened():
    cfg = AffectCurveConfig(
        name="user_approval",
        flow_threshold=10.0,
        curve_capacity=10.0,
        decay_half_life_s=1000.0,
        repeat_window_s=10.0,
        repeat_dampening=0.20,
    )
    first = apply_affect_pulse({}, cfg, now=100.0, incoming_strength=0.5, target_key="same")
    second = apply_affect_pulse(first.state, cfg, now=101.0, incoming_strength=0.5, target_key="same")

    assert first.effective_strength > second.effective_strength
    assert second.reason == "repeat_dampened"
    assert second.state["repeat_count"] == 1


def test_flow_threshold_creates_overload_and_limits_effective_strength():
    cfg = AffectCurveConfig(
        name="approval",
        flow_threshold=0.5,
        flow_window_s=10.0,
        curve_capacity=10.0,
        decay_half_life_s=1000.0,
    )
    first = apply_affect_pulse({}, cfg, now=1.0, incoming_strength=0.5, target_key="a")
    second = apply_affect_pulse(first.state, cfg, now=2.0, incoming_strength=0.5, target_key="b")

    assert first.flow_available > second.flow_available
    assert second.effective_strength == 0.0
    assert second.state["overload"] > 0.0
    assert second.reason in {"flow_limited", "overload_limited"}


def test_capacity_saturation_limits_future_pulses():
    cfg = AffectCurveConfig(
        name="small_capacity",
        flow_threshold=10.0,
        curve_capacity=0.5,
        decay_half_life_s=1000.0,
    )
    first = apply_affect_pulse({}, cfg, now=1.0, incoming_strength=0.5, target_key="a")
    second = apply_affect_pulse(first.state, cfg, now=2.0, incoming_strength=0.5, target_key="b")

    assert first.state["saturation"] == 1.0
    assert second.effective_strength == 0.0
    assert second.reason == "capacity_saturated"


def test_decay_restores_some_capacity():
    cfg = AffectCurveConfig(
        name="approval",
        flow_threshold=10.0,
        curve_capacity=1.0,
        decay_half_life_s=10.0,
    )
    first = apply_affect_pulse({}, cfg, now=0.0, incoming_strength=1.0, target_key="a")
    decayed = decay_curve_state(first.state, cfg, now=10.0)

    assert 0.45 <= decayed["level"] <= 0.55
    assert decayed["saturation"] < 1.0


def test_summary_is_small_and_stable():
    cfg = AffectCurveConfig(name="approval")
    first = apply_affect_pulse({}, cfg, now=5.0, incoming_strength=0.3, target_key="target-x")
    summary = summarize_curve_bucket({"approval": first.state})

    assert set(summary["approval"]) == {
        "level",
        "saturation",
        "overload",
        "flow_used",
        "repeat_count",
        "last_target",
    }
    assert summary["approval"]["last_target"] == "target-x"
