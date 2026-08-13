from __future__ import annotations

from microbrain.affect_curves import (
    apply_curve_pulse,
    curve_spec,
    decay_curve_map,
    signed_feedback_curve,
)


def test_flow_threshold_limits_immediate_repeat_input():
    curves = {}
    now = 1000.0
    curves, first = apply_curve_pulse(curves, name="user_approval", signed_amount=0.8, now=now, target_key="same")
    curves, second = apply_curve_pulse(curves, name="user_approval", signed_amount=0.8, now=now + 0.1, target_key="same")

    assert first.effective > 0.0
    assert second.effective < first.effective
    assert second.flow_available < first.flow_available
    assert second.saturation >= first.saturation


def test_curve_capacity_saturates_repeated_same_target():
    curves = {}
    now = 2000.0
    effects = []
    for i in range(8):
        curves, result = signed_feedback_curve(curves, signed_strength=5.0, now=now + i, target_key="last_visible_action")
        effects.append(result["reward_delta"])

    assert effects[0] > effects[-1]
    assert curves["user_approval"]["saturation"] > 0.4
    assert curves["user_approval"]["repeat_count"] >= 3


def test_decay_reopens_capacity_and_flow_over_time():
    curves = {}
    now = 3000.0
    curves, _ = apply_curve_pulse(curves, name="user_approval", signed_amount=1.0, now=now, target_key="x")
    saturated = curves["user_approval"]
    decayed = decay_curve_map(curves, now=now + 60.0)

    assert decayed["user_approval"]["level"] < saturated["level"]
    assert decayed["user_approval"]["flow"] < saturated["flow"]
    assert decayed["user_approval"]["saturation"] < saturated["saturation"]


def test_negative_feedback_uses_correction_curve_not_approval_curve():
    curves, effect = signed_feedback_curve({}, signed_strength=-7.0, now=4000.0, target_key="bad_guess")

    assert "user_correction" in curves
    assert "user_approval" not in curves
    assert effect["reward_delta"] < 0.0
    assert effect["valence_delta"] < 0.0
    assert effect["salience_delta"] > 0.0


def test_curve_spec_override_keeps_positive_safety_bounds():
    spec = curve_spec("custom", {"flow_threshold": -1, "curve_capacity": 0, "decay_half_life_s": -2})

    assert spec.flow_threshold > 0.0
    assert spec.curve_capacity > 0.0
    assert spec.decay_half_life_s > 0.0
