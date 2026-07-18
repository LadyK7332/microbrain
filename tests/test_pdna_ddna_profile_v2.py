from microbrain.hormone import derive_ddna_modulators
from microbrain.pdna.core import PDNAProfile


def test_pdna_v2_extra_sections_survive_save_roundtrip():
    raw = {
        "name": "Demi",
        "warmth": 0.8,
        "playfulness": 0.7,
        "ddna_mutators": {"playfulness": {"novelty_gain": 1.18}},
        "affect_model": {"decay": {"dopamine_decay_per_second": 0.2}},
        "wans": {"preferred_routes": {"learning": ["compare_attempt_to_feedback"]}},
    }
    profile = PDNAProfile.from_dict(raw)
    out = profile.to_dict()
    assert out["name"] == "Demi"
    assert out["ddna_mutators"]["playfulness"]["novelty_gain"] == 1.18
    assert out["affect_model"]["decay"]["dopamine_decay_per_second"] == 0.2
    assert out["wans"]["preferred_routes"]["learning"] == ["compare_attempt_to_feedback"]


def test_ddna_mutators_press_into_metabolic_mods():
    profile = PDNAProfile.from_dict(
        {
            "name": "Demi",
            "warmth": 0.8,
            "playfulness": 0.7,
            "safety_orientation": 0.95,
            "support_level": 0.8,
            "ddna_mutators": {
                "warmth": {"social_reward_gain": 1.15},
                "playfulness": {"novelty_gain": 1.18, "boredom_growth_gain": 1.08},
                "safety_orientation": {"risk_salience_gain": 1.4, "action_gate_strictness": 1.35},
                "support_level": {"trainer_alignment_gain": 1.12},
            },
        }
    )
    base_mods = derive_ddna_modulators(PDNAProfile.from_dict({"name": "Demi", "warmth": 0.8, "playfulness": 0.7, "safety_orientation": 0.95, "support_level": 0.8}))
    mods = derive_ddna_modulators(profile)
    assert mods["reward_gain"] > 1.0
    assert mods["salience_gain"] > 1.0
    assert mods["trainer_alignment_gain"] == 1.12
    assert mods["boredom_growth_gain"] == 1.08
    assert mods["novelty_gain"] > base_mods["novelty_gain"]
