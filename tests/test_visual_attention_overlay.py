from __future__ import annotations

from microbrain.vision_state import (
    has_visual_motion_salience,
    visual_object_uncertain,
    visual_ref_text,
)


def test_visual_ref_suffix_marks_uncertain_unknown_object() -> None:
    obj = {"track_id": "vobj:07", "label": "unknown", "status": "candidate", "confidence": 0.34}
    assert visual_object_uncertain(obj) is True
    assert visual_ref_text(obj) == "vobj:07?"


def test_visual_ref_leaves_confident_object_unmarked() -> None:
    obj = {"track_id": "vobj:03", "label": "face", "status": "identified", "confidence": 0.91}
    assert visual_object_uncertain(obj) is False
    assert visual_ref_text(obj) == "vobj:03"


def test_motion_salience_detects_motion_state_and_vector() -> None:
    assert has_visual_motion_salience({"track_id": "vobj:12", "motion_state": "motion_onset"}) is True
    assert has_visual_motion_salience({"track_id": "vobj:13", "motion": {"dx": 0.03, "dy": 0.0}}) is True
    assert has_visual_motion_salience({"track_id": "vobj:14", "motion": {"dx": 0.0, "dy": 0.0}}) is False
