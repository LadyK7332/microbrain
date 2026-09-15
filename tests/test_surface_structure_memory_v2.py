from microbrain.language.surface_structure_memory_v2 import (
    build_surface_candidate,
    build_surface_plan,
    infer_surface_pattern,
    merge_structure,
    normalize_structure_candidate,
    render_pattern,
    select_structure,
    structure_candidates_from_text,
)


def _question_structure():
    return normalize_structure_candidate(
        {
            "structure_id": "lstruct2:test",
            "structure_kind": "unknown_identity_question",
            "speech_act": "clarify",
            "surface_example": "What is that?",
        }
    )


def test_visual_handle_stays_internal_while_surface_uses_that():
    store = merge_structure({}, _question_structure())
    plan = build_surface_plan(
        {
            "obligation_id": "obligation:test",
            "required": True,
            "required_act": "clarify",
            "target_ref": "vobj:07",
            "target_surface": "that",
        },
        store,
    )
    assert plan["surface"] == "What is that?"
    assert plan["target_ref"] == "vobj:07"
    assert plan["surface_complete"] is True
    assert build_surface_candidate(plan)["obligation_id"] == "obligation:test"


def test_missing_target_does_not_render_double_question_placeholder():
    structure = _question_structure()
    assert select_structure({"required_act": "clarify"}, {"x": structure}, target_surface="", signal="") is None
    assert render_pattern("What is {target}?") == ""


def test_slotless_prose_cannot_be_selected_as_canned_reply():
    structure = normalize_structure_candidate({"surface_example": "Birds fly.", "speech_act": "clarify"})
    assert structure is not None
    assert select_structure({"required_act": "clarify"}, {"x": structure}, target_surface="that", signal="") is None


def test_it_and_there_need_explicit_target_slot():
    assert infer_surface_pattern("It is raining.") == "It is raining."
    assert infer_surface_pattern("It is unknown.", slots={"target": {}}) == "{target} is unknown."


def test_malformed_confidence_uses_safe_default():
    structure = normalize_structure_candidate({"surface_example": "What is that?", "confidence": "unknown"})
    assert structure is not None
    assert structure["confidence"] == 0.52


def test_reading_producer_filters_slotless_sentences():
    candidates = structure_candidates_from_text(
        "Birds fly. What is that? It is raining.", source="reading:test"
    )
    assert [item["surface_pattern"] for item in candidates] == ["What is {target}?"]
