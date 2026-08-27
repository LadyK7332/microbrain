from microbrain.language.surface_structure_memory import (
    STORE_KV_KEY,
    build_surface_candidate_from_plan,
    build_surface_plan_for_gap,
    infer_surface_pattern,
    merge_structure_into_store,
    normalize_structure_candidate,
    primitive_gap_surface,
)


def test_infers_target_slot_from_what_is_that_quote():
    pattern = infer_surface_pattern("What is that?", "unknown_identity_question")
    assert pattern == "What is {target}?"


def test_normalizes_structure_candidate_without_making_truth_claim():
    candidate = {
        "schema": "language.structure_candidate.v1",
        "structure_id": "lstruct:test",
        "structure_kind": "unknown_identity_question",
        "surface_example": "What is that?",
        "source_quote_id": "quote:abc",
        "slots": {"target": {"role": "unknown_target"}},
    }
    structure = normalize_structure_candidate(candidate)
    assert structure is not None
    assert structure["surface_pattern"] == "What is {target}?"
    assert structure["not_canned_response"] is True
    assert structure["truth_status"] == "structure_shape_not_answer_truth"
    assert structure["learned_from_context"]["source_quote_id"] == "quote:abc"


def test_gap_renders_learned_unknown_identity_question():
    structure = normalize_structure_candidate(
        {
            "structure_id": "lstruct:what",
            "structure_kind": "unknown_identity_question",
            "surface_example": "What is that?",
        }
    )
    store = merge_structure_into_store({}, structure)
    plan = build_surface_plan_for_gap(
        {"gap_kind": "object_identity_unknown", "target": "vobj:07"},
        store,
    )
    assert plan["surface"] == "What is vobj:07?"
    assert plan["surface_status"] == "constructed_from_learned_structure"
    assert plan["not_canned_response"] is True
    candidate = build_surface_candidate_from_plan(plan)
    assert candidate["surface"] == "What is vobj:07?"
    assert candidate["requires_review_by_mouth"] is True


def test_gap_without_learned_structure_uses_primitive_handle_question():
    plan = build_surface_plan_for_gap(
        {"gap_kind": "object_identity_unknown", "target": "vobj:99"},
        {},
    )
    assert plan["surface"] == "vobj:99?"
    assert plan["surface_status"] == "primitive_placeholder_no_learned_structure"


def test_i_do_not_know_structure_can_be_reused_when_learned():
    structure = normalize_structure_candidate(
        {
            "structure_id": "lstruct:dontknow",
            "structure_kind": "unknown_identity_statement",
            "surface_example": "I don't know what that is.",
        }
    )
    store = merge_structure_into_store({}, structure)
    plan = build_surface_plan_for_gap(
        {"gap_kind": "object_identity_unknown", "target_ref": "vobj:12"},
        store,
    )
    assert plan["surface"] == "I don't know what vobj:12 is."


def test_primitive_surface_prefers_target_then_signal_then_question_mark():
    assert primitive_gap_surface("vobj:07") == "vobj:07?"
    assert primitive_gap_surface("", signal="o.o") == "o.o?"
    assert primitive_gap_surface("") == "?"
