from microbrain.language.response_obligation import build_response_obligation


def test_released_status_update_creates_acknowledgement_obligation():
    obligation = build_response_obligation(
        context={"input": {"text": "I am halfway done.", "correlation_id": "turn-1"}},
        hypothesis={
            "hypothesis_id": "hyp-1",
            "should_respond": True,
            "recommended_action": "acknowledge",
            "pattern_analysis": {"statement_kind": "status_update"},
        },
        trigger={"should_release": True, "recommended_action": "acknowledge"},
        correlation_id="fallback",
    )
    assert obligation is not None
    assert obligation["required_act"] == "acknowledge"
    assert obligation["required"] is True
    assert obligation["silence_allowed"] is False


def test_deliberate_silence_does_not_create_obligation():
    assert build_response_obligation(
        context={"input": {"text": "A plain statement."}},
        hypothesis={"should_respond": False, "recommended_action": "silence"},
        trigger={"should_release": False, "recommended_action": "silence"},
        correlation_id="turn-2",
    ) is None


def test_visual_binding_keeps_internal_ref_out_of_surface():
    obligation = build_response_obligation(
        context={
            "input": {"text": "What is that?", "correlation_id": "turn-3"},
            "attention": {"visual_ref": {"track_id": "vobj:07", "label_hint": "unknown"}},
        },
        hypothesis={
            "hypothesis_id": "hyp-3",
            "should_respond": True,
            "recommended_action": "clarify",
            "pattern_analysis": {"statement_kind": "question"},
        },
        trigger={"should_release": True, "recommended_action": "clarify"},
        correlation_id="turn-3",
    )
    assert obligation is not None
    assert obligation["target_ref"] == "vobj:07"
    assert obligation["target_surface"] == "that"
