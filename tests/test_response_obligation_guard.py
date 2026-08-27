from microbrain.response_obligation_guard import guard_native_response


def test_rejects_orphan_with_from_social_update():
    result = guard_native_response(
        user_text="I'm halfway done with wok!",
        proposed_reply="with",
    )
    assert result.action == "repair"
    assert result.text == "Nice, halfway done with work."
    assert result.turn_type == "social_progress_update"


def test_rejects_what_is_fragment_for_work_question():
    result = guard_native_response(
        user_text="what is my work",
        proposed_reply="what is",
    )
    assert result.action == "repair"
    assert result.text == "work?"
    assert result.subject == "work"


def test_rejects_label_without_question_surface():
    result = guard_native_response(
        user_text="question, subject?",
        proposed_reply="question",
    )
    assert result.action == "repair"
    assert result.text == "subject?"


def test_low_confidence_label_becomes_question_handle():
    result = guard_native_response(
        user_text="?",
        proposed_reply="question",
    )
    assert result.action == "repair"
    assert result.text == "question?"


def test_accepts_guess_what_followup():
    result = guard_native_response(
        user_text="Demi, guess what",
        proposed_reply="what",
    )
    assert result.action == "accept"
    assert result.text == "what"


def test_accepts_valid_fact_answer():
    result = guard_native_response(
        user_text="What is my work?",
        proposed_reply="You work at EMS as a call center agent.",
    )
    assert result.action == "accept"
    assert result.text.startswith("You work at EMS")


def test_empty_direct_question_gets_subject_handle():
    result = guard_native_response(
        user_text="What is my work?",
        proposed_reply="",
    )
    assert result.action == "repair"
    assert result.text == "work?"


def test_empty_plain_statement_stays_silent():
    result = guard_native_response(
        user_text="I walked to the kitchen",
        proposed_reply="",
    )
    assert result.action == "drop"
    assert result.text == ""


def test_vobj_subject_surfaces_as_handle():
    result = guard_native_response(
        user_text="what is vobj:00007",
        proposed_reply="what is",
    )
    assert result.action == "repair"
    assert result.text == "vobj:00007?"


def test_accepts_existing_question_handle():
    result = guard_native_response(
        user_text="what is that",
        proposed_reply="vobj:00007?",
    )
    assert result.action == "accept"
    assert result.text == "vobj:00007?"
