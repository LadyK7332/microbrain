from __future__ import annotations

import time

from microbrain.memory.mem_cell_usage_appender import (
    USAGE_APPENDIX_KEY,
    attach_temporary_usage_appendix,
    classify_response_request_slots,
    merge_meta_with_usage_appendix,
    merge_usage_appendices,
)


def _cell(cell_id: str, kind: str, ref: str, **meta):
    return {
        "id": cell_id,
        "kind": kind,
        "anchor": {"kind": f"test/{kind}", "ref": ref, "norm": ref.lower()},
        "refs": [],
        "meta": dict(meta),
    }


def test_response_request_slots_preserve_compound_yes_explanation_shape():
    slots = classify_response_request_slots("Do you approve? And what do you think of it?")
    assert slots["compound_request"] is True
    assert slots["explanation_expected"] is True
    names = [slot["slot"] for slot in slots["slots"]]
    assert "approval_judgment" in names
    assert "evaluation_explanation" in names


def test_usage_appendix_merge_appends_without_overwriting():
    now = time.time()
    old = [{"id": "ua1", "created_ts": now - 10, "expires_ts": now + 100, "cell_kind": "token_anchor"}]
    new = [{"id": "ua2", "created_ts": now, "expires_ts": now + 100, "cell_kind": "word_role"}]
    merged = merge_usage_appendices(old, new, now_ts=now, limit=8)
    assert [row["id"] for row in merged] == ["ua2", "ua1"]


def test_usage_appendix_prunes_expired_atoms():
    now = time.time()
    rows = merge_usage_appendices(
        [{"id": "old", "created_ts": now - 9999, "expires_ts": now - 1}],
        [{"id": "fresh", "created_ts": now, "expires_ts": now + 60}],
        now_ts=now,
    )
    assert [row["id"] for row in rows] == ["fresh"]


def test_meta_merge_preserves_normal_meta_and_appends_usage_atoms():
    now = time.time()
    old = {"role": "user", USAGE_APPENDIX_KEY: [{"id": "ua1", "created_ts": now - 1, "expires_ts": now + 60, "cell_kind": "token_anchor"}]}
    new = {"pattern_type": "question_about", USAGE_APPENDIX_KEY: [{"id": "ua2", "created_ts": now, "expires_ts": now + 60, "cell_kind": "clause_frame"}]}
    merged = merge_meta_with_usage_appendix(old, new, now_ts=now)
    assert merged["role"] == "user"
    assert merged["pattern_type"] == "question_about"
    assert len(merged[USAGE_APPENDIX_KEY]) == 2
    assert merged["usage_appendix_state"]["promotes_directly_to_truth"] is False


def test_attach_temporary_usage_appendix_marks_word_context():
    utterance = _cell("u1", "utterance_anchor", "Can I paint my shell?")
    token_cells = [
        _cell("t1", "token_anchor", "can"),
        _cell("t2", "token_anchor", "i"),
        _cell("t3", "token_anchor", "paint"),
        _cell("t4", "token_anchor", "my"),
        _cell("t5", "token_anchor", "shell"),
    ]
    word_role_cells = [_cell("wr1", "word_role", "paint:action_or_process", token_index=2, tool_role="action_or_process", functional_role="relation_or_action_binding")]
    out = attach_temporary_usage_appendix(
        utterance=utterance,
        token_cells=token_cells,
        word_role_cells=word_role_cells,
        thought_template_cells=[],
        clause_frame_cells=[],
        learning_frame_cells=[],
        general_pattern_cells=[],
        linker_cells=[],
        text="Can I paint my shell?",
        role="user",
        topic="input/text",
        source="cli",
        structure={},
        meta={},
    )
    by_id = {row["id"]: row for row in out}
    assert "wr1" in by_id
    usage = by_id["wr1"]["meta"][USAGE_APPENDIX_KEY][0]
    assert usage["left_context"] == "i"
    assert usage["right_context"] == "my"
    assert usage["epistemic_status"] == "temporary_usage_evidence"
