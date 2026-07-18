from microbrain.memory.mem_cell_store import MemCellStore


def test_internal_need_becomes_thought_template(tmp_path):
    store = MemCellStore(tmp_path)
    result = store.ingest_text(
        text="I need to charge soon",
        topic="percept/text",
        role="assistant",
        transport_source="test",
        source="test",
        tier="now",
    )

    thought_templates = result["thought_templates"]
    assert any(c["meta"]["pattern_type"] == "need_action" for c in thought_templates)
    need = next(c for c in thought_templates if c["meta"]["pattern_type"] == "need_action")
    assert need["meta"]["slots"]["need_type"] == "power_recovery"
    assert need["meta"]["slots"]["urgency"] == "soon"


def test_request_action_becomes_structured_tool(tmp_path):
    store = MemCellStore(tmp_path)
    result = store.ingest_text(
        text="Can you patch it please",
        topic="percept/text",
        role="user",
        transport_source="test",
        source="test",
        tier="now",
    )

    assert any(c["meta"]["tool_role"] == "listener_reference" for c in result["word_roles"])
    request = next(c for c in result["thought_templates"] if c["meta"]["pattern_type"] == "request_action")
    assert request["meta"]["slots"]["action"] == "patch"
    assert request["meta"]["slots"]["target"] == "it"


def test_preference_action_uses_words_as_tools(tmp_path):
    store = MemCellStore(tmp_path)
    result = store.ingest_text(
        text="Well, we like to visit old friends now and then.",
        topic="percept/text",
        role="user",
        transport_source="test",
        source="test",
        tier="now",
    )

    roles = {(c["meta"]["token"], c["meta"]["tool_role"]) for c in result["word_roles"]}
    assert ("we", "group_self_reference") in roles
    assert ("like", "preference_relation") in roles
    assert ("old", "attribute_modifier") in roles

    pref = next(c for c in result["thought_templates"] if c["meta"]["pattern_type"] == "preference_action")
    assert pref["meta"]["slots"]["action"] == "visit"
    assert pref["meta"]["slots"]["object"] == "old friends"
