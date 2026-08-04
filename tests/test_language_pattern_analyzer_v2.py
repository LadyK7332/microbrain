from microbrain.language_scaffold import analyze_english_structure, parse_text
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.objects.base_object import infer_grammar_roles


def _role(parsed, token: str):
    token = token.lower()
    return next(item for item in parsed.role_candidates if str(item.get("norm", "")).lower() == token)


def test_unknown_words_use_english_order_to_infer_subject_verb_object():
    parsed = parse_text("The dax snorp the blen.")

    assert _role(parsed, "dax")["best_role"] == "noun"
    assert _role(parsed, "snorp")["best_role"] == "verb"
    assert _role(parsed, "blen")["best_role"] == "noun"

    frame = parsed.best_clause
    assert frame["clause_type"] == "declarative"
    assert frame["subject"] == "dax"
    assert frame["action"] == "snorp"
    assert frame["object"] == "blen"
    assert frame["confidence"] >= 0.70


def test_imperative_exposes_implied_listener_and_payload():
    parsed = parse_text("Say Rise of the Machine")
    frame = parsed.best_clause

    assert frame["clause_type"] == "imperative"
    assert frame["subject"] == "you"
    assert frame["subject_implied"] is True
    assert frame["action"] == "say"
    assert frame["object_text"] == "Rise of the Machine"


def test_wh_question_normalizes_missing_object_slot():
    parsed = parse_text("What did the fox eat?")
    frame = parsed.best_clause

    assert frame["clause_type"] == "question"
    assert frame["subject"] == "fox"
    assert frame["action"] == "eat"
    assert frame["object"] == ""
    assert frame["query_target"] == "object"


def test_passive_and_active_share_same_semantic_relation():
    passive = parse_text("The door was opened by Haz.").best_clause
    active = parse_text("Haz opened the door.").best_clause

    assert passive["voice"] == "passive"
    assert passive["agent"] == "haz"
    assert passive["patient"] == "door"
    assert passive["action"] == "open"

    assert active["subject"] == "haz"
    assert active["object"] == "door"
    assert active["action"] == "open"


def test_ambiguous_sentence_keeps_multiple_parse_candidates():
    parsed = parse_text("I saw her duck.")
    candidates = parsed.clause_candidates

    assert len(candidates) >= 2
    assert any(c.get("ambiguity") == "possessive_noun" for c in candidates)
    assert any(c.get("ambiguity") == "object_plus_embedded_action" for c in candidates)


def test_reading_structure_keeps_sentence_local_frames():
    structure = analyze_english_structure("The fox entered the barn. It looked around.")

    assert len(structure["sentence_structures"]) == 2
    assert structure["sentence_structures"][0]["best_clause"]["subject"] == "fox"
    assert structure["sentence_structures"][1]["best_clause"]["subject"] == "it"


def test_memcell_ingest_persists_clause_frame_and_unknown_verb_role(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    result = store.ingest_text(
        text="The dax snorp the blen.",
        topic="percept/reading",
        role="assistant",
        transport_source="reading",
        source="test.txt",
        tier="now",
    )

    roles = {(c["meta"]["token"], c["meta"]["tool_role"]) for c in result["word_roles"]}
    assert ("snorp", "action_or_process") in roles

    frame = result["clause_frames"][0]
    assert frame["kind"] == "clause_frame"
    assert frame["anchor"]["ref"] == "dax snorp blen"
    assert frame["meta"]["epistemic_status"] == "parsed_structure_candidate"

    relation = next(c for c in result["thought_templates"] if c["meta"]["pattern_type"] == "action_relation")
    assert relation["meta"]["slots"]["subject"] == "dax"
    assert relation["meta"]["slots"]["action"] == "snorp"
    assert relation["meta"]["slots"]["object"] == "blen"


def test_base_object_grammar_roles_use_same_structure_engine():
    roles = infer_grammar_roles("The dax snorp the blen.")

    assert "dax" in roles["noun_like"]
    assert "snorp" in roles["verb_like"]
    assert roles["best_clause"]["action"] == "snorp"
