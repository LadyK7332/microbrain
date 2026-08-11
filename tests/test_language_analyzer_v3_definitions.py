from microbrain.language_scaffold import analyze_english_structure, parse_text
from microbrain.memory.cross_modal_answer import compose_answer, gather_support
from microbrain.memory.mem_cell_store import MemCellStore


def test_what_is_question_preserves_definition_target():
    parsed = parse_text("What is prison?")
    frame = parsed.best_clause

    assert frame["clause_type"] == "question"
    assert frame["query_target"] == "definition"
    assert frame["definition_target"] == "prison"
    assert frame["object"] == "prison"

    learning = parsed.learning_frames[0]
    assert learning["frame_type"] == "definition_question"
    assert learning["query_target"] == "prison"
    assert learning["understanding_gaps"][0]["gap_type"] == "definition_missing"


def test_glorp_is_flib_becomes_soft_classification_and_gap():
    structure = analyze_english_structure("Glorp is a flib.")
    learning = structure["learning_frames"][0]

    assert learning["frame_type"] == "classification_claim"
    assert learning["subject"] == "glorp"
    assert learning["designation"] == "flib"
    assert learning["relation"] == "classified_as"
    gaps = {g["term"] for g in learning["understanding_gaps"]}
    assert {"glorp", "flib"} <= gaps


def test_definition_sentences_create_learning_frame_cells(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    result = store.ingest_text(
        text="Prison is a place where rule breakers are kept as punishment.",
        topic="trainer",
        role="user",
        transport_source="trainer",
        source="test",
        tier="now",
    )

    assert result["learning_frames"]
    cell = result["learning_frames"][0]
    assert cell["kind"] == "learning_frame"
    assert cell["meta"]["pattern_type"] == "definition_claim"
    assert cell["meta"]["slots"]["subject"] == "prison"
    assert cell["meta"]["slots"]["designation"] == "place"
    assert "rule breakers" in cell["meta"]["slots"]["definition"]
    assert cell["meta"]["creates_prebuilt_answer"] is False


def test_what_is_answer_composes_from_learned_definition_frame(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    store.ingest_text(
        text="Prison is a place where rule breakers are kept as punishment.",
        topic="trainer",
        role="user",
        transport_source="trainer",
        source="test",
        tier="now",
    )

    bundle = gather_support(query_text="What is prison?", mem_cell_store=store)
    answer, confidence, meta = compose_answer(bundle)

    assert answer == "Prison is a place where rule breakers are kept as punishment."
    assert confidence >= 0.80
    assert meta["used_learning_frame"] is True
    assert meta["used_prebuilt_answer"] is False


def test_deictic_designation_detects_multimodal_style_car_label():
    structure = analyze_english_structure("This is a car another kind of car.")
    learning = structure["learning_frames"][0]

    assert learning["frame_type"] == "designation_claim"
    assert learning["subject"] == "this"
    assert learning["designation"] == "car"
    assert learning["deictic"] is True
    assert learning["subtype_of"] == "car"
