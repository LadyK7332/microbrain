import json

from microbrain.memory.segmented_language_memory import SegmentedLanguageMemory


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_memory_v2_observe_text_builds_token_edges_and_phrase(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)

    result = store.observe_text("Good morning", role="user", source="test")

    assert result["ok"] is True
    assert result["route"] == "surface/social"
    assert result["expected_response_type"] == "greeting_return"
    assert (tmp_path / "memory_v2" / "tok_graph" / "tokens.jsonl").exists()
    assert (tmp_path / "memory_v2" / "tok_graph" / "adjacency.jsonl").exists()
    assert (tmp_path / "memory_v2" / "tok_graph" / "phrase_patterns.jsonl").exists()

    tokens = _jsonl(tmp_path / "memory_v2" / "tok_graph" / "tokens.jsonl")
    phrases = _jsonl(tmp_path / "memory_v2" / "tok_graph" / "phrase_patterns.jsonl")
    assert {row["text"] for row in tokens} == {"good", "morning"}
    assert phrases[0]["surface"] == "good morning"
    assert phrases[0]["weak_candidate"] is False


def test_memory_v2_trainer_correction_is_strong_and_separate(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)

    update = store.apply_trainer_correction(
        correction_text="thank you",
        context_text="I did something for you",
        bad_utterance="you're welcome",
        need="gratitude",
        style="direct",
    )

    corrections = _jsonl(tmp_path / "memory_v2" / "teaching_memory" / "corrections.jsonl")
    assert update["desired"] == "thank you"
    assert corrections[0]["desired"] == "thank you"
    assert corrections[0]["confidence"] == 0.95
    assert corrections[0]["avoid_surface"] == "you're welcome"


def test_memory_v2_slearn_batch_writes_rule_and_reply_guidance(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)

    result = store.apply_slearn_rules([
        {
            "condition_text": "USER says moin",
            "condition_raw": "USER says \"moin\"",
            "rule_kind": "syntax_rule",
            "classifiers": ["social_greeting"],
            "concept_classifiers": ["social_greeting"],
            "reply_text": "good morning",
            "source_name": "lesson.slearn",
            "source_line": 1,
        }
    ])

    assert result["accepted"] == 1
    rules = _jsonl(tmp_path / "memory_v2" / "pattern_memory" / "slearn_rules.jsonl")
    corrections = _jsonl(tmp_path / "memory_v2" / "teaching_memory" / "corrections.jsonl")
    assert rules[0]["reply_text"] == "good morning"
    assert rules[0]["correction_id"] == corrections[0]["correction_id"]
