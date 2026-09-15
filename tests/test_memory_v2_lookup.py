import asyncio

from microbrain.memory.answer_composer import compose_answer_from_memory
from microbrain.memory.memory_v2_lookup import MemoryV2Lookup
from microbrain.memory.segmented_language_memory import SegmentedLanguageMemory


class _MemStoreRef:
    def __init__(self, memdir):
        self.memdir = str(memdir)


class _Ctx:
    def __init__(self, memdir):
        self.kv = {"memory:store": _MemStoreRef(memdir)}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value


def test_memory_v2_lookup_answers_simple_definition(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)
    store.observe_text("apple is a fruit", role="user", source="test")

    answer = MemoryV2Lookup(tmp_path).answer_question("What is apple?")

    assert answer["ok"] is True
    assert answer["text"] == "Apple is fruit."
    assert answer["selected_cell_ids"]


def test_memory_v2_lookup_returns_trainer_guidance(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)
    store.apply_trainer_correction(
        correction_text="thank you",
        context_text="I did something for you",
        bad_utterance="you're welcome",
    )

    guidance = MemoryV2Lookup(tmp_path).syntax_guidance("I did something for you")

    assert guidance["preferred_replies"][0]["text"] == "thank you"
    assert "you're welcome" in guidance["avoid_replies"]


def test_answer_composer_uses_memory_v2_before_v1(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)
    store.observe_text("apple is a fruit", role="user", source="test")

    async def run():
        return await compose_answer_from_memory(_Ctx(tmp_path), "What is apple?")

    answer = asyncio.run(run())

    assert answer["ok"] is True
    assert answer["reason"] == "memory_v2_phrase_slot"
    assert answer["text"] == "Apple is fruit."
