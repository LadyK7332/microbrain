import asyncio

from microbrain.language.language_build_checker import (
    LanguageBuildChecker,
    candidate_was_rejected,
    check_language_candidate,
    clear_language_failures,
)
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


def test_language_build_checker_accepts_known_phrase_path(tmp_path):
    store = SegmentedLanguageMemory(tmp_path)
    store.observe_text("good morning", role="assistant", source="test", strength="trainer")

    result = LanguageBuildChecker(tmp_path).check("good morning")

    assert result["ok"] is True
    assert result["plan"]["phrase_support"]["exact"] is True


def test_language_build_checker_blocks_repeated_token_loop(tmp_path):
    result = LanguageBuildChecker(tmp_path).check("open open open open")

    assert result["blocked"] is True
    assert "token_loop" in result["flags"]


def test_language_build_temp_reject_bucket_clears_after_success(tmp_path):
    async def run():
        ctx = _Ctx(tmp_path)
        failed = await check_language_candidate(ctx, "open open open open", lookup_text="good morning")
        assert failed["blocked"] is True
        assert await candidate_was_rejected(ctx, "open open open open", lookup_text="good morning")

        await clear_language_failures(ctx, lookup_text="good morning")
        assert not await candidate_was_rejected(ctx, "open open open open", lookup_text="good morning")

    asyncio.run(run())
