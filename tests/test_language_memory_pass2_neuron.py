import asyncio
import json

from microbrain.neurons.language_memory_pass2_neuron import (
    LAST_PARSE_KV_KEY,
    LAST_TRAINER_KV_KEY,
    STORE_KV_KEY,
    LanguageMemoryPass2Neuron,
)
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class _MemStoreRef:
    def __init__(self, memdir):
        self.memdir = str(memdir)


class _Ctx:
    def __init__(self, memdir):
        self.kv = {"memory:store": _MemStoreRef(memdir)}
        self.warns = []

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_warn(self, msg, **kwargs):
        self.warns.append((msg, kwargs))


def _neuron():
    return LanguageMemoryPass2Neuron(
        NeuronConfig(
            name="language_memory_pass2_neuron",
            subscribed_topics=["percept/text", "act/speech", "control/trainer_correction"],
        )
    )


def test_language_pass2_neuron_observes_text_without_memcell_store(tmp_path):
    async def run():
        ctx = _Ctx(tmp_path)
        event = Event(
            topic="percept/text",
            payload={"text": "Good morning", "source": "user", "raw_meta": {"transport_source": "console"}},
            source="input_text_neuron",
        )
        outputs = list(await _neuron().process(event, ctx))
        return ctx, outputs

    ctx, outputs = asyncio.run(run())

    assert len(outputs) == 1
    assert outputs[0].topic == "language/pass2_parse"
    assert ctx.kv[LAST_PARSE_KV_KEY]["route"] == "surface/social"
    assert STORE_KV_KEY in ctx.kv
    assert "memory:mem_cell_store" not in ctx.kv
    token_path = tmp_path / "memory_v2" / "tok_graph" / "tokens.jsonl"
    assert token_path.exists()
    tokens = [json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()]
    assert {row["text"] for row in tokens} == {"good", "morning"}


def test_language_pass2_neuron_records_trainer_correction(tmp_path):
    async def run():
        ctx = _Ctx(tmp_path)
        event = Event(
            topic="control/trainer_correction",
            payload={
                "correction_text": "thank you",
                "target": {
                    "context_query": "I did something for you",
                    "utterance": "you're welcome",
                    "need": "gratitude",
                },
            },
            source="input_text_neuron",
        )
        outputs = list(await _neuron().process(event, ctx))
        return ctx, outputs

    ctx, outputs = asyncio.run(run())

    assert len(outputs) == 1
    assert outputs[0].topic == "language/pass2_trainer_update"
    assert ctx.kv[LAST_TRAINER_KV_KEY]["desired"] == "thank you"
    assert (tmp_path / "memory_v2" / "teaching_memory" / "corrections.jsonl").exists()
