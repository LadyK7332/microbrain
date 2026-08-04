import asyncio

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.native_responder_neuron import NativeResponderNeuron
from microbrain.neurons.syntax_learning_neuron import SyntaxLearningNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


def _syntax_learner() -> SyntaxLearningNeuron:
    return SyntaxLearningNeuron(
        NeuronConfig(
            name="syntax_learning_neuron",
            subscribed_topics=["control/slearn"],
            output_topics=["ui/status"],
        )
    )


def _responder() -> NativeResponderNeuron:
    return NativeResponderNeuron(
        NeuronConfig(
            name="native_responder_neuron",
            subscribed_topics=["context/request"],
            output_topics=["act/speech"],
        )
    )


def _store_rule(store: MemCellStore, note: str) -> dict:
    learner = _syntax_learner()
    parsed = learner._parse_teaching_note(note)
    assert parsed
    parsed.update(
        {
            "reinforce_weight": 3,
            "source_mode": "slearn",
            "source_name": "test.slearn",
            "source_path": "test.slearn",
            "source_line": 1,
        }
    )
    saved = learner._store_rule(store, parsed)
    parsed["saved"] = saved
    return parsed


def test_slearn_bound_slot_replies_with_captured_surface_text(tmp_path):
    store = MemCellStore(tmp_path)
    parsed = _store_rule(
        store,
        'IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"',
    )

    assert parsed["condition_slots"] == ["payload"]
    assert parsed["reply_slots"] == ["payload"]
    # Templated rules remain syntax rules; they do not create a literal
    # trainer-alignment utterance containing "{payload}".
    assert parsed["saved"] == 1

    responder = _responder()

    guidance = responder._syntax_guidance(store, "say haz")
    assert responder._preferred_rule_reply(guidance, warm=0.0) == "haz"

    guidance = responder._syntax_guidance(store, "say Rise of the Machine")
    assert responder._preferred_rule_reply(guidance, warm=0.0) == "Rise of the Machine"


def test_slearn_bound_slot_strips_wrapper_quotes(tmp_path):
    store = MemCellStore(tmp_path)
    _store_rule(
        store,
        'IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"',
    )

    responder = _responder()
    guidance = responder._syntax_guidance(store, 'say "haz"')
    assert responder._preferred_rule_reply(guidance, warm=0.0) == "haz"


def test_slearn_rejects_reply_slot_not_bound_by_condition():
    learner = _syntax_learner()
    parsed = learner._parse_teaching_note(
        'IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{other}"'
    )
    assert parsed == {}


def test_slearn_rejects_unanchored_catch_all_slot():
    learner = _syntax_learner()
    parsed = learner._parse_teaching_note(
        'IF USER says "{payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"'
    )
    assert parsed == {}


def test_slearn_template_does_not_match_unrelated_input(tmp_path):
    store = MemCellStore(tmp_path)
    _store_rule(
        store,
        'IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"',
    )

    responder = _responder()
    guidance = responder._syntax_guidance(store, "tell me about haz")
    assert responder._preferred_rule_reply(guidance, warm=0.0) == ""


class _FakeCtx:
    def __init__(self, initial=None):
        self.kv = dict(initial or {})
        self.emitted = []

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def emit(self, event):
        self.emitted.append(event)

    async def log_debug(self, *args, **kwargs):
        pass

    async def log_info(self, *args, **kwargs):
        pass

    async def log_warn(self, *args, **kwargs):
        pass

    async def log_error(self, *args, **kwargs):
        pass


def test_learned_bound_rule_owns_direct_response_without_say_hardcode(tmp_path):
    async def run():
        store = MemCellStore(tmp_path)
        _store_rule(
            store,
            'IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"',
        )
        ctx = _FakeCtx(
            {
                "memory:mem_cell_store": store,
                "llm:enabled": False,
                "drive:rosehip": {"direct_reply_floor": 0.30, "outward_scale": 1.0},
            }
        )
        responder = _responder()
        event = Event(
            topic="reason/request",
            payload={"text": "say haz", "channel": "textual", "source": "user"},
            correlation_id="corr-slearn-say",
        )

        outputs = list(await responder.process(event, ctx))
        assert len(outputs) == 1
        assert outputs[0].topic == "act/speech"
        assert outputs[0].payload["text"] == "haz"
        assert outputs[0].meta["shape"]["suppress"] is False

    asyncio.run(run())
