import asyncio
import tempfile

from microbrain.memory.mem_cell_composer import MemCellComposer
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.language_surface_obligation_neuron import LanguageSurfaceObligationNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_warn(self, *args, **kwargs):
        return None


def _neuron():
    return LanguageSurfaceObligationNeuron(
        NeuronConfig(name="language_surface_obligation_neuron", subscribed_topics=[])
    )


def test_neuron_learns_then_realizes_explicit_obligation():
    ctx = DummyCtx()
    neuron = _neuron()
    learned = list(
        asyncio.run(
            neuron.process(
                Event(
                    topic="language/structure_candidate",
                    payload={
                        "structure_id": "lstruct2:test",
                        "structure_kind": "unknown_identity_question",
                        "speech_act": "clarify",
                        "surface_example": "What is that?",
                    },
                ),
                ctx,
            )
        )
    )
    assert [event.topic for event in learned] == ["language/surface_structure_v2"]
    outputs = list(
        asyncio.run(
            neuron.process(
                Event(
                    topic="speech/response_obligation",
                    correlation_id="turn",
                    payload={
                        "obligation_id": "obligation:test",
                        "required": True,
                        "required_act": "clarify",
                        "target_ref": "vobj:07",
                        "target_surface": "that",
                    },
                ),
                ctx,
            )
        )
    )
    assert [event.topic for event in outputs] == ["language/surface_plan_v2", "language/surface_candidate"]
    assert outputs[-1].payload["surface"] == "What is that?"
    assert outputs[-1].payload["surface_complete"] is True


def test_required_ack_without_learned_surface_becomes_expression_gap():
    ctx = DummyCtx()
    outputs = list(
        asyncio.run(
            _neuron().process(
                Event(
                    topic="speech/response_obligation",
                    payload={
                        "obligation_id": "obligation:ack",
                        "required": True,
                        "required_act": "acknowledge",
                    },
                ),
                ctx,
            )
        )
    )
    assert [event.topic for event in outputs] == [
        "language/surface_plan_v2",
        "language/surface_candidate",
        "thought/self_question",
    ]
    assert outputs[1].payload["surface"] == "I follow."
    assert outputs[1].payload["surface_complete"] is True
    assert outputs[1].payload["surface_floor"] is True
    assert outputs[1].payload["store_in_memory"] is False
    assert outputs[2].payload["reason"] == "capability_surface_mismatch"


def test_floor_direct_say_command_discharges_without_learning():
    ctx = DummyCtx()
    outputs = list(
        asyncio.run(
            _neuron().process(
                Event(
                    topic="speech/response_obligation",
                    payload={
                        "obligation_id": "obligation:say",
                        "required": True,
                        "required_act": "acknowledge",
                        "source_text": "SAY good evening",
                    },
                ),
                ctx,
            )
        )
    )
    candidate = next(event for event in outputs if event.topic == "language/surface_candidate")
    assert candidate.payload["surface"] == "Good evening."
    assert candidate.payload["surface_status"] == "speech_floor_direct_say"
    assert candidate.payload["reinforcement_eligible"] is False


def test_structure_hydrates_after_composer_and_reinforces_on_delivery():
    async def scenario():
        with tempfile.TemporaryDirectory() as root:
            first = DummyCtx()
            first.kv["memory:mem_cell_store"] = MemCellStore(root)
            await _neuron().process(
                Event(
                    topic="language/structure_candidate",
                    payload={
                        "structure_id": "lstruct2:persist",
                        "structure_kind": "unknown_identity_question",
                        "speech_act": "clarify",
                        "surface_example": "What is that?",
                        "source": "reading:test",
                    },
                ),
                first,
            )
            MemCellComposer(root).compose_once(tiers=["now"])

            second = DummyCtx()
            second.kv["memory:mem_cell_store"] = MemCellStore(root)
            outputs = list(
                await _neuron().process(
                    Event(
                        topic="speech/response_obligation",
                        payload={
                            "obligation_id": "obligation:persist",
                            "required": True,
                            "required_act": "clarify",
                            "target_ref": "vobj:2",
                            "target_surface": "that",
                        },
                    ),
                    second,
                )
            )
            candidate = next(event for event in outputs if event.topic == "language/surface_candidate")
            assert candidate.payload["selected_structure_id"] == "lstruct2:persist"
            await _neuron().process(
                Event(topic="speech/delivered", payload={"selected_structure_id": "lstruct2:persist"}),
                second,
            )
            assert second.kv["language:surface_structures_v2"]["lstruct2:persist"]["use_count"] == 1

    asyncio.run(scenario())
