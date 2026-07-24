import asyncio
from pathlib import Path

from microbrain.neurons.desire_trigger_neuron import DesireTriggerNeuron
from microbrain.neurons.hypothesis_engine_neuron import HypothesisEngineNeuron
from microbrain.neurons.interaction_release_vector_neuron import InteractionReleaseVectorNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self, initial=None):
        self.kv = dict(initial or {})

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass

    async def log_info(self, *args, **kwargs):
        pass

    async def log_warn(self, *args, **kwargs):
        pass

    async def log_error(self, *args, **kwargs):
        pass


def make_interaction_neuron():
    return InteractionReleaseVectorNeuron(
        NeuronConfig(
            name="interaction_release_vector_neuron",
            subscribed_topics=["clock/tick", "percept/text"],
            output_topics=["drive/interaction_request", "speech/reason"],
        )
    )


def make_desire_neuron():
    return DesireTriggerNeuron(
        NeuronConfig(
            name="desire_trigger_neuron",
            subscribed_topics=["hypothesis/ready"],
            output_topics=["hypothesis/action_committed", "release/request"],
        )
    )


def make_hypothesis_neuron():
    return HypothesisEngineNeuron(
        NeuronConfig(
            name="hypothesis_engine_neuron",
            subscribed_topics=["context/built"],
            output_topics=["pattern/analysis", "hypothesis/ready"],
        )
    )


def hypothesis_ready_event(statement_kind="status_update"):
    hypothesis = {
        "hypothesis_id": "hyp-test-1",
        "response_demand": 0.70,
        "expected_usefulness": 0.70,
        "silence_score": 0.15,
        "recommended_action": "acknowledge",
        "should_respond": True,
        "pattern_analysis": {
            "continuity": 0.50,
            "risk": 0.0,
            "uncertainty": 0.20,
            "statement_kind": statement_kind,
        },
    }
    context = {
        "input": {
            "text": "Progress update: the ingest is still moving.",
            "source": "user",
            "channel": "textual",
        },
        "cues": {},
        "constraints": {},
        "drives": {},
        "association_meta": {},
    }
    return Event(
        topic="hypothesis/ready",
        payload={"context": context, "hypothesis": hypothesis},
        correlation_id="corr-hypothesis-test",
    )


def test_external_turn_has_one_response_owner():
    async def run():
        ctx = FakeCtx()
        neuron = make_interaction_neuron()
        event = Event(
            topic="percept/text",
            payload={
                "text": "good morning",
                "source": "user",
                "channel": "textual",
            },
            correlation_id="corr-good-morning",
        )

        outputs = list(await neuron.process(event, ctx))
        topics = [item.topic for item in outputs]

        assert "drive/interaction_request" in topics
        assert "speech/reason" not in topics

        ownership = ctx.kv["drive:interaction:last_response_ownership"]
        assert ownership["owner"] == "hypothesis"
        assert ownership["hypothesis_owned"] is True
        assert ownership["outward_speech_allowed"] is False

        request = next(item for item in outputs if item.topic == "drive/interaction_request")
        assert request.payload["response_owner"] == "hypothesis"
        assert request.payload["outward_speech_allowed"] is False

    asyncio.run(run())


def test_internal_interaction_need_can_still_use_need_speech_route():
    async def run():
        ctx = FakeCtx(
            {
                "initiative:last": {
                    "pending_text": "hello",
                    "talk_pressure": 1.0,
                    "think_pressure": 0.4,
                    "pending_age_s": 30.0,
                    "clarify_ready": False,
                    "interruption_cost": 0.0,
                },
                "neuron:InitiativeThresholdNeuron:initiative_threshold_neuron:initiative_state": {
                    "pending_flags": {
                        "has_question": False,
                        "has_response_request": False,
                        "has_error_language": False,
                        "short_fragment": True,
                        "clarify_ready": False,
                    },
                    "pending_answered": False,
                    "clarify_said": False,
                },
            }
        )
        neuron = make_interaction_neuron()

        outputs = list(await neuron.process(Event(topic="clock/tick", payload={}), ctx))
        topics = [item.topic for item in outputs]

        assert "speech/reason" in topics
        ownership = ctx.kv["drive:interaction:last_response_ownership"]
        assert ownership["owner"] == "interaction_release_vector"
        assert ownership["outward_speech_allowed"] is True

    asyncio.run(run())


def test_ddna_release_threshold_is_bounded_and_inspectable():
    async def resolve(ddna):
        ctx = FakeCtx({"drive:ddna_modulators": ddna})
        neuron = make_desire_neuron()
        await neuron.process(hypothesis_ready_event(), ctx)
        return ctx.kv["hypothesis:release_tuning"]

    neutral = asyncio.run(resolve({}))
    restrained = asyncio.run(
        resolve(
            {
                "restraint_bias": 2.0,
                "action_gate_strictness": 2.0,
                "expression_threshold_gain": 2.0,
                "expression_bias": 0.5,
                "expression_activation_gain": 0.5,
                "social_gain": 0.5,
            }
        )
    )
    expressive = asyncio.run(
        resolve(
            {
                "restraint_bias": 0.5,
                "action_gate_strictness": 0.5,
                "expression_threshold_gain": 0.5,
                "expression_bias": 2.0,
                "expression_activation_gain": 2.0,
                "social_gain": 2.0,
            }
        )
    )

    assert restrained["effective"] > neutral["effective"]
    assert expressive["effective"] < neutral["effective"]
    assert restrained["ddna_offset_applied"] == 0.08
    assert expressive["ddna_offset_applied"] == -0.08
    assert restrained["minimum"] <= restrained["effective"] <= restrained["maximum"]
    assert "ddna_inputs" in restrained


def test_ddna_bias_changes_action_preference_not_evidence():
    async def run():
        ctx = FakeCtx(
            {
                "drive:ddna_modulators": {
                    "expression_bias": 2.0,
                    "restraint_bias": 0.5,
                    "social_gain": 2.0,
                    "continuity_gain": 2.0,
                    "inquiry_gain": 1.5,
                    "curiosity_gain": 1.5,
                    "caution_gain": 1.0,
                    "action_gate_strictness": 0.5,
                    "expression_threshold_gain": 0.5,
                }
            }
        )
        neuron = make_hypothesis_neuron()
        original = [
            {"action": "acknowledge", "score": 0.50, "evidence_refs": ["cell:a"]},
            {"action": "silence", "score": 0.50, "evidence_refs": ["cell:b"]},
        ]

        adjusted, trace = await neuron._apply_ddna_action_bias(
            ctx,
            pattern_analysis={"statement_kind": "status_update"},
            candidates=original,
        )
        by_action = {item["action"]: item for item in adjusted}

        assert by_action["acknowledge"]["score"] > 0.50
        assert by_action["silence"]["score"] < 0.50
        assert by_action["acknowledge"]["evidence_refs"] == ["cell:a"]
        assert by_action["silence"]["evidence_refs"] == ["cell:b"]
        assert trace["bias_min"] == -0.08
        assert trace["bias_max"] == 0.08

    asyncio.run(run())


def test_patched_reasoning_modules_use_canonical_configuration_sections():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        "microbrain/neurons/interaction_release_vector_neuron.py",
        "microbrain/neurons/desire_trigger_neuron.py",
        "microbrain/neurons/hypothesis_engine_neuron.py",
        "microbrain/neurons/hypothesis_outcome_observer_neuron.py",
        "microbrain/neurons/hypothesis_memory_reinforcement_neuron.py",
        "microbrain/patterns/pattern_toolkit.py",
        "microbrain/memory/mem_cell_store.py",
    ]

    for relative_path in targets:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "# Behavioral tuning" in text, relative_path
        assert "# Required static constants" in text, relative_path
