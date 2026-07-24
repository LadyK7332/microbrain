import asyncio

from microbrain.neurons.capability_circulation_neuron import CapabilityCirculationNeuron
from microbrain.neurons.thought_momentum_neuron import ThoughtMomentumNeuron
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig


class FakeCtx:
    def __init__(self):
        self.kv = {}
        self.debug = []

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        self.debug.append((args, kwargs))

    async def log_info(self, *args, **kwargs):
        pass

    async def log_warn(self, *args, **kwargs):
        pass

    async def log_error(self, *args, **kwargs):
        pass


class EchoOutputNeuron(BaseNeuron):
    async def process(self, event, ctx):
        return [Event(topic="state/pulse", payload={"seen": event.topic}, source=self.name)]


def test_clock_tick_never_earns_hebbian_weight_even_when_neuron_outputs():
    async def run():
        ctx = FakeCtx()
        neuron = EchoOutputNeuron(
            NeuronConfig(
                name="clock_test",
                subscribed_topics=["clock/tick"],
                output_topics=["state/pulse"],
                hebbian_learning_rate=0.5,
            )
        )
        event = Event(
            topic="clock/tick",
            payload={"ts": 1.0},
            source="system_clock",
            meta={
                "event_class": "infrastructure",
                "semantic_input": False,
                "reinforcement_eligible": False,
            },
        )
        outputs = list(await neuron.handle_event(event, ctx))
        assert len(outputs) == 1
        assert neuron.get_hebbian_weight("clock/tick") == 0.0
        record = neuron.get_activation_history()[-1]
        assert record.hebbian_context == ""
        assert record.hebbian_weight_after == 0.0

    asyncio.run(run())


def test_reinforcement_ineligible_semantic_event_does_not_gain_weight():
    async def run():
        ctx = FakeCtx()
        neuron = EchoOutputNeuron(
            NeuronConfig(
                name="eligibility_test",
                subscribed_topics=["percept/text"],
                output_topics=["state/pulse"],
                hebbian_learning_rate=0.5,
            )
        )
        event = Event(
            topic="percept/text",
            payload={"text": "system generated status"},
            meta={"reinforcement_eligible": False},
        )
        await neuron.handle_event(event, ctx)
        assert neuron.get_hebbian_weight("percept/text") == 0.0

    asyncio.run(run())


def test_thought_momentum_uses_clock_only_for_passive_decay():
    async def run():
        ctx = FakeCtx()
        neuron = ThoughtMomentumNeuron(
            NeuronConfig(
                name="thought_momentum_neuron",
                subscribed_topics=["percept/text", "clock/tick"],
                output_topics=["thought/momentum"],
            )
        )

        semantic = Event(topic="percept/text", payload={"text": "Where is the charger?"})
        semantic_outputs = list(await neuron.process(semantic, ctx))
        assert semantic_outputs
        assert ctx.kv["thought:momentum"]["last_event_topic"] == "percept/text"

        tick = Event(
            topic="clock/tick",
            payload={"ts": 2.0},
            source="system_clock",
            meta={"event_class": "infrastructure", "semantic_input": False},
        )
        tick_outputs = list(await neuron.process(tick, ctx))
        assert tick_outputs == []
        assert ctx.kv["thought:momentum"]["last_event_topic"] == "percept/text"
        assert ctx.kv["thought:momentum"]["last_update_reason"] == "passive_decay"

    asyncio.run(run())


def test_capability_clock_tick_does_not_publish_unchanged_state():
    async def run():
        ctx = FakeCtx()
        neuron = CapabilityCirculationNeuron(
            NeuronConfig(
                name="capability_circulation_neuron",
                subscribed_topics=["clock/tick"],
                output_topics=["capability/state", "thought/drawer_recheck"],
            )
        )
        tick = Event(
            topic="clock/tick",
            payload={"ts": 1.0},
            source="system_clock",
            meta={"event_class": "infrastructure", "semantic_input": False},
        )
        # First tick is allowed to publish because it initializes capability state.
        first_outputs = list(await neuron.process(tick, ctx))
        assert first_outputs
        assert "capability:state" in ctx.kv

        # Once state is stable, repeated scheduler pulses remain internal.
        second_outputs = list(await neuron.process(tick, ctx))
        assert second_outputs == []

    asyncio.run(run())
