import asyncio

from microbrain.neurons.capability_circulation_neuron import CapabilityCirculationNeuron
from microbrain.neurons.thought_momentum_neuron import ThoughtMomentumNeuron
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.utils.heartbeat_stream import (
    heartbeat_meta,
    service_tick_meta,
    service_tick_payload,
    service_topic,
)


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


def test_legacy_clock_subscription_normalizes_but_body_pulse_never_learns_or_traces():
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
        # Legacy subscriptions normalize to the one canonical heartbeat topic.
        assert neuron.subscribed_topics == ("body/heartbeat",)

        event = Event(
            topic="body/heartbeat",
            payload={"tick": 1, "ts": 1.0},
            source="system_clock",
            meta=heartbeat_meta(),
        )
        outputs = list(await neuron.handle_event(event, ctx))
        assert len(outputs) == 1
        assert neuron.get_hebbian_weight("body/heartbeat") == 0.0
        # Body cadence is intentionally absent from cognitive activation history.
        assert neuron.get_activation_history() == ()
        # A meaningful output must not inherit the infrastructure correlation.
        assert outputs[0].correlation_id != event.correlation_id

    asyncio.run(run())



def test_body_cadence_is_independent_of_semantic_cooldown():
    async def run():
        ctx = FakeCtx()
        neuron = EchoOutputNeuron(
            NeuronConfig(
                name="cooldown_isolation",
                subscribed_topics=["body/heartbeat"],
                output_topics=["state/pulse"],
                cooldown_sec=60.0,
            )
        )
        neuron._last_fire_time = 1234.5
        outputs = list(await neuron.handle_event(
            Event(
                topic="body/heartbeat",
                payload={"tick": 1, "ts": 1.0},
                meta=heartbeat_meta(),
            ),
            ctx,
        ))
        assert len(outputs) == 1
        assert neuron._last_fire_time == 1234.5

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


def _service_event(target: str, tick: int = 1) -> Event:
    heartbeat = {
        "tick": tick,
        "epoch_s": 1000.0 + (tick * 0.05),
        "monotonic_s": 10.0 + (tick * 0.05),
        "delta_s": 0.05,
    }
    return Event(
        topic=service_topic(target),
        payload=service_tick_payload(heartbeat, target=target, mode="normal", divisor=1),
        source="body_adrenaline_scheduler_neuron",
        meta=service_tick_meta(target),
    )


def test_thought_momentum_uses_cognition_service_only_for_private_passive_decay():
    async def run():
        ctx = FakeCtx()
        neuron = ThoughtMomentumNeuron(
            NeuronConfig(
                name="thought_momentum_neuron",
                subscribed_topics=["percept/text", service_topic("cognition")],
                output_topics=["thought/momentum"],
            )
        )

        semantic = Event(topic="percept/text", payload={"text": "Where is the charger?"})
        semantic_outputs = list(await neuron.process(semantic, ctx))
        assert semantic_outputs
        assert ctx.kv["thought:momentum"]["last_event_topic"] == "percept/text"

        tick_outputs = list(await neuron.process(_service_event("cognition", 2), ctx))
        assert tick_outputs == []
        assert ctx.kv["thought:momentum"]["last_event_topic"] == "percept/text"
        assert ctx.kv["thought:momentum"]["last_update_reason"] == "passive_decay"

    asyncio.run(run())


def test_capability_service_does_not_publish_unchanged_state():
    async def run():
        ctx = FakeCtx()
        neuron = CapabilityCirculationNeuron(
            NeuronConfig(
                name="capability_circulation_neuron",
                subscribed_topics=[service_topic("capability")],
                output_topics=["capability/state", "thought/drawer_recheck"],
            )
        )
        service = _service_event("capability", 4)
        # First service initializes the dashboard/KV instrument privately.
        first_outputs = list(await neuron.process(service, ctx))
        assert first_outputs == []
        assert "capability:state" in ctx.kv

        # Once state is stable, repeated scheduler opportunities remain internal.
        second_outputs = list(await neuron.process(service, ctx))
        assert second_outputs == []

        # A real capability change still emits state + drawer recheck immediately.
        changed_outputs = list(await neuron.process(
            Event(topic="control/capability", payload={"component": "motion_available", "available": True, "ttl_s": 0}),
            ctx,
        ))
        assert {ev.topic for ev in changed_outputs} == {"capability/state", "thought/drawer_recheck"}

    asyncio.run(run())
