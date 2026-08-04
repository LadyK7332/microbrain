import asyncio

from microbrain.memory.filters import classify_event_for_memory
from microbrain.neurons.body_adrenaline_scheduler_neuron import (
    AROUSAL_STATE_TOPIC,
    HAZARD_TOPIC,
    PRIMARY_HEARTBEAT_TOPIC,
    SCENE_TOPIC,
    VISION_DELTA_TOPIC,
    BodyAdrenalineSchedulerNeuron,
)
from microbrain.neurons.thought_turn_arbitration_neuron import ThoughtTurnArbitrationNeuron
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import (
    HEARTBEAT_HZ,
    HEARTBEAT_INTERVAL_S,
    heartbeat_meta,
    heartbeat_payload,
    is_infrastructure_event,
    service_target,
    service_tick_meta,
    service_tick_payload,
    service_topic,
)


class FakeCtx:
    def __init__(self):
        self.kv = {}

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


class AliasHeartbeatNeuron(BaseNeuron):
    async def process(self, event, ctx):
        count = int(await ctx.get_kv("test:alias_count", 0) or 0) + 1
        await ctx.set_kv("test:alias_count", count)
        return [
            Event(
                topic="diagnostic/alias_seen",
                payload={"topic": event.topic, "count": count},
                source=self.name,
                correlation_id=event.correlation_id,
            )
        ]


class CountingServiceNeuron(BaseNeuron):
    async def process(self, event, ctx):
        target = service_target(event.topic)
        key = f"test:service_count:{target}"
        await ctx.set_kv(key, int(await ctx.get_kv(key, 0) or 0) + 1)
        return []


def _scheduler() -> BodyAdrenalineSchedulerNeuron:
    return BodyAdrenalineSchedulerNeuron(
        NeuronConfig(
            name="body_adrenaline_scheduler_neuron",
            subscribed_topics=[
                PRIMARY_HEARTBEAT_TOPIC,
                HAZARD_TOPIC,
                VISION_DELTA_TOPIC,
                SCENE_TOPIC,
            ],
            output_topics=[AROUSAL_STATE_TOPIC],
            priority=35,
        )
    )


def _heartbeat_event(tick: int) -> Event:
    return Event(
        topic=PRIMARY_HEARTBEAT_TOPIC,
        payload=heartbeat_payload(
            tick=tick,
            epoch_s=1000.0 + (tick * HEARTBEAT_INTERVAL_S),
            monotonic_s=10.0 + (tick * HEARTBEAT_INTERVAL_S),
            delta_s=HEARTBEAT_INTERVAL_S,
        ),
        source="mind.body_pacemaker",
        meta=heartbeat_meta(),
    )


def _service_event(target: str, tick: int = 1) -> Event:
    heartbeat = _heartbeat_event(tick).payload
    return Event(
        topic=service_topic(target),
        payload=service_tick_payload(heartbeat, target=target, mode="normal", divisor=1),
        source="body_adrenaline_scheduler_neuron",
        meta=service_tick_meta(target),
    )


def test_heartbeat_v2_spec_is_20_tps_and_infrastructure_only():
    assert HEARTBEAT_HZ == 20.0
    assert HEARTBEAT_INTERVAL_S == 0.05
    event = _heartbeat_event(12)
    assert event.payload["tick"] == 12
    assert event.payload["nominal_interval_s"] == 0.05
    assert event.meta["semantic_input"] is False
    assert event.meta["store_in_memory"] is False
    assert event.meta["reinforcement_eligible"] is False
    assert is_infrastructure_event(event) is True


def test_legacy_and_canonical_subscription_collapse_to_one_route():
    neuron = AliasHeartbeatNeuron(
        NeuronConfig(
            name="dual_named_consumer",
            subscribed_topics=["clock/tick", PRIMARY_HEARTBEAT_TOPIC],
            output_topics=[],
        )
    )
    assert set(neuron.subscribed_topics) == {PRIMARY_HEARTBEAT_TOPIC}
    assert len(neuron.subscribed_topics) == 1


def test_legacy_clock_alias_routes_once_on_body_bus_and_never_duplicates():
    async def run():
        orch = Orchestrator()
        alias = AliasHeartbeatNeuron(
            NeuronConfig(
                name="legacy_clock_consumer",
                subscribed_topics=["clock/tick"],
                output_topics=["diagnostic/alias_seen"],
            )
        )
        orch.register_neuron(alias)

        observed = []

        async def observe(event):
            observed.append(event)
            return []

        orch.bus.subscribe("test_main_observer", ["*"], observe, priority=-100)
        await orch.start()
        try:
            await orch.push_event("clock/tick", {"tick": 1, "ts": 1000.0}, source="legacy_clock")
            assert await orch.wait_for_idle(timeout=1.0)
        finally:
            await orch.stop()

        assert orch.kv_store["test:alias_count"] == 1
        assert [event.topic for event in observed] == ["diagnostic/alias_seen"]
        assert observed[0].payload["topic"] == "body/heartbeat"
        assert orch.body_bus.metrics.total_published == 1
        assert orch.bus.metrics.total_published == 1

    asyncio.run(run())


def test_200_heartbeat_pulses_stay_on_body_bus_while_cognition_service_runs_200_times():
    async def run():
        orch = Orchestrator()
        scheduler = _scheduler()
        orch.register_neuron(scheduler)
        counter = CountingServiceNeuron(
            NeuronConfig(
                name="count_cognition_service",
                subscribed_topics=[service_topic("cognition")],
                output_topics=[],
            )
        )
        orch.register_neuron(counter)

        await orch.start()
        try:
            for tick in range(1, 201):
                event = _heartbeat_event(tick)
                await orch.push_body_event(
                    event.topic,
                    event.payload,
                    meta=event.meta,
                    source=event.source,
                )
            assert await orch.wait_for_idle(timeout=2.0)
        finally:
            await orch.stop()

        assert orch.kv_store["test:service_count:cognition"] == 200
        # 200 raw heartbeats + 200 cognition service opportunities.
        assert orch.body_bus.metrics.total_published == 400
        # No semantic/cognitive event was generated merely because time passed.
        assert orch.bus.metrics.total_published == 0
        assert orch.event_queue.empty()
        assert orch.body_event_queue.empty()
        assert scheduler.get_hebbian_weight(PRIMARY_HEARTBEAT_TOPIC) == 0.0
        assert scheduler.get_activation_history() == ()

    asyncio.run(run())


def test_adrenaline_changes_selected_service_cadence_not_heartbeat_frequency():
    async def run():
        ctx = FakeCtx()
        ctx.kv["body:service_targets"] = ["vision", "power"]
        neuron = _scheduler()

        normal = list(await neuron.process(_heartbeat_event(4), ctx))
        assert [service_target(ev.topic) for ev in normal] == ["vision"]
        assert all(is_infrastructure_event(ev) for ev in normal)

        hazard = Event(topic=HAZARD_TOPIC, payload={"level": 3, "reason": "test_danger"}, source="test")
        transition = list(await neuron.process(hazard, ctx))
        assert len(transition) == 1
        assert transition[0].topic == AROUSAL_STATE_TOPIC
        assert transition[0].payload["mode"] == "emergency"
        assert transition[0].payload["policy"] == "fixed_20tps_selective_organ_surge"

        # Repeated hazard extends the hold but does not create another transition event.
        assert list(await neuron.process(hazard, ctx)) == []

        emergency = list(await neuron.process(_heartbeat_event(8), ctx))
        assert {service_target(ev.topic) for ev in emergency} == {"vision", "power"}
        assert ctx.kv["body:arousal_mode"] == "emergency"

    asyncio.run(run())


def test_body_infrastructure_is_rejected_by_every_memory_path():
    heartbeat = classify_event_for_memory(_heartbeat_event(1))
    service = classify_event_for_memory(_service_event("memory", 20))
    for classification in (heartbeat, service):
        assert classification["junk_reason"] == "body_infrastructure"
        assert classification["allow_longterm"] is False
        assert classification["allow_trace"] is False
        assert classification["allow_hrm"] is False
        assert classification["allow_pattern"] is False


def test_thought_turn_cognition_service_is_housekeeping_only_and_preserves_last_turn_state():
    async def run():
        ctx = FakeCtx()
        ctx.kv["thought:turn:last_state"] = {"reason": "semantic_input", "dominant_need": "curiosity"}
        neuron = ThoughtTurnArbitrationNeuron(
            NeuronConfig(
                name="thought_turn_arbitration_neuron",
                subscribed_topics=[service_topic("cognition"), "thought/internal"],
                output_topics=["thought/object", "thought/action_candidate", "thought/turn_state"],
            )
        )
        outputs = list(await neuron.process(_service_event("cognition", 1), ctx))
        assert outputs == []
        assert ctx.kv["thought:turn:last_state"] == {
            "reason": "semantic_input",
            "dominant_need": "curiosity",
        }
        assert ctx.kv.get("thought:turn:last_housekeeping_ts", 0.0) > 0.0

    asyncio.run(run())
