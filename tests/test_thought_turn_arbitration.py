import asyncio

from microbrain.neurons.thought_turn_arbitration_neuron import ThoughtTurnArbitrationNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig
from microbrain.utils.heartbeat_stream import service_topic


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


def make_neuron():
    return ThoughtTurnArbitrationNeuron(
        NeuronConfig(
            name="thought_turn_arbitration_neuron",
            subscribed_topics=["drive/power_request", service_topic("cognition"), "event/relief/power"],
            output_topics=["thought/object", "thought/action_candidate", "thought/turn_state"],
        )
    )


def test_power_need_becomes_ready_action_candidate_when_components_exist():
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        event = Event(
            topic="drive/power_request",
            payload={
                "thought_text": "Power is low at 40%. I need to charge.",
                "pressure": {"urgency": 0.72},
                "vector": {"outlet": "textual", "style": "direct_simple"},
                "style": "direct_simple",
            },
        )
        outputs = list(await neuron.process(event, ctx))
        topics = [ev.topic for ev in outputs]
        drawer = ctx.kv["thought:drawer"]

        assert "thought/object" in topics
        assert "thought/action_candidate" in topics
        assert drawer[0]["family"] == "power"
        assert drawer[0]["status"] == "ready"
        assert drawer[0]["missing_components"] == []

    asyncio.run(run())


def test_missing_motion_component_places_thought_in_drawer():
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        event = Event(
            topic="drive/power_request",
            payload={
                "thought_text": "Power is low and a motion route would reach the charger.",
                "pressure": {"urgency": 0.8},
                "vector": {"outlet": "motion", "style": "direct_simple"},
                "style": "direct_simple",
            },
        )
        outputs = list(await neuron.process(event, ctx))
        topics = [ev.topic for ev in outputs]
        drawer = ctx.kv["thought:drawer"]

        assert "thought/object" in topics
        assert "thought/action_candidate" not in topics
        assert drawer[0]["status"] == "drawer_waiting"
        assert "motion_available" in drawer[0]["missing_components"]

    asyncio.run(run())


def test_power_relief_marks_power_thought_fulfilled_and_learns_modifier():
    async def run():
        ctx = FakeCtx()
        neuron = make_neuron()
        event = Event(
            topic="drive/power_request",
            payload={
                "thought_text": "Power is low at 40%. I need to charge.",
                "pressure": {"urgency": 0.72},
                "vector": {"outlet": "textual", "style": "direct_simple"},
            },
        )
        await neuron.process(event, ctx)
        relief = Event(topic="event/relief/power", payload={"delta_pct": 8.0})
        await neuron.process(relief, ctx)

        drawer = ctx.kv["thought:drawer"]
        learned = ctx.kv["thought:priority:learned_modifiers"]
        assert drawer[0]["status"] == "fulfilled"
        assert drawer[0]["memory_candidate"] is True
        assert learned["power"] > 0.0

    asyncio.run(run())
