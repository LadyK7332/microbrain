import asyncio

from microbrain.neurons.capability_circulation_neuron import CapabilityCirculationNeuron
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


def make_capability():
    return CapabilityCirculationNeuron(
        NeuronConfig(
            name="capability_circulation_neuron",
            subscribed_topics=["control/capability", "thought/action_candidate", service_topic("capability")],
            output_topics=["capability/state", "capability/readiness", "thought/drawer_recheck"],
        )
    )


def make_turn():
    return ThoughtTurnArbitrationNeuron(
        NeuronConfig(
            name="thought_turn_arbitration_neuron",
            subscribed_topics=["drive/power_request", "thought/drawer_recheck"],
            output_topics=["thought/object", "thought/action_candidate", "thought/turn_state"],
        )
    )


def test_capability_fallback_audio_satisfies_textual_requirement():
    async def run():
        ctx = FakeCtx()
        cap = make_capability()
        await cap.process(Event(topic="control/capability", payload={"component": "textual_available", "available": False, "ttl_s": 0}), ctx)
        await cap.process(Event(topic="control/capability", payload={"component": "audio_available", "available": True, "ttl_s": 0}), ctx)

        thought = {
            "id": "thought:test",
            "family": "expression",
            "need": "speech",
            "required_components": ["textual_available", "speech_allowed"],
            "route": {"outlet": "textual"},
        }
        outputs = list(await cap.process(Event(topic="thought/action_candidate", payload={"thought": thought}), ctx))
        readiness = next(ev.payload for ev in outputs if ev.topic == "capability/readiness")

        assert readiness["ready"] is True
        assert readiness["fallback_used"]["textual_available"] == "audio_available"

    asyncio.run(run())


def test_capability_update_rechecks_waiting_thought_drawer():
    async def run():
        ctx = FakeCtx()
        turn = make_turn()
        cap = make_capability()

        power_event = Event(
            topic="drive/power_request",
            payload={
                "thought_text": "Power is low and a motion route would reach the charger.",
                "pressure": {"urgency": 0.8},
                "vector": {"outlet": "motion"},
            },
        )
        await turn.process(power_event, ctx)
        assert ctx.kv["thought:drawer"][0]["status"] == "drawer_waiting"

        cap_outputs = list(await cap.process(Event(topic="control/capability", payload={"component": "motion_available", "available": True, "ttl_s": 0}), ctx))
        recheck = next(ev for ev in cap_outputs if ev.topic == "thought/drawer_recheck")
        turn_outputs = list(await turn.process(recheck, ctx))

        assert ctx.kv["thought:drawer"][0]["status"] == "ready"
        assert any(ev.topic == "thought/action_candidate" for ev in turn_outputs)

    asyncio.run(run())
