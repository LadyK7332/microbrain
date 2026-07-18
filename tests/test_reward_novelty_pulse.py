import asyncio

from microbrain.neurons.boredom_drive_neuron import BoredomDriveNeuron
from microbrain.neurons.reward_novelty_pulse_neuron import RewardNoveltyPulseNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


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


def make_reward():
    return RewardNoveltyPulseNeuron(
        NeuronConfig(
            name="reward_novelty_pulse_neuron",
            subscribed_topics=["percept/text", "control/reinforce", "clock/tick"],
            output_topics=["affect/reward"],
        )
    )


def make_boredom():
    return BoredomDriveNeuron(
        NeuronConfig(
            name="boredom_drive_neuron",
            subscribed_topics=["percept/text", "control/reinforce", "affect/reward", "clock/tick"],
            output_topics=[],
        )
    )


def test_positive_accent_creates_dopamine_reward_state():
    async def run():
        ctx = FakeCtx()
        reward = make_reward()
        event = Event(
            topic="percept/text",
            payload={
                "text": "great job",
                "source": "user",
                "channel": "textual",
                "raw_meta": {"accent_positive": 5.0, "accent_source": "acc_command"},
            },
        )
        outputs = list(await reward.process(event, ctx))
        state = ctx.kv["affect:reward_state"]
        novelty = ctx.kv["affect:novelty_state"]

        assert state["level"] > 0.45
        assert novelty["level"] > 0.1
        assert any(ev.topic == "affect/reward" for ev in outputs)

    asyncio.run(run())


def test_positive_reward_relieves_boredom():
    async def run():
        ctx = FakeCtx()
        boredom = make_boredom()
        # Seed a high boredom state, as seen during live UI testing.
        await boredom.save_state(
            ctx,
            "boredom_state",
            {
                "prev_idx": None,
                "repetitions": 0,
                "level": 1.0,
                "last_tick_ts": 1000.0,
                "last_external_ts": 1000.0,
                "last_output_fp": "",
                "last_user_fp": "",
                "same_output_repetitions": 0,
                "same_user_repetitions": 0,
                "novelty_delta": 0.0,
            },
        )
        event = Event(topic="affect/reward", payload={"reward_delta": 0.6, "boredom_relief": 0.6})
        await boredom.process(event, ctx)

        assert ctx.kv["drive:boredom"]["level"] < 1.0
        assert ctx.kv["drive:boredom"]["novelty_delta"] > 0.0

    asyncio.run(run())
