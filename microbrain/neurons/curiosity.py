from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("curiosity")


class CuriosityNeuron(BaseNeuron):
    """
    Curiosity = drive -> probe -> action.

    v1: only emits a babble probe when boredom is active and attention allows it.
    v2+: can emit vision/audio/movement probes.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != SERVICE_TOPIC:
            return []

        # Legacy curiosity mode is now opt-in. The initiative threshold neuron
        # owns autonomous reflection by default so MB stays constrained/objective.
        if not bool(await ctx.get_kv("initiative:legacy_curiosity", False)):
            return []

        boredom = await ctx.get_kv("drive:boredom", {})
        level = float(boredom.get("level", 0.0) or 0.0)
        active = bool(boredom.get("active", False))

        if not active:
            return []

        # v1 gating: only babble if attention says it's okay
        allow_babble = bool(await ctx.get_kv("attention:allow_babble", True))
        if not allow_babble:
            return []

        # v1 action: "babble" probe (empty prompt)
        self.debug("curiosity_probe", kind="babble", boredom_level=level)

        return [
            Event(
                topic="reason/request",
                payload={
                    "text": "",  # babble probe: empty prompt is fine
                    "source": "internal",
                    "channel": "thought",
                    "raw_meta": {
                        "autonomous": True,
                        "curiosity": "babble",
                        "mode": "babble",
                    },
                },
                source=NEURON_NAME,
                meta={"autonomous": True, "curiosity": "babble", "mode": "babble"},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[SERVICE_TOPIC],
        output_topics=["reason/request"],
        priority=0,
    )
    yield CuriosityNeuron(cfg)
