from __future__ import annotations

import asyncio

from microbrain.neurons.input_text import TextInputNeuron
from microbrain.neurons.vision_object_family_neuron import VisionObjectFamilyNeuron
from microbrain.neurons.scene_expectation_neuron import SceneExpectationNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self):
        self.kv = {}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value


def test_scene_command_is_context_only():
    async def run():
        neuron = TextInputNeuron(NeuronConfig(
            name="input_text", subscribed_topics=["input/text"],
            output_topics=["percept/text"],
        ))
        ctx = FakeCtx()
        out = await neuron.process(Event(topic="input/text", payload="/scene workshop"), ctx)
        assert ctx.kv["scene:manual_active"]["name"] == "workshop"
        assert ctx.kv["scene:manual_active"]["verified"] is False
        assert not any(event.topic == "percept/text" for event in out)
        scene = SceneExpectationNeuron(NeuronConfig(
            name="scene_expectation", subscribed_topics=["object/scene"], output_topics=[],
        ))
        assert scene._place_key({"meta": {"location_hint": {"name": "Workshop"}}}) == "manual:workshop"
        await neuron.process(Event(topic="input/text", payload="/scene clear"), ctx)
        assert ctx.kv["scene:manual_active"] is None

    asyncio.run(run())


def test_common_motion_family_is_provisional_and_revisable():
    async def run():
        neuron = VisionObjectFamilyNeuron(NeuronConfig(
            name="vision_family", subscribed_topics=["vision/object_isolation"],
            output_topics=[],
        ))
        ctx = FakeCtx()

        def obj(name, x, dx):
            return {
                "track_id": name, "status": "isolated",
                "bbox": [x, 20, 20, 20],
                "motion": {"dx": dx, "dy": 0},
            }

        for i in range(15):
            await neuron.process(Event(
                topic="vision/object_isolation",
                payload={"ts": 1 + i * 0.2,
                         "objects": [obj("A", 20 + i * 4, 4), obj("B", 60 + i * 4, 4)]},
            ), ctx)
        families = ctx.kv["vision:object_families"]["families"]
        assert families and families[0]["tier"] == "ESTABLISHED"
        assert families[0]["physical_identity"] == "unresolved"
        for i in range(8):
            await neuron.process(Event(
                topic="vision/object_isolation",
                payload={"ts": 4 + i * 0.2,
                         "objects": [obj("A", 80, 0), obj("B", 120 + i * 4, 4)]},
            ), ctx)
        families = ctx.kv["vision:object_families"]["families"]
        assert not families or families[0]["tier"] != "ESTABLISHED"

    asyncio.run(run())
