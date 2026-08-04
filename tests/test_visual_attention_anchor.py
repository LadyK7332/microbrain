from __future__ import annotations

import asyncio

from microbrain.neurons.context_builder_neuron import ContextBuilderNeuron
from microbrain.neurons.input_text import TextInputNeuron
from microbrain.neurons.visual_attention_anchor_neuron import VisualAttentionAnchorNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self):
        self.kv = {
            "visual:current": {
                "objects": [
                    {
                        "track_id": "vobj:42",
                        "label": "unknown",
                        "status": "isolated",
                        "confidence": 0.66,
                        "bbox": [100, 80, 60, 70],
                        "snippet_ref": "ram:vision:object:vobj:42",
                    }
                ]
            }
        }

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass


def test_selected_visual_object_binds_next_input_as_attention_not_identity() -> None:
    async def run():
        ctx = FakeCtx()
        anchor_neuron = VisualAttentionAnchorNeuron(
            NeuronConfig(
                name="visual_attention_anchor_neuron",
                subscribed_topics=["control/vision_attention"],
                output_topics=["vision/attention_anchor"],
            )
        )
        emitted = list(
            await anchor_neuron.process(
                Event(
                    topic="control/vision_attention",
                    payload={"action": "select", "track_id": "vobj:42"},
                ),
                ctx,
            )
        )
        assert emitted and emitted[0].topic == "vision/attention_anchor"
        anchor = ctx.kv["vision:attention_anchor"]
        assert anchor["track_id"] == "vobj:42"
        assert anchor["semantics"] == "attention_only_not_identity_assertion"
        assert anchor["label_hint"] == "unknown"

        input_neuron = TextInputNeuron(
            NeuronConfig(name="input_text", subscribed_topics=["input/text"], output_topics=["percept/text"])
        )
        percepts = list(
            await input_neuron.process(
                Event(topic="input/text", payload="what do you think this is?", meta={"source": "ui", "channel": "textual"}),
                ctx,
            )
        )
        percept = percepts[-1]
        ref = percept.payload["raw_meta"]["visual_attention_ref"]
        assert ref["track_id"] == "vobj:42"
        assert percept.payload["raw_meta"]["deictic_binding_hint"] == "vobj:42"
        assert ctx.kv["vision:attention_anchor"] is None, "one-turn pointing anchor should be consumed"
        assert ctx.kv["visual:current"]["objects"][0]["label"] == "unknown", "selection must not label the object"

        context_neuron = ContextBuilderNeuron(
            NeuronConfig(name="context_builder_neuron", subscribed_topics=["context/request"], output_topics=["context/built"])
        )
        built = list(
            await context_neuron.process(
                Event(
                    topic="context/request",
                    payload={
                        "text": percept.payload["text"],
                        "source": percept.payload["source"],
                        "channel": percept.payload["channel"],
                        "raw_meta": percept.payload["raw_meta"],
                    },
                ),
                ctx,
            )
        )
        context = built[0].payload["context"]
        assert context["attention"]["visual_ref"]["track_id"] == "vobj:42"
        assert context["attention"]["rule"] == "selected_visual_ref_is_attention_context_not_identity_truth"

    asyncio.run(run())
