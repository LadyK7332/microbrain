from __future__ import annotations

import asyncio

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.input_text import TextInputNeuron
from microbrain.neurons.visual_attention_anchor_neuron import VisualAttentionAnchorNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self):
        self.kv = {"visual:current": {"objects": []}}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass


async def _bind_frozen_visual_then_type(text: str):
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
                payload={
                    "action": "select",
                    "track_id": "vobj:eye-candidate",
                    "frozen": True,
                    "object_snapshot": {
                        "track_id": "vobj:eye-candidate",
                        "label": "unknown",
                        "status": "isolated",
                        "confidence": 0.68,
                        "bbox": [120, 80, 32, 18],
                        "snippet_ref": "ram:vision:object:vobj:eye-candidate",
                        "ui_frozen": True,
                        "ui_snapshot": {
                            "frame_label": "frozen teaching frame",
                            "source_width": 640,
                            "source_height": 480,
                            "frozen": True,
                        },
                    },
                },
            ),
            ctx,
        )
    )
    assert emitted and emitted[0].topic == "vision/attention_anchor"
    assert ctx.kv["vision:attention_anchor"]["frozen"] is True
    assert ctx.kv["vision:attention_anchor"]["visual_evidence_ref"].startswith("vision:evidence:")

    input_neuron = TextInputNeuron(
        NeuronConfig(name="input_text", subscribed_topics=["input/text"], output_topics=["percept/text"])
    )
    percepts = list(
        await input_neuron.process(
            Event(topic="input/text", payload=text, meta={"source": "ui", "channel": "dashboard"}),
            ctx,
        )
    )
    return percepts[-1]


def test_frozen_visual_selection_binds_this_is_eye_to_visual_evidence(tmp_path) -> None:
    percept = asyncio.run(_bind_frozen_visual_then_type("this is an eye"))
    raw_meta = percept.payload["raw_meta"]
    ref = raw_meta["visual_attention_ref"]
    assert ref["track_id"] == "vobj:eye-candidate"
    assert ref["frozen"] is True
    assert raw_meta["deictic_binding_hint"] == "vobj:eye-candidate"

    store = MemCellStore(tmp_path, composer_enabled=False)
    result = store.ingest_text(
        text=percept.payload["text"],
        topic="percept/text",
        role="user",
        transport_source="dashboard",
        source="test",
        meta=raw_meta,
        tier="now",
    )
    cell = result["learning_frames"][0]
    assert cell["meta"]["pattern_type"] == "designation_claim"
    assert cell["meta"]["visual_binding"]["bound"] is True
    assert cell["meta"]["visual_binding"]["frozen"] is True
    assert cell["meta"]["slots"]["designation"] == "eye"
    assert cell["meta"]["slots"]["visual_track_id"] == "vobj:eye-candidate"
    assert "vision" in cell["modalities"]
    assert any(ref["kind"] == "visual_ref" and ref["value"] == "vobj:eye-candidate" for ref in cell["refs"])
    assert cell["meta"]["creates_prebuilt_answer"] is False


def test_explicit_vobj_wording_can_bind_selected_visual_ref(tmp_path) -> None:
    store = MemCellStore(tmp_path, composer_enabled=False)
    anchor = {
        "track_id": "vobj:xyz",
        "selected_track_id": "vobj:xyz",
        "visual_evidence_ref": "vision:evidence:abc",
        "bbox": [1, 2, 3, 4],
        "frozen": True,
    }
    result = store.ingest_text(
        text="vobj xyz is an eye",
        topic="percept/text",
        role="user",
        transport_source="dashboard",
        source="test",
        meta={"visual_attention_ref": anchor},
        tier="now",
    )
    assert result["learning_frames"]
    cell = result["learning_frames"][0]
    assert cell["meta"]["visual_binding"]["bound"] is True
    assert cell["meta"]["slots"]["visual_track_id"] == "vobj:xyz"
