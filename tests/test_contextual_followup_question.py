from __future__ import annotations

import asyncio
import time

from microbrain.neurons.context_builder_neuron import ContextBuilderNeuron
from microbrain.neurons.native_responder_neuron import NativeResponderNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig
from microbrain.patterns.pattern_toolkit import PatternToolkit


class FakeCtx:
    def __init__(self, initial=None):
        self.kv = dict(initial or {})

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass

    async def log_warn(self, *args, **kwargs):
        pass


def make_context_builder():
    return ContextBuilderNeuron(
        NeuronConfig(
            name="context_builder_neuron",
            subscribed_topics=["context/request"],
            output_topics=["context/built"],
        )
    )


def make_native_responder():
    return NativeResponderNeuron(
        NeuronConfig(
            name="native_responder_neuron",
            subscribed_topics=["reason/request"],
            output_topics=["act/speech"],
        )
    )


def _scene_with_turns(turns):
    now = time.time()
    return {
        "turns": [
            {
                "role": role,
                "text": text,
                "tokens": [tok for tok in text.lower().replace("?", " ").replace(".", " ").split() if len(tok) >= 4],
                "correlation_id": f"turn-{idx}",
                "ts": now - (len(turns) - idx),
            }
            for idx, (role, text) in enumerate(turns)
        ],
        "summary": {"topic": "conversation", "active_threads": ["conversation"]},
    }


def test_bare_question_uses_previous_dialogue_turn_as_target():
    async def run():
        scene = _scene_with_turns(
            [
                ("user", "good morning"),
                ("user", "Saluations?"),
                ("user", "No interactions today?"),
                ("user", "You sure?"),
                ("assistant", "are you sure"),
                ("user", "well, of course I am, gotta talk to you some."),
                ("user", "Would'nt you think so."),
                ("user", "?"),
            ]
        )
        ctx = FakeCtx({"conversation:scene": scene, "conversation:summary": scene["summary"]})
        event = Event(
            topic="context/request",
            payload={"text": "?", "source": "user", "channel": "textual"},
            correlation_id="turn-7",
        )

        out = list(await make_context_builder().process(event, ctx))
        context = out[0].payload["context"]
        followup = context["followup_question"]

        assert followup["active"] is True
        assert followup["target_text"] == "Would'nt you think so."
        assert followup["target_role"] == "user"
        assert context["cues"]["contextual_followup_question"] is True
        assert context["dialogue_window"][-1]["text"] == "Would'nt you think so."
        assert "think" in context["input"]["meaningful_tokens"]
        assert context["input"]["literal_meaningful_tokens"] == []

        analysis = PatternToolkit().analyze(context)
        assert analysis["statement_kind"] == "question"
        assert analysis["contextual_followup"] is True
        assert analysis["question_target_text"] == "Would'nt you think so."
        assert "think" in analysis["meaningful_tokens"]
        assert any(item["type"] == "contextual_followup_question" for item in analysis["patterns"])

    asyncio.run(run())


def test_dialogue_window_is_capped_at_last_ten_turns():
    async def run():
        turns = [("user", f"line {idx} with context") for idx in range(14)]
        turns.append(("user", "?"))
        scene = _scene_with_turns(turns)
        ctx = FakeCtx({"conversation:scene": scene, "conversation:summary": scene["summary"]})
        event = Event(topic="context/request", payload={"text": "?"}, correlation_id="turn-14")

        out = list(await make_context_builder().process(event, ctx))
        context = out[0].payload["context"]

        assert len(context["dialogue_window"]) == 10
        assert context["dialogue_window"][0]["text"] == "line 4 with context"
        assert context["followup_question"]["target_text"] == "line 13 with context"

    asyncio.run(run())


def test_native_question_lookup_queries_resolved_target():
    responder = make_native_responder()
    context = {
        "followup_question": {
            "active": True,
            "target_text": "Would'nt you think so.",
            "target_role": "user",
        }
    }

    assert responder._effective_question_query_text("?", context) == "Would'nt you think so?"
