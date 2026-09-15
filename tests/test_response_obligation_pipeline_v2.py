import asyncio

from microbrain.neurons.action_selector_neuron import ActionSelectorNeuron
from microbrain.neurons.desire_trigger_neuron import DesireTriggerNeuron
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class DummyCtx:
    def __init__(self):
        self.kv = {"llm:enabled": False}

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value


def test_non_question_release_carries_one_obligation_into_reasoning():
    async def scenario():
        ctx = DummyCtx()
        desire = DesireTriggerNeuron(NeuronConfig(name="desire", subscribed_topics=["hypothesis/ready"]))
        ready = Event(
            topic="hypothesis/ready",
            correlation_id="turn-status",
            payload={
                "context": {
                    "input": {
                        "text": "I am halfway done.",
                        "source": "user",
                        "channel": "textual",
                        "correlation_id": "turn-status",
                        "meaningful_tokens": ["halfway", "done"],
                        "raw_meta": {},
                    },
                    "cues": {},
                    "constraints": {},
                    "drives": {"boredom": {}},
                },
                "hypothesis": {
                    "hypothesis_id": "hyp-status",
                    "should_respond": True,
                    "response_demand": 1.0,
                    "expected_usefulness": 1.0,
                    "silence_score": 0.0,
                    "recommended_action": "acknowledge",
                    "recommended_action_score": 0.9,
                    "pattern_analysis": {
                        "statement_kind": "status_update",
                        "continuity": 0.8,
                        "uncertainty": 0.1,
                        "risk": 0.0,
                    },
                },
            },
        )
        outputs = list(await desire.process(ready, ctx))
        assert [event.topic for event in outputs] == [
            "hypothesis/action_committed",
            "speech/response_obligation",
            "release/request",
        ]
        obligation = outputs[1].payload
        assert obligation["required_act"] == "acknowledge"
        assert obligation["required"] is True

        selector = ActionSelectorNeuron(NeuronConfig(name="selector", subscribed_topics=["release/request"]))
        reason_events = list(await selector.process(outputs[2], ctx))
        assert len(reason_events) == 1
        assert reason_events[0].topic == "reason/request"
        assert reason_events[0].payload["response_obligation"]["obligation_id"] == obligation["obligation_id"]

    asyncio.run(scenario())
