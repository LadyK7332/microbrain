from __future__ import annotations

import asyncio
import time
from pathlib import Path

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.light_probe_neuron import LightProbeNeuron, SERVICE_TOPIC
from microbrain.orchestrator.neuron_base import Event, NeuronConfig


class FakeCtx:
    def __init__(self, tmp_path: Path):
        self.store = MemCellStore(tmp_path)
        self.kv = {
            "memory:mem_cell_store": self.store,
            "probe:enabled": True,
            "probe:idle_wander_enabled": True,
            "probe:every_s": 0.0,
            "probe:settling_threshold_s": 10.0,
            "probe:idle_wander_threshold_s": 30.0,
            "probe:current_window_s": 300.0,
            "probe:idle_candidate_threshold": 0.0,
            "probe:block_during_slearn_enabled": True,
            "probe:block_during_read_enabled": True,
        }

    async def get_kv(self, key, default=None):
        return self.kv.get(key, default)

    async def set_kv(self, key, value):
        self.kv[key] = value

    async def log_debug(self, *args, **kwargs):
        pass

    async def log_warn(self, *args, **kwargs):
        pass

    async def log_error(self, *args, **kwargs):
        pass


def _seed_probe_cell(store: MemCellStore) -> None:
    store.append_cell(
        {
            "id": "cell:city",
            "kind": "token_anchor",
            "tier": "now",
            "anchor": {"kind": "token", "ref": "majestic city"},
            "activation": 0.22,
            "promotion": 0.02,
            "encounter_count": 1,
            "links_explicit": [],
            "refs": [],
            "modalities": ["text"],
        },
        tier="now",
    )


def test_light_probe_stays_quiet_when_present_context_is_active(tmp_path: Path) -> None:
    async def run():
        ctx = FakeCtx(tmp_path)
        _seed_probe_cell(ctx.store)
        ctx.kv["interaction:last_input"] = {"ts": time.time(), "text": "what is a majestic city"}
        neuron = LightProbeNeuron(
            NeuronConfig(name="light_probe_neuron", subscribed_topics=[SERVICE_TOPIC], output_topics=["thought/probe"])
        )

        out = list(await neuron.process(Event(topic=SERVICE_TOPIC, payload={}), ctx))

        assert out == []
        state = ctx.kv["probe:runtime_state"]
        assert state["mode"] == "active"
        assert state["blocked_reason"] == "present_context_active"
        assert "what is a majestic city" in state["anchors"]

    asyncio.run(run())


def test_light_probe_allows_origin_tagged_idle_wander_after_threshold(tmp_path: Path) -> None:
    async def run():
        ctx = FakeCtx(tmp_path)
        _seed_probe_cell(ctx.store)
        ctx.kv["interaction:last_input"] = {"ts": time.time() - 120.0, "text": "what is a majestic city"}
        neuron = LightProbeNeuron(
            NeuronConfig(name="light_probe_neuron", subscribed_topics=[SERVICE_TOPIC], output_topics=["thought/probe"])
        )

        out = list(await neuron.process(Event(topic=SERVICE_TOPIC, payload={}), ctx))

        assert len(out) == 1
        assert out[0].topic == "thought/probe"
        assert out[0].payload["origin"] == "idle_wander"
        assert out[0].meta["origin"] == "idle_wander"
        assert ctx.kv["probe:runtime_state"]["mode"] == "idle_wander"
        assert ctx.kv["probe:last"]["origin"] == "idle_wander"

    asyncio.run(run())


def test_light_probe_blocks_wander_while_slearn_background_is_active(tmp_path: Path) -> None:
    async def run():
        ctx = FakeCtx(tmp_path)
        _seed_probe_cell(ctx.store)
        ctx.kv["interaction:last_input"] = {"ts": time.time() - 120.0, "text": "what is a majestic city"}
        ctx.kv["slearn:status"] = "ingesting"
        neuron = LightProbeNeuron(
            NeuronConfig(name="light_probe_neuron", subscribed_topics=[SERVICE_TOPIC], output_topics=["thought/probe"])
        )

        out = list(await neuron.process(Event(topic=SERVICE_TOPIC, payload={}), ctx))

        assert out == []
        state = ctx.kv["probe:runtime_state"]
        assert state["mode"] == "background_blocked"
        assert "slearn_background_active" in state["blocked_reason"]

    asyncio.run(run())
