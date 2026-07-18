from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class SensorySpineNeuron(BaseNeuron):
    """
    Collect a compact rolling ledger of recent percepts so the cognition core
    can reason over convergence without each lobe owning its own little world.

    This neuron does not decide behavior. It only normalizes + records what the
    specialized lobes observed, which keeps the shared spine explicit.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in ("percept/text", "percept/vision", "percept/audio", "percept/touch"):
            return []

        payload = event.payload if isinstance(event.payload, dict) else {"value": event.payload}
        meta = dict(event.meta or {})
        raw_meta = dict(payload.get("raw_meta") or {}) if isinstance(payload, dict) else {}
        modality = str(meta.get("modality") or raw_meta.get("input_modality") or event.topic.split("/", 1)[-1])
        lobe = str(meta.get("lobe") or raw_meta.get("sensor_lobe") or modality)

        summary = self._summarize(event.topic, payload)
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "topic": event.topic,
            "source": event.source,
            "modality": modality,
            "lobe": lobe,
            "adapter": str(meta.get("adapter") or raw_meta.get("adapter") or event.source),
            "channel": str(payload.get("channel") or raw_meta.get("channel") or meta.get("interface") or "default"),
            "summary": summary,
            "correlation_id": event.correlation_id,
        }

        recent: List[Dict[str, Any]] = await self.load_state(ctx, "recent_percepts", default=[])
        if not isinstance(recent, list):
            recent = []
        recent.append(entry)
        max_items = int(await ctx.get_kv("spine:recent_percepts_max", 64) or 64)
        if len(recent) > max_items:
            recent = recent[-max_items:]
        await self.save_state(ctx, "recent_percepts", recent)

        await ctx.set_kv(f"spine:last:{modality}", entry)
        await ctx.set_kv("spine:last_percept", entry)

        # Active-sense latch for curiosity/clarification gating.
        # This lets curiosity ask about the sense that is actually live now
        # instead of interrupting text with stale vision/audio gaps.
        await ctx.set_kv("spine:active_sense", modality)
        await ctx.set_kv("spine:active_sense:last", entry)
        return []

    def _summarize(self, topic: str, payload: Dict[str, Any]) -> str:
        if topic == "percept/text":
            text = str(payload.get("text", "") or "").strip()
            return text[:240]
        if topic == "percept/vision":
            description = str(payload.get("description", "") or "").strip()
            objects = payload.get("objects", []) or []
            if not isinstance(objects, list):
                objects = [objects]
            obj_txt = ", ".join(str(o) for o in objects[:8]).strip()
            if description and obj_txt:
                return f"{description} | objects: {obj_txt}"[:240]
            return (description or obj_txt)[:240]
        if topic == "percept/audio":
            text = str(payload.get("text", "") or "").strip()
            return text[:240]
        return str(payload)[:240]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text", "percept/vision", "percept/audio", "percept/touch"],
        output_topics=[],
        priority=3,
    )
    yield SensorySpineNeuron(cfg)
