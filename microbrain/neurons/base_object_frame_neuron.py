from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.objects.base_object import build_event_object, build_scene_object
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class BaseObjectFrameNeuron(BaseNeuron):
    """
    First-pass base.object bridge.

    Turns cognition-plane percepts/actions into lightweight object frames and
    keeps a rolling `object:current_scene` KV packet.

    This is intentionally non-invasive: it does not rewrite memory yet. It gives
    context, memory, vision, and future action/result systems a shared object
    vocabulary to begin referencing.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        internal_state = await self._internal_state(ctx)
        obj = build_event_object(event, internal_state=internal_state)
        if not obj:
            return []

        recent: List[Dict[str, Any]] = await ctx.get_kv("object:recent", [])
        if not isinstance(recent, list):
            recent = []
        recent.append(obj)

        now = time.time()
        ttl_s = float(await ctx.get_kv("object:scene_ttl_s", 45.0) or 45.0)
        max_recent = int(await ctx.get_kv("object:recent_max", 48) or 48)
        pruned: List[Dict[str, Any]] = []
        for item in recent[-max_recent:]:
            if not isinstance(item, dict):
                continue
            try:
                updated_at = float(item.get("updated_at", item.get("created_at", now)) or now)
            except Exception:
                updated_at = now
            if (now - updated_at) <= ttl_s:
                pruned.append(item)
        recent = pruned[-max_recent:]

        previous_scene = await ctx.get_kv("object:current_scene", {})
        previous_scene_id = str(previous_scene.get("object_id", "") or "") if isinstance(previous_scene, dict) else ""
        scene = build_scene_object(recent, internal_state=internal_state, previous_scene_id=previous_scene_id)

        await ctx.set_kv("object:last", obj)
        await ctx.set_kv("object:recent", recent)
        await ctx.set_kv("object:current_scene", scene)
        await ctx.set_kv("context:current_scene", scene)

        return [
            Event(
                topic="object/base",
                payload=obj,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"schema_ver": obj.get("schema_ver"), "kind": obj.get("kind"), "cognitive_visible": False},
            ),
            Event(
                topic="object/scene",
                payload=scene,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"schema_ver": scene.get("schema_ver"), "kind": "scene.object", "cognitive_visible": False},
            ),
        ]

    async def _internal_state(self, ctx) -> Dict[str, Any]:
        keys = {
            "power": "power:state",
            "battery": "power:battery_state",
            "boredom": "drive:boredom",
            "social_interaction": "drive:social_interaction",
            "social_experimentation": "drive:social_experimentation",
            "hormones": "drive:hormones",
            "want_vector": "drive:want_vector",
            "affect": "affect:last",
            "relation": "relation:last",
        }
        out: Dict[str, Any] = {"ts": time.time()}
        for name, key in keys.items():
            try:
                value = await ctx.get_kv(key, None)
            except Exception:
                value = None
            if value not in (None, {}, []):
                out[name] = value
        return out


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "percept/vision",
            "vision/object_delta",
            "percept/audio",
            "percept/touch",
            "vision/proto_object",
            "act/speech",
        ],
        output_topics=["object/base", "object/scene"],
        priority=2,
    )
    return [BaseObjectFrameNeuron(cfg)]
