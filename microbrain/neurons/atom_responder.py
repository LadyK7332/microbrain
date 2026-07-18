from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _short_concept(x: str) -> str:
    # "concept:pumpkin" -> "pumpkin"
    s = str(x or "").strip()
    if ":" in s:
        return s.split(":", 1)[1]
    return s


class AtomResponderNeuron(BaseNeuron):
    """
    Atom label tracker.

    The old direct speech label scaffold was removed after proving the route.
    New labels now enter thought/internal and curiosity pressure instead of
    speaking a canned line.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "memory/atom":
            return []

        atom = event.payload if isinstance(event.payload, dict) else {}
        if not isinstance(atom, dict) or atom.get("schema") != "atom.v1":
            return []

        if atom.get("atom_type") != "isa":
            return []

        subj = str(atom.get("subj", "") or "").strip()
        pred = str(atom.get("pred", "") or "").strip()
        if not subj.startswith("ent:") or not pred.startswith("concept:"):
            return []

        # Don’t talk during sleep (you already clamp babble there)
        power = await ctx.get_kv("power:state", {}) or {}
        if isinstance(power, dict) and bool(power.get("sleep", False)):
            return []

        # Only narrate the current focus entity (prevents random “drive-by labels”)
        focus_ent = await ctx.get_kv("atoms:focus_ent", None)
        if isinstance(focus_ent, str) and focus_ent and subj != focus_ent:
            return []

        # Dedupe: only say this (ent, concept) once every N seconds
        now = time.time()
        last_map = await ctx.get_kv("atoms:announced", {}) or {}
        if not isinstance(last_map, dict):
            last_map = {}

        key = f"{subj}|{pred}"
        cooldown_s = float(await ctx.get_kv("atoms:announce_cooldown_s", 30.0) or 30.0)
        t_last = float(last_map.get(key, 0.0) or 0.0)
        if (now - t_last) < cooldown_s:
            return []

        last_map[key] = now
        await ctx.set_kv("atoms:announced", last_map)

        label = _short_concept(pred)

        pause_s = float(await ctx.get_kv("atoms:after_label_pause_s", 5.0) or 5.0)

        return [
            Event(
                topic="thought/internal",
                payload={
                    "kind": "atom_label",
                    "entity": subj,
                    "concept": pred,
                    "label": label,
                },
                source=NEURON_NAME,
                correlation_id=event.correlation_id,
                meta={"channel": "thought", "kind": "atom_label", "store_in_memory": False},
            ),
            Event(
                topic="curiosity/adjust",
                payload={"pause_s": pause_s, "reason": "atom_label_wait"},
                source=NEURON_NAME,
                correlation_id=event.correlation_id,
                meta={"kind": "atom_label_pause"},
            ),
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["memory/atom"],
        output_topics=["thought/internal", "curiosity/adjust"],
        priority=8,
    )
    yield AtomResponderNeuron(cfg)