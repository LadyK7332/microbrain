from __future__ import annotations

import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

# Minimal, teachable grammar
# Examples:
#   "this is a car"
#   "that is an apple"
#   "car is a vehicle"           (concept -> concept)
#   "the car is red"             (entity property)
_RX_DET_ISA = re.compile(
    r"^(?P<det>this|that|the)\s+is\s+(?:a|an)\s+(?P<noun>[a-zA-Z][a-zA-Z0-9_-]{1,48})\s*$",
    re.IGNORECASE,
)
_RX_CONCEPT_ISA = re.compile(
    r"^(?P<x>[a-zA-Z][a-zA-Z0-9_-]{1,48})\s+is\s+(?:a|an)\s+(?P<y>[a-zA-Z][a-zA-Z0-9_-]{1,48})\s*$",
    re.IGNORECASE,
)
_RX_NP_IS_ADJ = re.compile(
    r"^(?:(?P<det>this|that|the)\s+)?(?P<noun>[a-zA-Z][a-zA-Z0-9_-]{1,48})\s+is\s+(?P<adj>[a-zA-Z][a-zA-Z0-9_-]{1,48})\s*$",
    re.IGNORECASE,
)

_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "brown"
}


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _sha16(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


class AtomSpeechNeuron(BaseNeuron):
    """
    Extracts simple "atom" facts from speech:
      - isa(ent, concept)          from "this/that/the is a NOUN"
      - isa(concept, concept)      from "car is a vehicle"
      - prop(ent, attr, value)     from "the car is red"

    Emits:
      topic: memory/atom
      payload: {schema:"atom.v1", atom_type:"isa|prop", subj:"...", pred/attr/value:"..."}
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/text":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {"text": event.payload}
        text = str(payload.get("text") or "").strip()
        if not text:
            return []

        channel = str(payload.get("channel", "default") or "default")
        source = str(payload.get("source", event.source or "unknown") or "unknown")
        now = time.time()
        s = _norm(text)

        atoms: List[Dict[str, Any]] = []

        # A) "this/that/the is a NOUN" -> isa(entity, concept:noun)
        m = _RX_DET_ISA.match(s)
        if m:
            det = m.group("det").lower()
            noun = m.group("noun").lower()
            ent = await self._resolve_entity(ctx, det=det, noun_hint=noun)
            atoms.append(self._atom_isa(subj=ent, pred=f"concept:{noun}", channel=channel, source=source, ts=now))

        # B) "car is a vehicle" -> isa(concept:car, concept:vehicle)
        else:
            m2 = _RX_CONCEPT_ISA.match(s)
            if m2:
                x = m2.group("x").lower()
                y = m2.group("y").lower()
                atoms.append(self._atom_isa(subj=f"concept:{x}", pred=f"concept:{y}", channel=channel, source=source, ts=now))

            # C) "the car is red" / "car is red" -> prop(entity, color/state, value)
            else:
                m3 = _RX_NP_IS_ADJ.match(s)
                if m3:
                    det = (m3.group("det") or "the").lower()
                    noun = m3.group("noun").lower()
                    adj = m3.group("adj").lower()

                    ent = await self._resolve_entity(ctx, det=det, noun_hint=noun)

                    # ensure entity type is attached too
                    atoms.append(self._atom_isa(subj=ent, pred=f"concept:{noun}", channel=channel, source=source, ts=now))

                    if adj in _COLOR_WORDS:
                        atoms.append(self._atom_prop(subj=ent, attr="attr:color", value=f"value:{adj}", channel=channel, source=source, ts=now))
                    else:
                        atoms.append(self._atom_prop(subj=ent, attr="attr:state", value=f"value:{adj}", channel=channel, source=source, ts=now))

        if not atoms:
            return []

        await ctx.set_kv("atoms:last", atoms)

        # Emit atoms (for a binder/logger neuron to persist)
        out = []
        for a in atoms:
            out.append(Event(topic="memory/atom", payload=a, source=NEURON_NAME, correlation_id=event.correlation_id))
        return out

    async def _resolve_entity(self, ctx, *, det: str, noun_hint: str) -> str:
        """
        Resolve an entity referent.
          - "this" => current focus entity (create if missing)
          - "that/the" => last entity for that noun type (else create)
        """
        focus = await ctx.get_kv("atoms:focus_ent", None)
        if isinstance(focus, str) and focus and det == "this":
            return focus

        last_map = await ctx.get_kv("atoms:last_ent_by_concept", {}) or {}
        if not isinstance(last_map, dict):
            last_map = {}

        ck = f"concept:{noun_hint}"
        if det in ("that", "the") and ck in last_map:
            ent = str(last_map.get(ck) or "")
            if ent:
                await ctx.set_kv("atoms:focus_ent", ent)
                return ent

        ent = f"ent:{_sha16('ent', ck, str(time.time()))}"
        last_map[ck] = ent
        await ctx.set_kv("atoms:last_ent_by_concept", last_map)
        await ctx.set_kv("atoms:focus_ent", ent)
        return ent

    def _atom_isa(self, *, subj: str, pred: str, channel: str, source: str, ts: float) -> Dict[str, Any]:
        return {
            "schema": "atom.v1",
            "atom_type": "isa",
            "subj": subj,
            "pred": pred,
            "ts": ts,
            "channel": channel,
            "source": source,
        }

    def _atom_prop(self, *, subj: str, attr: str, value: str, channel: str, source: str, ts: float) -> Dict[str, Any]:
        return {
            "schema": "atom.v1",
            "atom_type": "prop",
            "subj": subj,
            "attr": attr,
            "value": value,
            "ts": ts,
            "channel": channel,
            "source": source,
        }


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text"],
        output_topics=["memory/atom"],
        priority=6,
    )
    yield AtomSpeechNeuron(cfg)