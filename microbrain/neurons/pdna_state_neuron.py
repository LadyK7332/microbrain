from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.hormone import derive_ddna_modulators
from microbrain.pdna.access import publish_profile_sections

NEURON_NAME = Path(__file__).stem


class PDNAStateNeuron(BaseNeuron):
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call ---
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        topic = event.topic
        if topic not in ("percept/text", "act/speech"):
            return []

        # Get PDNA store / profile from KV
        pdna_store = await ctx.get_kv("pdna:store", None)
        pdna_profile = await ctx.get_kv("pdna:profile", None)
        if pdna_store is None or pdna_profile is None:
            self.debug("no_pdna_store_found")
            return []

        # Crisis flag from GoalGuardianNeuron
        crisis_mode = bool(await ctx.get_kv("goals:crisis_mode", False))

        affectionate = False
        technical = False

        # Affect neuron may store:
        #   affect:last = {"salience": x, "valence": y, "arousal": z}
        affect = await ctx.get_kv("affect:last", None)
        if isinstance(affect, dict):
            try:
                sal = float(affect.get("salience", 0.0) or 0.0)
                val = float(affect.get("valence", 0.0) or 0.0)
            except (TypeError, ValueError):
                sal, val = 0.0, 0.0

            # Very rough heuristic:
            # positive + salient -> affectionate / social interaction
            if val > 0.3 and sal > 0.2:
                affectionate = True

        # Relation neuron may store:
        #   relation:last = {"state": ..., "interpersonal_tags": [...]}
        relation = await ctx.get_kv("relation:last", None)
        if isinstance(relation, dict):
            state = relation.get("state")
            if state in ("work", "technical", "learning"):
                technical = True

        # Update PDNA online.
        # PDNAProfile.register_interaction now also handles:
        # - focus
        # - energy
        # - support_level
        pdna_profile.register_interaction(
            crisis=crisis_mode,
            affectionate=affectionate,
            technical=technical,
        )

        # Persist PDNA to disk (v1: simple, every interaction)
        pdna_store.save()

        # Expose a snapshot and v2 profile organs for other neurons.
        await ctx.set_kv("pdna:last", pdna_profile.to_dict())
        await publish_profile_sections(ctx, pdna_profile)
        await ctx.set_kv("drive:ddna_modulators", derive_ddna_modulators(pdna_profile))

        self.debug(
            "pdna_updated",
            crisis=crisis_mode,
            affectionate=affectionate,
            technical=technical,
            interactions=pdna_profile.interactions,
        )

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
        ],
        output_topics=[],   # PDNA is internal state only
        priority=-2,        # after affect/relation, before goal-guardian if needed
    )
    yield PDNAStateNeuron(cfg)
