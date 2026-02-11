"""
HRM Observer Neuron

Bridges MicroBrain's event stream into the HRMCore concept graph.

Listens to:
    - percept/text  -> USER nodes
    - act/speech    -> ASSISTANT nodes

Each message becomes:
    - a concept node w/ embedding
    - Hebbian updates to synapse graph (via HRMCore)
    - affect & relation tags attached when available

This neuron is foundational for PDNA, persona shaping, 
alignment stabilization, and coherent identity drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class HRMObserverNeuron(BaseNeuron):
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
        payload = event.payload

        if topic not in ("percept/text", "act/speech"):
            return []

        # Skip system/control messages (menus, debug, etc.)
        if event.meta.get("control"):
            return []

        # ------------------------------
        # Normalize text + role
        # ------------------------------
        text = ""
        role = "user"

        if topic == "percept/text":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                role = "user"
            else:
                text = str(payload).strip()
                role = "user"

        elif topic == "act/speech":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                style = str(payload.get("style", "assistant"))
                role = "assistant" if style != "system" else "system"
            else:
                text = str(payload).strip()
                role = "assistant"

        if not text:
            return []

        # ------------------------------
        # Pull HRM Core from orchestrator
        # ------------------------------
        hrm = await ctx.get_kv("hrm:core", None)
        if hrm is None:
            self.debug("no_hrm_core_found")
            return []

        #Internal ECHO cancelation for "Experience."
        if text.startswith("[echo:"):
            return []

        # ------------------------------
        # Create a new HRM node
        # ------------------------------
        try:
            node = hrm.observe(text, role=role, meta=event.meta)
        except Exception as exc:
            # DO NOT swallow this; it is why you "see input but no node"
            self.debug(
                "hrm_observe_error",
                error=repr(exc),
                role=role,
                text_preview=text[:80],
                meta=event.meta,
            )

        self.debug(
            "hrm_node_created",
            idx=getattr(node, "idx", None),
            role=role,
            text_preview=text[:80],
        )

        # Expose last HRM node index so other neurons can query neighbors
        if getattr(node, "idx", None) is not None:
            await ctx.set_kv("hrm:last_idx", node.idx)
            self.debug("hrm_last_idx_set", idx=node.idx)
        else:
            self.debug("hrm_node_missing_idx", role=role, text_preview=text[:80])
            return []

        self.debug(
            "hrm_node_created",
            idx=node.idx,
            role=role,
            text_preview=text[:60],
        )

        # ------------------------------
        # Attach affect (if exist in kv)
        # ------------------------------
        affect = await ctx.get_kv("affect:last", None)
        if isinstance(affect, dict):
            hrm.set_affect(
                node.idx,
                salience=affect.get("salience"),
                valence=affect.get("valence"),
                arousal=affect.get("arousal"),
            )

        # ------------------------------
        # Attach relation tags (if exist in kv)
        # ------------------------------
        relation = await ctx.get_kv("relation:last", None)
        if isinstance(relation, dict):
            hrm.tag(
                node.idx,
                state=relation.get("state"),
                interpersonal=relation.get("interpersonal_tags"),
            )

        # ------------------------------
        # Attach PDNA snapshot (personality state at this moment)
        # ------------------------------
        pdna_snapshot = await ctx.get_kv("pdna:last", None)
        if isinstance(pdna_snapshot, dict):
            try:
                hrm.tag(
                    node.idx,
                    pdna=pdna_snapshot,
                )
                self.debug("hrm_pdna_tagged", idx=node.idx)
            except Exception as exc:
                self.debug("hrm_pdna_tag_error", error=str(exc))

        # Expose last HRM node index so other neurons (e.g. reasoner) can query neighbors
        await ctx.set_kv("hrm:last_idx", node.idx)

        return []
        

def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
        ],
        output_topics=[],   # HRM is internal only
        priority=-3,        # after affect/relation, before goal-guardian if needed
    )
    yield HRMObserverNeuron(cfg)
