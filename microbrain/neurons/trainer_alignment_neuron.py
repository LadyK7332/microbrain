from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class TrainerAlignmentNeuron(BaseNeuron):
    """
    Guided utterance correction lane.

    /t arms trainer mode in input_text.
    The next user line becomes a correction payload on control/trainer_correction.

    This neuron then:
      - splits alternates on '|'
      - ingests each alternate as assistant text (so it still gets decomposed)
      - stores a trainer-alignment cell keyed by the prior utterance situation
      - appends strong semantic/episodic memory for future recall bias
    """

    def _split_alternatives(self, raw: str) -> List[str]:
        parts = [str(p or "").strip() for p in str(raw or "").split("|")]
        seen: set[str] = set()
        out: List[str] = []
        for part in parts:
            if not part:
                continue
            norm = " ".join(part.lower().split())
            if norm in seen:
                continue
            seen.add(norm)
            out.append(part)
            if len(out) >= 8:
                break
        return out

    def _context_query(self, target: Dict[str, Any]) -> str:
        if not isinstance(target, dict):
            return ""
        existing = str(target.get("context_query", "") or "").strip()
        if existing:
            return existing
        parts = [
            str(target.get("need", "") or "").strip(),
            str(target.get("style", "") or "").strip(),
            str(target.get("message", "") or "").strip(),
            str(target.get("utterance", "") or "").strip(),
        ]
        return " | ".join([p for p in parts if p])

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "control/trainer_correction":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        correction_text = str(payload.get("correction_text", "") or "").strip()
        if not correction_text:
            return []

        target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
        bad_utterance = str(target.get("utterance", "") or "").strip()
        need = str(target.get("need", "") or "").strip()
        style = str(target.get("style", "") or "").strip()
        message = str(target.get("message", "") or "").strip()
        context_query = self._context_query(target)
        if not context_query:
            context_query = bad_utterance or message or correction_text

        mem_store = await ctx.get_kv("memory:store", None)
        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        if mem_cell_store is None:
            try:
                memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
                mem_cell_store = MemCellStore(memdir)
                await ctx.set_kv("memory:mem_cell_store", mem_cell_store)
            except Exception:
                mem_cell_store = None

        alternatives = self._split_alternatives(correction_text)
        if not alternatives:
            return []

        stored: List[Dict[str, Any]] = []
        ts_now = time.time()
        for idx, desired in enumerate(alternatives, start=1):
            meta = {
                "kind": "trainer_correction",
                "trainer_need": need,
                "trainer_style": style,
                "trainer_bad_utterance": bad_utterance,
                "trainer_context": context_query,
                "trainer_message": message,
                "trainer_alt_index": idx,
                "trainer_alt_count": len(alternatives),
                "trainer_ts": ts_now,
            }

            if mem_store is not None:
                try:
                    sal = {
                        "score": 0.85,
                        "valence": 0.20,
                        "satisfaction": 0.35,
                        "arousal": 0.08,
                        "reinforce_sum": 3.0,
                        "reinforce_count": 1,
                        "last_reinforced_ts": float(ts_now),
                    }
                    mem_store.add_semantic(desired, {"role": "assistant", **meta}, salience=sal)
                    mem_store.add_episodic(f"TRAINER: {desired}", {"role": "system", **meta}, salience=sal)
                except Exception:
                    pass

            alignment_id = ""
            if isinstance(mem_cell_store, MemCellStore):
                try:
                    result = mem_cell_store.ingest_trainer_alignment(
                        desired_text=desired,
                        context_query=context_query,
                        bad_utterance=bad_utterance,
                        need=need,
                        style=style,
                        source=self.name,
                        meta=meta,
                        tier="learned",
                    )
                    alignment_id = str(((result.get("alignment", {}) or {}).get("id", "") or "")).strip()
                except Exception:
                    alignment_id = ""

            stored.append({
                "desired": desired,
                "alignment_id": alignment_id,
            })

        await ctx.set_kv(
            "trainer:last_correction",
            {
                "ts": ts_now,
                "count": len(stored),
                "bad_utterance": bad_utterance,
                "context_query": context_query,
                "need": need,
                "style": style,
                "alternatives": stored,
            },
        )

        label = "alternative" if len(stored) == 1 else "alternatives"
        return [
            Event(
                topic="act/speech",
                payload={
                    "text": f"Trainer aligned {len(stored)} {label}.",
                    "style": "system",
                    "channel": "default",
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"control": True, "kind": "trainer_correction_ack"},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["control/trainer_correction"],
        output_topics=["act/speech"],
        priority=9,
        cooldown_sec=0.0,
    )
    yield TrainerAlignmentNeuron(cfg)
