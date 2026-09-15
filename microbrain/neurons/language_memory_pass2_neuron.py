from __future__ import annotations

"""Second-pass language memory writer.

This neuron builds the segmented token/phrase graph while keeping parser
guesses weak and trainer corrections strong.
"""

from pathlib import Path
from typing import Iterable, Mapping

from microbrain.memory.segmented_language_memory import SegmentedLanguageMemory
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem
STORE_KV_KEY = "memory:v2_language_store"
LAST_PARSE_KV_KEY = "language:pass2:last_parse"
LAST_TRAINER_KV_KEY = "language:pass2:last_trainer_update"


class LanguageMemoryPass2Neuron(BaseNeuron):
    async def _store(self, ctx) -> SegmentedLanguageMemory | None:
        store = await ctx.get_kv(STORE_KV_KEY, None)
        if isinstance(store, SegmentedLanguageMemory):
            return store
        try:
            memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
            store = SegmentedLanguageMemory(memdir)
            await ctx.set_kv(STORE_KV_KEY, store)
            return store
        except Exception as exc:
            await ctx.log_warn(f"[{self.name}] segmented language store unavailable", error=repr(exc))
            return None

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in {"percept/text", "act/speech", "control/trainer_correction"}:
            return []

        store = await self._store(ctx)
        if store is None:
            return []

        if event.topic == "control/trainer_correction":
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            target = payload.get("target", {}) if isinstance(payload.get("target"), Mapping) else {}
            correction_text = str(payload.get("correction_text", "") or "").strip()
            if not correction_text:
                return []
            context = str(target.get("context_query", "") or target.get("message", "") or target.get("utterance", "") or "").strip()
            update = store.apply_trainer_correction(
                correction_text=correction_text,
                context_text=context,
                bad_utterance=str(target.get("utterance", "") or ""),
                need=str(target.get("need", "") or ""),
                style=str(target.get("style", "") or ""),
                source=self.name,
                meta={"event_source": event.source, "correlation_id": event.correlation_id},
            )
            await ctx.set_kv(LAST_TRAINER_KV_KEY, update)
            return [
                Event(
                    topic="language/pass2_trainer_update",
                    payload=update,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "language_pass2_trainer_update", "store_in_memory": False},
                )
            ]

        if event.topic == "act/speech" and bool((event.meta or {}).get("control", False)):
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {"text": event.payload}
        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        if event.topic == "percept/text":
            raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), Mapping) else {}
            role = str(payload.get("source", "user") or "user")
            if role not in {"user", "assistant", "system"}:
                role = "user"
            source = str(raw_meta.get("transport_source", role) or role)
        else:
            role = "assistant"
            source = str(event.source or "assistant")

        parsed = store.observe_text(
            text,
            role=role,
            source=source,
            strength="observed",
            meta={"event_topic": event.topic, "event_source": event.source},
        )
        await ctx.set_kv(LAST_PARSE_KV_KEY, parsed)
        return [
            Event(
                topic="language/pass2_parse",
                payload=parsed,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "language_pass2_parse", "store_in_memory": False},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
            "control/trainer_correction",
        ],
        output_topics=[
            "language/pass2_parse",
            "language/pass2_trainer_update",
        ],
        priority=-4,
        cooldown_sec=0.0,
    )
    yield LanguageMemoryPass2Neuron(cfg)
