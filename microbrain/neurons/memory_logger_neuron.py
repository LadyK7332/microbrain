from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


NEURON_NAME = Path(__file__).stem


class MemoryLoggerNeuron(BaseNeuron):
    """
    Bridge between the orchestrator event flow and the legacy JSONL memories.

    - On 'percept/text': logs USER turns to episodic (+ optional semantic) + emotion_journal
    - On 'act/speech':  logs ASSISTANT turns to episodic + semantic + emotion_journal
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        # Grab the memory objects injected by mind.py
        mem_store = await ctx.get_kv("memory:store", None)
        ejournal = await ctx.get_kv("memory:emotion_journal", None)

        # If nothing is wired, bail out quietly
        if mem_store is None and ejournal is None:
            return []

        topic = event.topic
        payload = event.payload

        # Skip control/system helper messages (menus, debug UI, etc.)
        # These are useful operationally, but should not be stored as long-term memory.
        if (event.meta or {}).get("control"):
            return []


        # Normalize text + role depending on topic
        text: str = ""
        role: str = "user"

        if topic == "percept/text":
            # TextInputNeuron always passes a dict payload with 'text'
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                role = str(payload.get("source", "user"))
            else:
                text = str(payload).strip()
                role = "user"

        elif topic == "act/speech":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                # style 'assistant' vs 'system' etc; treat non-user as assistant-ish
                style = str(payload.get("style", "assistant"))
                role = "assistant" if style != "system" else "system"
            else:
                text = str(payload).strip()
                role = "assistant"

        # Nothing meaningful to log
        if not text:
            return []

        ts = int(time.time())

        # --- Write to MemoryStore ------------------------------------------------
        try:
            if mem_store is not None:
                # Preserve meta so recall can filter system/control noise later
                meta_base: Dict[str, Any] = {"role": role}
                if event.meta:
                    for k in ("kind", "control", "channel", "source"):
                        if k in event.meta:
                            meta_base[k] = event.meta[k]

                if topic == "percept/text":
                    # Old Agent.step only wrote episodic for user input
                    mem_store.add_episodic(f"USER: {text}", meta_base)
                elif topic == "act/speech":
                    # Old Agent.step wrote both semantic + episodic for assistant
                    mem_store.add_semantic(text, meta_base)
                    mem_store.add_episodic(f"ASSISTANT: {text}", meta_base)
        except Exception as e:
            # Keep logging failures from killing the brain
            self.debug("mem_store_error", error=str(e))

        # --- Write to EmotionJournal --------------------------------------------
        try:
            if ejournal is not None:
                entry: Dict[str, Any] = {
                    "actor": "assistant" if role != "user" else "user",
                    "text": text,
                    # neutral defaults for now; your affect neuron can refine later
                    "valence": 0.0,
                    "arousal": 0.0,
                    "salience": 0.2 if role == "user" else 0.3,
                    "tags": ["auto"],
                    "ts": ts,
                }
                ejournal.append(entry)
        except Exception as e:
            self.debug("ejournal_error", error=str(e))

        # This neuron is a side-effect logger only; no new events
        return []


def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook for memory logging.
    """
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
        ],
        output_topics=[],
        # Run fairly late, but *before* speech_output (which has priority -10)
        priority=-5,
    )
    yield MemoryLoggerNeuron(cfg)
