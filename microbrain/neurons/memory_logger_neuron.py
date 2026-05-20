from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Any, Dict

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.memory.filters import classify_event_for_memory


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
        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        if mem_cell_store is None:
            try:
                memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
                mem_cell_store = MemCellStore(memdir)
                await ctx.set_kv("memory:mem_cell_store", mem_cell_store)
            except Exception as e:
                self.debug("mem_cell_store_error", error=str(e))
                mem_cell_store = None

        # If nothing is wired, bail out quietly
        if mem_store is None and ejournal is None and mem_cell_store is None:
            return []

        topic = event.topic
        payload = event.payload

        # Skip control/UI-plane helper messages (menus, command confirmations, errors, debug UI, etc.).
        # These are useful for the operator, but must not become cognition-plane memory.
        mem_class = classify_event_for_memory(event)
        if topic != "control/reinforce" and not mem_class.get("allow_longterm", False):
            return []

        trainer_pending = bool(await ctx.get_kv("control:t_pending", False))
        if trainer_pending and topic == "act/speech" and not (event.meta or {}).get("control"):
            return []

        # Persist reinforcement events emitted by /r +/- (append-only; no rewriting old memory rows)
        if topic == "control/reinforce":
            if mem_store is not None and isinstance(payload, dict):
                try:
                    w = int(payload.get("weight", 0) or 0)
                except Exception:
                    w = 0
                w = max(-5, min(5, w))
                scale = w / 5.0

                target_role = str(payload.get("target_role", "assistant") or "assistant")
                if target_role not in ("user", "assistant"):
                    target_role = "assistant"

                target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
                target_text = str(target.get("text", "") or "").strip()
                nonce = str(payload.get("nonce", "") or "")
                hrm_idx = target.get("hrm_idx", None)

                # Always log the reinforcement action (low-trust internal trace)
                meta_evt: Dict[str, Any] = {
                    "role": "system",
                    "schema_ver": 2,
                    "kind": "reinforcement_event",
                    "reinforce_weight": w,
                    "reinforce_nonce": nonce,
                    "reinforce_target_role": target_role,
                    "reinforce_target_hrm_idx": hrm_idx,
                }
                mem_store.add_episodic(
                    f"REINFORCE {w:+d} [{target_role}] {target_text[:180]}",
                    meta_evt,
                )

                # If we have a real target text, append a *reinforced mirror* semantic item
                if target_text:
                    sal = {
                        "score": max(0.0, 0.15 + 0.25 * max(0.0, scale)),
                        "valence": 0.6 * scale,
                        "satisfaction": 0.8 * scale,
                        "arousal": 0.15 * abs(scale),
                        "reinforce_sum": float(w),
                        "reinforce_count": 1,
                        "last_reinforced_ts": float(time.time()),
                    }
                    meta_r: Dict[str, Any] = {
                        "role": target_role,
                        "schema_ver": 2,
                        "kind": "reinforced",
                        "reinforce_weight": w,
                        "reinforce_nonce": nonce,
                        "reinforce_target_hrm_idx": hrm_idx,
                    }
                    # NOTE: Step 3 will add salience= support to MemoryStore methods
                    mem_store.add_semantic(target_text, meta_r, salience=sal)
            return []


        # Normalize text + role depending on topic
        text: str = ""
        role: str = "user"
        transport_source: str = "user"

        if topic == "percept/text":
            # TextInputNeuron always passes a dict payload with 'text'
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()

                # Canonical role (never allow "ui" to become a persona/role in memory)
                role = str(payload.get("source", "user") or "user")

                raw_meta = payload.get("raw_meta", {}) or {}
                transport_source = str(raw_meta.get("transport_source", role) or role)

                if role not in ("user", "assistant", "system"):
                    role = "user"
            else:
                text = str(payload).strip()
                role = "user"
                transport_source = "unknown"

        elif topic == "act/speech":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                # style 'assistant' vs 'system' etc; treat non-user as assistant-ish
                style = str(payload.get("style", "assistant"))
                role = "assistant" if style != "system" else "system"
            else:
                text = str(payload).strip()
                role = "assistant"

            transport_source = str(event.source or "assistant")
            if role == "assistant" and text:
                await ctx.set_kv(
                    "trainer:last_assistant_utterance",
                    {
                        "ts": time.time(),
                        "text": text,
                        "source": str(event.source or "assistant"),
                        "meta": dict(event.meta or {}),
                    },
                )

        # Nothing meaningful to log
        if not text:
            return []

        ts = int(time.time())

        # --- Write to MemoryStore ------------------------------------------------
        try:
            if mem_store is not None:
                # Preserve meta so recall can filter system/control noise later
                meta_base: Dict[str, Any] = {"role": role, "schema_ver": 2, "transport_source": transport_source}
                if event.meta:
                    for k in ("kind", "control", "channel", "source"):
                        if k in event.meta:
                            meta_base[k] = event.meta[k]

                if topic == "percept/text":
                    # Old Agent.step only wrote episodic for user input
                    mem_store.add_episodic(f"USER: {text}", meta_base)
                elif topic == "act/speech":
                    # Old Agent.step wrote both semantic + episodic for assistant
                    sal = None
                    kind = str(meta_base.get("kind", "") or "")

                    if kind in ("curiosity_babble", "router_reply"):
                        # Mandatory negative bias for unreinforced chatter / canned router replies
                        sal = {"score": 0.0, "valence": -0.15, "satisfaction": -0.25, "arousal": 0.05}

                    mem_store.add_semantic(text, meta_base, salience=sal)
                    mem_store.add_episodic(f"ASSISTANT: {text}", meta_base, salience=sal)
                    
        except Exception as e:
            # Keep logging failures from killing the brain
            self.debug("mem_store_error", error=str(e))

        # --- Write to mem_cell/now -----------------------------------------------
        try:
            if mem_cell_store is not None and role in ("user", "assistant"):
                ingest_result = mem_cell_store.ingest_text(
                    text=text,
                    topic=topic,
                    role=role,
                    transport_source=transport_source,
                    source=str(event.source or role),
                    meta=dict(event.meta or {}),
                    tier="now",
                )
                await ctx.set_kv("memory:last_memcell_ingest", {
                    "utterance_id": str((ingest_result.get("utterance", {}) or {}).get("id", "") or ""),
                    "token_ids": [str((c or {}).get("id", "") or "") for c in ingest_result.get("tokens", [])],
                    "pattern_ids": [str((c or {}).get("id", "") or "") for c in ingest_result.get("patterns", [])],
                    "general_patterns": [str((c or {}).get("id", "") or "") for c in ingest_result.get("general_patterns", [])],
                    "linker_ids": [str((c or {}).get("id", "") or "") for c in ingest_result.get("linkers", [])],
                })
        except Exception as e:
            self.debug("mem_cell_error", error=str(e))

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
            "control/reinforce",
        ],
        output_topics=[],
        # Run fairly late, but *before* speech_output (which has priority -10)
        priority=-5,
    )
    yield MemoryLoggerNeuron(cfg)
