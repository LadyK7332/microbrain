from __future__ import annotations

import json
import os
import re
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.memory.filters import classify_event_for_memory

NEURON_NAME = Path(__file__).stem


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-']{0,63}")


@dataclass
class RecentItem:
    ts: float
    payload: Any
    source: str
    meta: Dict[str, Any]


class ConceptBinderNeuron(BaseNeuron):
    """
    Cross-modal co-occurrence binder.

    Goal:
      - Observe recent percepts (text/vision/audio/touch).
      - When a likely label arrives (e.g. user says "apple"),
        bind it to recent percept(s) within a time window.
      - DO NOT write every time; only write after 'confirm' repeats.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # Debug roll-call (only active when --debug is passed)
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        # ---- Tunables (can be promoted to config later) ----
        window_sec = float(await ctx.get_kv("concept:window_sec", 5.0) or 5.0)
        confirm = int(await ctx.get_kv("concept:confirm", 4) or 4)
        max_recent_text = int(await ctx.get_kv("concept:max_recent_text", 25) or 25)

        # ---- Load state ----
        recents: Dict[str, Any] = await ctx.get_kv("concept:recents", {}) or {}
        pending: Dict[str, Any] = await ctx.get_kv("concept:pending", {}) or {}
        committed: Dict[str, Any] = await ctx.get_kv("concept:committed", {}) or {}

        now = time.time()

        # ---- Update recent buffers ----
        if event.topic == "percept/text":
            guard = classify_event_for_memory(event)
            if not guard.get("allow_pattern", False):
                return []
            text = self._normalize_text(event.payload)
            if text:
                items: List[Dict[str, Any]] = recents.get("text", []) or []
                items.append({"ts": now, "text": text, "source": event.source, "meta": dict(event.meta or {})})
                if len(items) > max_recent_text:
                    items = items[-max_recent_text:]
                recents["text"] = items

                # If this looks like a label, attempt binding immediately.
                if self._looks_like_label(text):
                    self._try_bind_label(
                        label=text,
                        now=now,
                        window_sec=window_sec,
                        recents=recents,
                        pending=pending,
                        committed=committed,
                        confirm=confirm,
                        ctx=ctx,
                    )

        elif event.topic in ("percept/vision", "percept/audio", "percept/touch"):
            # Keep only the latest item per channel (cheap STM)
            recents[event.topic] = {
                "ts": now,
                "payload": event.payload,
                "source": event.source,
                "meta": dict(event.meta or {}),
            }

        # Persist updated STM + counters
        await ctx.set_kv("concept:recents", recents)
        await ctx.set_kv("concept:pending", pending)
        await ctx.set_kv("concept:committed", committed)

        return []

    # ----------------------------
    # Binding logic
    # ----------------------------
    def _try_bind_label(
        self,
        label: str,
        now: float,
        window_sec: float,
        recents: Dict[str, Any],
        pending: Dict[str, Any],
        committed: Dict[str, Any],
        confirm: int,
        ctx,
    ) -> None:
        # Gather any recent sensory items in the time window
        bindings: Dict[str, Dict[str, Any]] = {}
        for topic in ("percept/vision", "percept/audio", "percept/touch"):
            item = recents.get(topic)
            if not isinstance(item, dict):
                continue
            if (now - float(item.get("ts", 0.0))) <= window_sec:
                bindings[topic] = item

        if not bindings:
            return

        # Create a stable key: label + signatures of each bound channel
        sig_parts = [f"label={label}"]
        packed: Dict[str, Any] = {"label": label, "ts": now, "bindings": {}}

        for topic, item in bindings.items():
            payload = item.get("payload")
            sig = self._sig_for_payload(payload)
            sig_parts.append(f"{topic}={sig}")
            packed["bindings"][topic] = {
                "sig": sig,
                "payload": payload,
                "source": item.get("source", ""),
                "meta": item.get("meta", {}) or {},
                "age_sec": max(0.0, now - float(item.get("ts", now))),
            }

        key = "|".join(sig_parts)

        # If we already committed this exact association, don't spam.
        if key in committed:
            return

        # Count repeats (RAM/KV only)
        entry = pending.get(key) or {"hits": 0, "first_ts": now, "last_ts": now}
        entry["hits"] = int(entry.get("hits", 0)) + 1
        entry["last_ts"] = now
        pending[key] = entry

        # Only commit to disk when it crosses confirm threshold
        if entry["hits"] < confirm:
            return

        # Commit once
        committed[key] = {"committed_ts": now, "hits": entry["hits"], "first_ts": entry.get("first_ts", now)}
        pending.pop(key, None)

        # Write to memdir/concepts/pending.jsonl
        self._append_concept_row(ctx, packed, key=key, hits=entry["hits"])

    def _append_concept_row(self, ctx, row: Dict[str, Any], key: str, hits: int) -> None:
        # Resolve memdir via MemoryStore if present, else fallback.
        memdir = None
        try:
            # ctx.get_kv is async, but this method is sync; we use a safe fallback path.
            # We will firm this up in the next step by making the writer fully async.
            pass
        except Exception:
            pass

        # Fallbacks that match your current setup
        memdir = os.getenv("MB_MEMDIR") or r"Z:\memory"

        out_dir = Path(memdir) / "concepts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pending.jsonl"

        payload = dict(row)
        payload["key"] = key
        payload["confirm_hits"] = hits
        payload["schema"] = "concept.cooccur.v1"

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _normalize_text(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            t = payload.strip().lower()
        else:
            t = str(payload).strip().lower()
        return t

    def _looks_like_label(self, text: str) -> bool:
        # Strict on purpose: keeps random chatter from becoming "labels".
        # Single word, mostly alnum, small.
        words = _WORD_RE.findall(text)
        return len(words) == 1 and words[0] == text and 1 <= len(text) <= 24

    def _sig_for_payload(self, payload: Any) -> str:
        # Try to find an obvious stable identity first
        if isinstance(payload, dict):
            for k in ("path", "file", "filepath", "frame_path", "image_path", "ref", "id"):
                v = payload.get(k)
                if isinstance(v, str) and v:
                    return f"{k}:{Path(v).name}"
        # Fallback: hash a short representation
        s = repr(payload)
        if len(s) > 800:
            s = s[:800]
        h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"sha1:{h}"


def build_neurons(orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "percept/vision",  # already used elsewhere in your stack
            "percept/audio",   # produced by cochlear_neuron
            "percept/touch",   # reserved for later tactile wiring
        ],
        output_topics=[],
        priority=4,  # after raw percepts exist, before most “thinking” cascades
    )
    yield ConceptBinderNeuron(cfg)