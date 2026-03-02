from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem

ALLOWED_PREFIXES = ("rt/", "percept/", "act/", "plan/", "input/")


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_token(memdir: Path) -> str:
    # Use your existing memory-root token file (you’ve already got this)
    p = memdir / "ipc_token.txt"
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


class IpcInboxNeuron(BaseNeuron):
    """
    File mailbox IPC:
      - reads memdir/ipc/inbox/*.json
      - validates auth token and topic prefix
      - emits bus Events with the embedded topic/payload
      - deletes processed files

    This keeps vision/audio/etc as separate programs without opening sockets.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "clock/tick":
            return []

        memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
        token = _read_token(memdir)
        if not token:
            # If token missing, do nothing (safer than accepting unauthenticated messages)
            self.debug("ipc_no_token", memdir=str(memdir))
            return []

        inbox = memdir / "ipc" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        max_files = _safe_int(await ctx.get_kv("ipc:inbox_max_files", 25), 25)
        quarantine = memdir / "ipc" / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)

        files = sorted(inbox.glob("*.json"))
        if not files:
            return []

        out: List[Event] = []
        now = time.time()

        for p in files[:max_files]:
            try:
                raw = p.read_text(encoding="utf-8-sig", errors="strict")
                # extra safety in case it came in as a literal BOM char
                raw = raw.lstrip("\ufeff")
                msg = json.loads(raw)
            except Exception:
                # bad file -> quarantine
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            if not isinstance(msg, dict):
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            # auth gate
            if msg.get("auth") != token:
                self.debug("ipc_auth_fail", file=str(p))
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            topic = msg.get("topic")
            if not isinstance(topic, str) or not topic or not topic.startswith(ALLOWED_PREFIXES):
                self.debug("ipc_topic_fail", file=str(p), topic=repr(topic))
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            payload = msg.get("payload")
            src = str(msg.get("src") or "ipc")
            corr = str(msg.get("correlation_id") or "")
            ts = msg.get("timestamp")
            meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}

            # Emit into bus
            out.append(
                Event(
                    topic=topic,
                    payload=payload,
                    source=src,
                    correlation_id=corr or event.correlation_id,
                    timestamp=float(ts) if isinstance(ts, (int, float)) else now,
                    meta={"via": "ipc_inbox", **meta},
                )
            )

            # record for debug (KV)
            await ctx.set_kv("ipc:last_msg", {"topic": topic, "src": src, "ts": now})

            # delete after success
            try:
                p.unlink()
            except Exception:
                pass

        if out:
            self.debug("ipc_ingested", count=len(out), first_topic=out[0].topic)

        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick"],
        output_topics=[],
        priority=5,
        cooldown_sec=0.25,
    )
    yield IpcInboxNeuron(cfg)