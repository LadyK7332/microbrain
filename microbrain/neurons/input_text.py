from __future__ import annotations

import time
import uuid

from typing import Iterable, Any, Dict, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class TextInputNeuron(BaseNeuron):
    """
    First-stop neuron for incoming text.

    Listens on:
        - "input/text"

    Emits:
        - "percept/text" with a normalized payload:
            {
                "text": <str>,
                "source": <str>,   # e.g. "user", "ui", "minecraft"
                "channel": <str>,  # e.g. "cli", "webui", "discord"
                "raw_meta": {...}, # merged view of any extra metadata
            }

    This keeps the rest of the system talking in a consistent shape,
    regardless of how external systems format their text messages.
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

        # ----------------------------------------------
        # 1) Extract text + side metadata from payload
        # ----------------------------------------------
        text: str
        extra_meta: Dict[str, Any]

        if isinstance(event.payload, str):
            text = event.payload
            extra_meta = {}
        elif isinstance(event.payload, dict):
            # Common shape: {"text": "...", "source": "...", "channel": "...", ...}
            text = str(event.payload.get("text", ""))
            extra_meta = {k: v for k, v in event.payload.items() if k != "text"}
        else:
            # Fallback: stringify whatever was handed to us
            text = str(event.payload)
            extra_meta = {}

        text_norm = text.strip()
        if not text_norm:
            # Don't generate percepts for empty/whitespace-only input
            await ctx.log_debug(
                f"[{self.name}] Ignoring empty input payload",
                topic=event.topic,
            )
            return []

        # ----------------------------------------------
        # 2) Derive source/channel & merge metadata
        # ----------------------------------------------
        # Event meta wins over payload meta if both provide the same key.
        merged_meta: Dict[str, Any] = {}
        merged_meta.update(extra_meta)
        merged_meta.update(event.meta)

        source = merged_meta.get("source", "user")
        channel = merged_meta.get("channel", "default")

        # ----------------------------------------------
        # 2.5) Reinforcement snapshot latch (/r ...)
        # ----------------------------------------------
        r_pending = bool(await ctx.get_kv("control:r_pending", False))

        # If a /r menu is open, refuse non-/r input until it is resolved.
        if r_pending and not text_norm.startswith("/r"):
            return [
                self._speech_control(
                    "Reinforcement menu is still open. Use `/r +3 2`, `/r -2 4`, or `/r clear`.",
                    channel=channel,
                    correlation_id=event.correlation_id,
                )
            ]

        # Handle /r commands here so they don't become percept/text (no HRM/memory pollution).
        if text_norm.startswith("/r"):
            return await self._handle_r_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )


        # ----------------------------------------------
        # 3) Construct normalized percept payload
        # ----------------------------------------------
        percept_payload: Dict[str, Any] = {
            "text": text_norm,
            "source": source,
            "channel": channel,
            "raw_meta": merged_meta,
        }

        # Optionally: PDNA hints could be attached here later based on channel/source.

        percept_event = Event(
            topic="percept/text",
            payload=percept_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "percept",
                "modality": "text",
                "normalized": True,
            },
        )

        await ctx.log_debug(
            f"[{self.name}] Emitted percept/text",
            source=source,
            channel=channel,
        )

        return [percept_event]

    # ------------------------------------------------------------------
    # /r reinforcement: ephemeral snapshot menu + apply weight + clear
    # ------------------------------------------------------------------
    async def _handle_r_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        """
        Supported:
          /r u 5        -> show last 5 USER items (snapshot opens, MB waits)
          /r a 5        -> show last 5 ASSISTANT items (snapshot opens, MB waits)
          /r +3 2       -> apply +3 to snapshot index #2 (then snapshot clears, resume)
          /r -5 1       -> apply -5 to snapshot index #1 (then snapshot clears, resume)
          /r clear      -> clear snapshot and resume
        """
        line = (cmd_text or "").strip()
        parts = line.split()

        # parts[0] is "/r"
        if len(parts) == 1:
            return [
                self._speech_control(
                    "Usage:\n  /r u 5   (last 5 user)\n  /r a 5   (last 5 assistant)\n  /r +3 2  (score index)\n  /r clear",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        sub = parts[1].lower()

        # /r clear
        if sub in ("clear", "c", "reset"):
            await ctx.set_kv("control:r_snapshot", None)
            await ctx.set_kv("control:r_pending", False)
            await ctx.set_kv("attention:allow_babble", True)
            return [
                self._speech_control(
                    "Reinforcement snapshot cleared. Resuming.",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        # /r u N  OR  /r a N
        if sub in ("u", "a"):
            want_role = "user" if sub == "u" else "assistant"
            n = 5
            if len(parts) >= 3:
                try:
                    n = int(parts[2])
                except Exception:
                    n = 5
            n = max(1, min(20, n))

            hrm = await ctx.get_kv("hrm:core", None)
            items = self._hrm_recent_items(hrm=hrm, want_role=want_role, n=n) if hrm else []

            snap = {
                "nonce": uuid.uuid4().hex[:8],
                "role": want_role,
                "created_ts": time.time(),
                "items": items,
            }
            await ctx.set_kv("control:r_snapshot", snap)
            await ctx.set_kv("control:r_pending", True)

            # Hard pause: stop curiosity babble while menu is open
            await ctx.set_kv("attention:allow_babble", False)
            await ctx.set_kv("attention:focus_target", "external")

            if not items:
                return [
                    self._speech_control(
                        f"No recent {want_role} items found. Snapshot open anyway: use `/r clear` to exit.",
                        channel=channel,
                        correlation_id=correlation_id,
                    )
                ]

            lines: List[str] = []
            lines.append(f"Reinforcement snapshot [{want_role}] nonce={snap['nonce']}")
            for i, it in enumerate(items, start=1):
                preview = (it.get('text', '') or '').replace('\n', ' ').strip()
                if len(preview) > 90:
                    preview = preview[:90] + "…"
                lines.append(f"{i}) idx={it.get('hrm_idx')}  {preview}")
            lines.append("")
            lines.append("Score one item:")
            lines.append("  /r +3 2   or   /r -2 4")
            lines.append("  /r clear  (exit without scoring)")

            return [
                self._speech_control(
                    "\n".join(lines),
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        # /r +W I   or   /r -W I
        try:
            weight = int(sub)  # works for "+3" and "-2"
        except Exception:
            weight = None

        if weight is None:
            return [
                self._speech_control(
                    "Unknown /r command. Try `/r u 5`, `/r a 5`, `/r +3 2`, or `/r clear`.",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        weight = max(-5, min(5, weight))

        if len(parts) < 3:
            return [
                self._speech_control(
                    "Missing index. Example: `/r +3 2`",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        try:
            which = int(parts[2])
        except Exception:
            which = -1

        snap = await ctx.get_kv("control:r_snapshot", None)
        items = snap.get("items", []) if isinstance(snap, dict) else []
        if not items:
            return [
                self._speech_control(
                    "No active /r snapshot. Run `/r u 5` or `/r a 5` first.",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        if which < 1 or which > len(items):
            return [
                self._speech_control(
                    f"Index out of range. Pick 1..{len(items)}.",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        target = items[which - 1]
        await ctx.set_kv(
            "reinforce:last",
            {"ts": time.time(), "weight": weight, "target": target, "nonce": snap.get("nonce")},
        )

        # Clear snapshot + resume
        await ctx.set_kv("control:r_snapshot", None)
        await ctx.set_kv("control:r_pending", False)
        await ctx.set_kv("attention:allow_babble", True)

        return [
            self._speech_control(
                f"Applied {weight:+d} to item #{which}. Snapshot cleared. Resuming.",
                channel=channel,
                correlation_id=correlation_id,
            )
        ]

    def _hrm_recent_items(self, hrm, want_role: str, n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if hrm is None:
            return out
        recent = list(getattr(hrm, "recent_indices", []))
        for idx in reversed(recent):
            node = hrm.get_node(int(idx)) if hasattr(hrm, "get_node") else None
            if not node:
                continue
            role = str(getattr(node, "role", "") or "")
            if role != want_role:
                continue
            text = str(getattr(node, "text", "") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "hrm_idx": int(getattr(node, "idx", idx)),
                    "ts": float(getattr(node, "ts", 0.0)),
                    "text": text,
                }
            )
            if len(out) >= n:
                break
        return out

    def _speech_control(self, text: str, channel: str, correlation_id: str) -> Event:
        return Event(
            topic="act/speech",
            payload={"text": text, "style": "system", "channel": channel},
            source=self.name,
            correlation_id=correlation_id,
            meta={"control": True, "kind": "reinforcement"},
        )


def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    The orchestrator.neuron_loader.auto_register_neurons() will call this.
    """
    cfg = NeuronConfig(
        name="text_input",
        subscribed_topics=["input/text"],
        output_topics=["percept/text"],
        priority=10,  # early in the chain; feeds other percept neurons
    )
    yield TextInputNeuron(cfg)
