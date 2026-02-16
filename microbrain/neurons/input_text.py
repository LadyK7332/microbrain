from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from typing import Iterable, Any, Dict, List, Optional


from microbrain.utils.memdir import resolve_memdir_ctx

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

        # Canonicalize transport sources so "ui" doesn't become a persona/name in memory/prompts.
        transport_source = str(source or "user")
        if transport_source not in ("user", "assistant", "system"):
            # UI / CLI / bridges are still the human user.
            source = "user"
        merged_meta["transport_source"] = transport_source

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

        # Handle /user commands here so they don't become percept/text.
        if text_norm.startswith("/user"):
            return await self._handle_user_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /vision commands here so they don't become percept/text (babble can't see them).
        if text_norm.startswith("/vision"):
            return await self._handle_vision_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /focus commands here so they don't become percept/text (babble can't see them).
        if text_norm.startswith("/focus"):
            return await self._handle_focus_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )


        # Attach current speaker identity (noun_id) for downstream binding/recall.
        if source == "user":
            noun_id = await ctx.get_kv("context:user_noun_id", None)
            noun_label = await ctx.get_kv("context:user_label", None)
            if noun_id:
                merged_meta.setdefault("noun_id", noun_id)
            if noun_label:
                merged_meta.setdefault("noun_label", noun_label)

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
    # /user: set a preferred user display name (used by the reasoning prompt)
    # ------------------------------------------------------------------
    async def _handle_user_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        parts = cmd_text.strip().split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        memdir = await resolve_memdir_ctx(ctx)
        state_dir = Path(memdir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        user_profile_path = state_dir / "user_profile.json"

        if not arg:
            current = await ctx.get_kv("profile:user_name", None)
            if not current and user_profile_path.exists():
                try:
                    current = json.loads(user_profile_path.read_text(encoding="utf-8")).get("user_name")
                except Exception:
                    current = None

            msg = (
                f"User name is set to '{current}'."
                if current
                else "User name not set. Use `/user Hazard` (or `/user clear`)."
            )
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        if arg.lower() in ("clear", "reset", "none", "off"):
            await ctx.set_kv("profile:user_name", None)
            try:
                if user_profile_path.exists():
                    user_profile_path.unlink()
            except Exception:
                pass
            return [self._speech_control("User name cleared.", channel=channel, correlation_id=correlation_id)]

        # Basic sanitation (keep it simple + safe)
        name = arg.strip().strip('"').strip("'")
        if len(name) > 48:
            name = name[:48]

        await ctx.set_kv("profile:user_name", name)

        try:
            user_profile_path.write_text(
                json.dumps({"user_name": name, "ts": time.time()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        return [self._speech_control(f"Got it. I'll call you '{name}'.", channel=channel, correlation_id=correlation_id)]

        # ------------------------------------------------------------------
    # /user: set current speaker identity (noun_id)
    # ------------------------------------------------------------------
    async def _handle_user_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        """
        Supported:
          /user Hazard        -> set user label + noun_id (noun:hazard)
          /user clear         -> clear user label + noun_id
          /user               -> show current user label + noun_id
        """
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=1)  # "/user", "<arg...>"

        current_label = await ctx.get_kv("context:user_label", None)
        current_noun = await ctx.get_kv("context:user_noun_id", None)

        if len(parts) == 1:
            label = str(current_label or "").strip() or "(unset)"
            noun = str(current_noun or "").strip() or "(unset)"
            msg = f"Current user: {label}  ({noun})\nUsage: /user <name>  or  /user clear"
            return [self._speech_user_profile(msg, channel=channel, correlation_id=correlation_id)]

        arg = (parts[1] or "").strip()
        if not arg:
            return [self._speech_user_profile("Usage: /user <name>  or  /user clear", channel=channel, correlation_id=correlation_id)]

        if arg.lower() in ("clear", "reset", "none", "off"):
            await ctx.set_kv("context:user_label", None)
            await ctx.set_kv("context:user_noun_id", None)
            await self._persist_user_profile(ctx, label=None, noun_id=None)
            return [self._speech_user_profile("User identity cleared.", channel=channel, correlation_id=correlation_id)]

        label = arg
        noun_id = self._noun_id_from_label(label)

        await ctx.set_kv("context:user_label", label)
        await ctx.set_kv("context:user_noun_id", noun_id)
        await self._persist_user_profile(ctx, label=label, noun_id=noun_id)

        return [self._speech_user_profile(f"User set to: {label}  ({noun_id})", channel=channel, correlation_id=correlation_id)]

    async def _persist_user_profile(self, ctx, label: Optional[str], noun_id: Optional[str]) -> None:
        """Best-effort persistence into memdir/state/user_profile.json"""
        try:
            mem_store = await ctx.get_kv("memory:store", None)
            base_dir = str(getattr(mem_store, "base_dir", "") or "")
            if not base_dir:
                return
            state_dir = Path(base_dir) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "speaker_noun_id": noun_id,
                "speaker_label": label,
                "ts": time.time(),
                "schema_ver": 1,
                "kind": "user_profile",
            }
            (state_dir / "user_profile.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _noun_id_from_label(self, label: str) -> str:
        """Convert freeform label -> stable noun_id."""
        s = (label or "").strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
        s = s.strip("._-")
        if not s:
            s = "user"
        return f"noun:{s}"

        # ------------------------------------------------------------------
    # /vision: control vision capture without generating percept/text
    # ------------------------------------------------------------------
    async def _handle_vision_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=2)  # "/vision", "<subcmd>", "<rest...>"

        if len(parts) == 1:
            return [self._speech_control(
                "Usage: /vision list | /vision select <n|title> | /vision on|off | /vision preview on|off",
                channel=channel,
                correlation_id=correlation_id,
            )]

        subcmd = (parts[1] or "").strip().lower()
        rest = (parts[2] or "").strip() if len(parts) > 2 else ""

        if subcmd == "list":
            try:
                from microbrain.utils.mb_vision.window_grabber import list_windows
                wins = list_windows()

                # Cache list for /vision select so capture neuron doesn't re-enumerate
                try:
                    await ctx.set_kv(
                        "vision:windows_last",
                        [{"title": w.title, "rect": w.rect} for w in wins],
                    )
                except Exception:
                    pass

            except Exception as e:
                return [self._speech_control(
                    f"Vision list failed: {e!r}",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            if not wins:
                return [self._speech_control(
                    "No windows found.",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            lines = ["Windows:"]
            for i, w in enumerate(wins[:25]):
                title = w.title
                if len(title) > 80:
                    title = title[:77] + "..."
                lines.append(f"  [{i}] {title} ({w.width}x{w.height})")
            if len(wins) > 25:
                lines.append(f"...and {len(wins) - 25} more")
            lines.append("Pick one with: /vision select <index|title_substring>")

            return [self._speech_control(
                "\n".join(lines),
                channel=channel,
                correlation_id=correlation_id,
            )]

        if subcmd == "select":
            if not rest:
                return [self._speech_control(
                    "Usage: /vision select <index|title_substring>",
                    channel=channel,
                    correlation_id=correlation_id,
                )]
            # Prefer selecting from cached /vision list (avoids a second window enumeration)
            chosen = None
            try:
                cached = await ctx.get_kv("vision:windows_last", None)
                if isinstance(cached, list) and cached:
                    # index select
                    if rest.isdigit():
                        idx = int(rest)
                        if 0 <= idx < len(cached):
                            chosen = cached[idx]
                    else:
                        low = rest.lower()
                        for row in cached:
                            if isinstance(row, dict) and low in str(row.get("title", "")).lower():
                                chosen = row
                                break
            except Exception:
                chosen = None

            if chosen and isinstance(chosen, dict):
                return [
                    Event(
                        topic="control/vision",
                        payload={"action": "select", "selector": rest, "window": chosen},
                        source=self.name,
                        correlation_id=correlation_id,
                        meta={"control": True, "kind": "vision_control"},
                    ),
                    self._speech_control(
                        f"Selected window (cached): {chosen.get('title','')!r}",
                        channel=channel,
                        correlation_id=correlation_id,
                    ),
                ]

            # Fallback: let vision capture neuron resolve selector
            return [
                Event(
                    topic="control/vision",
                    payload={"action": "select", "selector": rest},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "vision_control"},
                ),
                self._speech_control(
                    f"Selecting window: {rest!r}",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        if subcmd in ("on", "off"):
            return [
                Event(
                    topic="control/vision",
                    payload={"action": subcmd},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "vision_control"},
                ),
                self._speech_control(
                    f"Vision {subcmd}.",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        if subcmd == "preview":
            if rest.lower() in ("on", "off"):
                act = "preview_on" if rest.lower() == "on" else "preview_off"
                return [
                    Event(
                        topic="control/vision",
                        payload={"action": act},
                        source=self.name,
                        correlation_id=correlation_id,
                        meta={"control": True, "kind": "vision_control"},
                    ),
                    self._speech_control(
                        f"Vision preview {rest.lower()}.",
                        channel=channel,
                        correlation_id=correlation_id,
                    ),
                ]
            return [self._speech_control(
                "Usage: /vision preview on|off",
                channel=channel,
                correlation_id=correlation_id,
            )]

        return [self._speech_control(
            "Unknown /vision subcommand. Try: list, select, on, off, preview on|off",
            channel=channel,
            correlation_id=correlation_id,
        )]

    # ------------------------------------------------------------------
    # /focus: set vision reticle focus (normalized 0..1), no percept/text
    # ------------------------------------------------------------------
    async def _handle_focus_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split()

        if len(parts) == 1:
            return [self._speech_control(
                "Usage: /focus center | /focus <x> <y> (normalized 0..1)",
                channel=channel,
                correlation_id=correlation_id,
            )]

        if len(parts) == 2 and parts[1].lower() == "center":
            return [
                Event(
                    topic="control/focus",
                    payload={"action": "center"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "vision_focus"},
                ),
                self._speech_control(
                    "Focus set to center.",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        if len(parts) >= 3:
            try:
                x = float(parts[1])
                y = float(parts[2])
            except Exception:
                return [self._speech_control(
                    "Usage: /focus <x> <y> (numbers, normalized 0..1)",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            # clamp
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            return [
                Event(
                    topic="control/focus",
                    payload={"action": "set", "x": x, "y": y},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "vision_focus"},
                ),
                self._speech_control(
                    f"Focus set to ({x:.3f}, {y:.3f}).",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        return [self._speech_control(
            "Usage: /focus center | /focus <x> <y> (normalized 0..1)",
            channel=channel,
            correlation_id=correlation_id,
        )]


    def _speech_user_profile(self, text: str, channel: str, correlation_id: str) -> Event:
        return Event(
            topic="act/speech",
            payload={"text": text, "style": "system", "channel": channel},
            source=self.name,
            correlation_id=correlation_id,
            meta={"control": True, "kind": "user_profile"},
        )


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

        reinforce_event = Event(
            topic="control/reinforce",
            payload={
                "ts": time.time(),
                "weight": weight,
                "target_role": snap.get("role"),
                "target": target,
                "nonce": snap.get("nonce"),
            },
            source=self.name,
            correlation_id=correlation_id,
            meta={"control": True, "kind": "reinforcement"},
        )

        # Clear snapshot + resume
        await ctx.set_kv("control:r_snapshot", None)
        await ctx.set_kv("control:r_pending", False)
        await ctx.set_kv("attention:allow_babble", True)

        return [
            reinforce_event,
            self._speech_control(
                f"Applied {weight:+d} to item #{which}. Snapshot cleared. Resuming.",
                channel=channel,
                correlation_id=correlation_id,
            ),
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
