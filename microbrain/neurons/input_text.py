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




_LEADING_SLASH_COMMANDS = {
    "/r",
    "/t",
    "/user",
    "/power",
    "/cookie",
    "/read",
    "/slearn",
    "/vision",
    "/camera",
    "/focus",
    "/acc",
    "/echo",
    "/say",
    "/help",
    "/?",
    "/status",
    "/brain",
    "/who",
    "/reflect",
    "/introspect",
    "/self",
    "/why",
    "/explain",
    "/because",
    "/as_audio",
    "/audio",
}


def _leading_slash_command(text: str) -> str:
    stripped = str(text or "").lstrip()
    if not stripped.startswith("/"):
        return ""
    return stripped.split(maxsplit=1)[0].lower()

def _coerce_power_state(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        state = dict(raw)
    else:
        state = {"mode": str(raw or "active")}
    state["pct"] = max(0.0, min(100.0, float(state.get("pct", 100.0) or 100.0)))
    state["charging"] = bool(state.get("charging", False))
    state["sleep"] = bool(state.get("sleep", False))
    state["mode"] = str(state.get("mode", "active") or "active").lower()
    return state


def _display_power_pct(pct: Any) -> int:
    raw = max(0.0, min(100.0, float(pct or 0.0)))
    if raw >= 100.0:
        return 100
    bucket = int(raw // 5.0) * 5
    return max(0, min(95, bucket))


def _clamp_float(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _tone_label(value: float) -> str:
    """Human-readable signed valence/emphasis label for /acc metadata."""
    v = float(value or 0.0)
    mag = abs(v)
    if mag <= 0.25:
        return "neutral"
    if v > 0:
        if mag < 3.0:
            return "mild_positive_emphasis"
        if mag < 7.0:
            return "positive_emphasis"
        if mag < 8.0:
            return "strong_positive_emphasis"
        return "positive_salience_spike"

    # Negative /acc means unwanted/bad/corrective severity, not quietness.
    if mag < 3.0:
        return "mild_negative_correction"
    if mag < 7.0:
        return "negative_correction"
    if mag < 8.0:
        return "strong_negative_correction"
    return "negative_salience_warning"


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

        command_root = text_norm.split(maxsplit=1)[0].lower()

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
        # 2.1) /acc textual accent command
        # ----------------------------------------------
        # /acc is a control-plane wrapper that supplies signed tone/valence
        # metadata for an otherwise normal text input. The literal command
        # must not enter cognition or memory; only the cleaned text does.
        #
        #   +N = positive emphasis / salience / preference pull
        #   -N = unwanted/bad/corrective severity / preference push-away
        #    0 = neutral explicit marker
        acc_applied = False
        acc_match = re.match(r"^/acc\s+([+-]?\d+(?:\.\d+)?)\s+(.+)$", text_norm, re.IGNORECASE | re.DOTALL)
        if text_norm.lower() == "/acc" or text_norm.lower().startswith("/acc "):
            if not acc_match:
                return [
                    self._speech_error(
                        "Usage: /acc -10..10 <text>  (+ = positive emphasis, - = correction severity)",
                        channel=channel,
                        correlation_id=event.correlation_id,
                    )
                ]

            acc_value = _clamp_float(acc_match.group(1), -10.0, 10.0, 0.0)
            cleaned = str(acc_match.group(2) or "").strip()
            if not cleaned:
                return [
                    self._speech_error(
                        "Usage: /acc -10..10 <text>  (text cannot be empty)",
                        channel=channel,
                        correlation_id=event.correlation_id,
                    )
                ]

            text_norm = cleaned
            acc_applied = True
            magnitude = abs(acc_value)
            positive = max(0.0, acc_value)
            negative_severity = abs(min(0.0, acc_value))
            direction = "positive" if acc_value > 0 else ("negative" if acc_value < 0 else "neutral")

            merged_meta["input_mode"] = "textual_accent"
            merged_meta["accent_source"] = "acc_command"
            merged_meta["accent_value"] = acc_value
            merged_meta["accent_magnitude"] = magnitude
            merged_meta["accent_intensity"] = magnitude  # backward-compatible magnitude field
            merged_meta["accent_positive"] = positive
            merged_meta["accent_negative_severity"] = negative_severity
            merged_meta["accent_direction"] = direction
            merged_meta["accent_scale"] = "-10..10 signed: +=positive emphasis, -=negative correction"
            merged_meta["tone_label"] = _tone_label(acc_value)
            merged_meta["salience_delta"] = round(acc_value / 10.0, 4)
            merged_meta["preference_delta"] = round(acc_value / 10.0, 4)
            merged_meta["control_text_stripped"] = True

        # ----------------------------------------------
        # 2.5) Reinforcement snapshot latch (/r ...)
        # ----------------------------------------------
        r_pending = bool(await ctx.get_kv("control:r_pending", False))
        is_r_command = (not acc_applied) and (text_norm == "/r" or text_norm.startswith("/r "))

        # If a /r menu is open, refuse non-/r input until it is resolved.
        if r_pending and not is_r_command:
            return [
                self._speech_error(
                    "Reinforcement menu is still open. Use `/r +3 2`, `/r -2 4`, or `/r clear`.",
                    channel=channel,
                    correlation_id=event.correlation_id,
                )
            ]

        # Handle /r commands here so they don't become percept/text (no HRM/memory pollution).
        if (not acc_applied) and (text_norm == "/r" or text_norm.startswith("/r ")):
            return await self._handle_r_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # /t trainer latch: pause outward chatter and treat the NEXT user input as a correction.
        t_pending = bool(await ctx.get_kv("control:t_pending", False))
        is_t_command = (not acc_applied) and (text_norm == "/t" or text_norm.startswith("/t "))
        if t_pending and not is_t_command:
            return await self._handle_t_capture(
                correction_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        if (not acc_applied) and (text_norm == "/t" or text_norm.startswith("/t ")):
            return await self._handle_t_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /user commands here so they don't become percept/text.
        if (not acc_applied) and command_root == "/user":
            return await self._handle_user_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /power commands here so they don't become percept/text.
        if (not acc_applied) and command_root == "/power":
            return await self._handle_power_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /cookie here so it feeds virtual power without memory pollution.
        if (not acc_applied) and command_root == "/cookie":
            return await self._handle_cookie_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /read commands here so they don't become percept/text.
        if (not acc_applied) and (text_norm == "/read" or text_norm.startswith("/read ")):
            return await self._handle_read_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /slearn commands here so structured curriculum ingestion stays control-plane.
        if (not acc_applied) and (text_norm == "/slearn" or text_norm.startswith("/slearn ")):
            return await self._handle_slearn_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )
        
        # Handle /vision commands here so they don't become percept/text (babble can't see them).
        if (not acc_applied) and command_root == "/vision":
            return await self._handle_vision_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /camera commands here so webcam input stays control-plane until frames emit percept/vision.
        if (not acc_applied) and command_root == "/camera":
            return await self._handle_camera_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Handle /focus commands here so they don't become percept/text (babble can't see them).
        if (not acc_applied) and command_root == "/focus":
            return await self._handle_focus_command(
                cmd_text=text_norm,
                ctx=ctx,
                channel=channel,
                correlation_id=event.correlation_id,
            )

        # Leading slash is command plane only. Unknown leading-slash commands
        # are operator/UI errors and must never become percept/text, memory,
        # boredom/social input, or learned conversation. A slash inside normal
        # text (e.g. C:/tools or 3/4) remains valid conversational text.
        leading_cmd = _leading_slash_command(text_norm)
        leading_slash_command = bool(leading_cmd) and not acc_applied
        if leading_slash_command:
            if leading_cmd not in _LEADING_SLASH_COMMANDS:
                return [
                    self._speech_error(
                        f"I don't know the command '{leading_cmd}'. Try /help.",
                        channel=channel,
                        correlation_id=event.correlation_id,
                    )
                ]

            # Known commands that are still routed through router_text are marked
            # as control-plane packets so all cognition/memory/drive consumers can
            # ignore them defensively.
            merged_meta["control"] = True
            merged_meta["memory_source"] = "system_telemetry"
            merged_meta["store_in_memory"] = False
            merged_meta["reinforcement_eligible"] = False
            merged_meta["self_output_track"] = False
            merged_meta["cognitive_visible"] = False
            merged_meta["command_text"] = True
            merged_meta["command_root"] = leading_cmd

        # Record recent interaction mode only for cognition-plane text. Commands
        # are operator controls, not social/contact input.
        if not leading_slash_command:
            try:
                await ctx.set_kv(
                    "interaction:last_input",
                    {
                        "ts": time.time(),
                        "source": source,
                        "transport_source": transport_source,
                        "channel": channel,
                        "modality": "text",
                        "text": text_norm[:160],
                        "accent": {
                            "applied": bool(acc_applied),
                            "value": merged_meta.get("accent_value"),
                            "direction": merged_meta.get("accent_direction"),
                            "magnitude": merged_meta.get("accent_magnitude"),
                            "positive": merged_meta.get("accent_positive"),
                            "negative_severity": merged_meta.get("accent_negative_severity"),
                            "tone_label": merged_meta.get("tone_label"),
                        } if acc_applied else None,
                    },
                )
            except Exception:
                pass

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
                "kind": "command_text" if leading_slash_command else "percept",
                "modality": "text",
                "normalized": True,
                "control": bool(leading_slash_command),
                "memory_source": "system_telemetry" if leading_slash_command else None,
                "store_in_memory": False if leading_slash_command else None,
                "reinforcement_eligible": False if leading_slash_command else None,
                "self_output_track": False if leading_slash_command else None,
                "cognitive_visible": False if leading_slash_command else None,
                "command_text": bool(leading_slash_command),
                "command_root": leading_cmd if leading_slash_command else None,
                "accent_applied": bool(acc_applied),
                "accent_value": merged_meta.get("accent_value") if acc_applied else None,
                "accent_direction": merged_meta.get("accent_direction") if acc_applied else None,
                "accent_positive": merged_meta.get("accent_positive") if acc_applied else None,
                "accent_negative_severity": merged_meta.get("accent_negative_severity") if acc_applied else None,
            },
        )

        await ctx.log_debug(
            f"[{self.name}] Emitted percept/text",
            source=source,
            channel=channel,
        )

        return [percept_event]



    def _trainer_target_snapshot(self, last_spoken: Any, speech_reason_last: Any) -> Dict[str, Any]:
        spoken = dict(last_spoken) if isinstance(last_spoken, dict) else {}
        reason_last = dict(speech_reason_last) if isinstance(speech_reason_last, dict) else {}
        utterance = str(spoken.get("text", reason_last.get("utterance", "")) or "").strip()
        need = str(reason_last.get("need", "") or "").strip()
        style = str(reason_last.get("style", "") or "").strip()
        message = str(reason_last.get("message", "") or "").strip()
        context_parts = [p for p in [need, style, message, utterance] if p]
        context_query = " | ".join(context_parts) if context_parts else utterance
        return {
            "utterance": utterance,
            "need": need,
            "style": style,
            "message": message,
            "context_query": context_query,
            "spoken": spoken,
            "reason": reason_last,
        }

    async def _handle_t_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=1)
        arg = (parts[1] or "").strip().lower() if len(parts) > 1 else ""

        if arg in ("cancel", "clear", "off"):
            prev_allow = await ctx.get_kv("control:t_prev_allow_babble", True)
            await ctx.set_kv("control:t_pending", False)
            await ctx.set_kv("control:t_target", None)
            await ctx.set_kv("control:t_prev_allow_babble", None)
            await ctx.set_kv("attention:allow_babble", bool(prev_allow))
            await ctx.set_kv("attention:focus_target", "internal" if bool(prev_allow) else "external")
            return [self._speech_control("Trainer mode cancelled.", channel=channel, correlation_id=correlation_id)]

        if arg == "status":
            pending = bool(await ctx.get_kv("control:t_pending", False))
            target = await ctx.get_kv("control:t_target", None)
            utterance = str((target or {}).get("utterance", "") or "").strip() if isinstance(target, dict) else ""
            msg = f"Trainer: {'armed' if pending else 'off'}"
            if utterance:
                msg += f" | target={utterance}"
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        last_spoken = await ctx.get_kv("trainer:last_assistant_utterance", None)
        speech_reason_last = await ctx.get_kv("speech_reason:last", None)
        target = self._trainer_target_snapshot(last_spoken, speech_reason_last)
        if not str(target.get("utterance", "") or "").strip():
            return [self._speech_error("No recent assistant utterance to correct.", channel=channel, correlation_id=correlation_id)]

        prev_allow = bool(await ctx.get_kv("attention:allow_babble", True))
        await ctx.set_kv("control:t_prev_allow_babble", prev_allow)
        await ctx.set_kv("control:t_target", target)
        await ctx.set_kv("control:t_pending", True)
        await ctx.set_kv("attention:allow_babble", False)
        await ctx.set_kv("attention:focus_target", "external")
        return [
            self._speech_control(
                f"Trainer armed. Next input will correct: {target.get('utterance','')}",
                channel=channel,
                correlation_id=correlation_id,
            )
        ]

    async def _handle_t_capture(
        self,
        correction_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        target = await ctx.get_kv("control:t_target", None)
        prev_allow = await ctx.get_kv("control:t_prev_allow_babble", True)
        await ctx.set_kv("control:t_pending", False)
        await ctx.set_kv("control:t_target", None)
        await ctx.set_kv("control:t_prev_allow_babble", None)
        await ctx.set_kv("attention:allow_babble", bool(prev_allow))
        await ctx.set_kv("attention:focus_target", "internal" if bool(prev_allow) else "external")

        payload = {
            "correction_text": str(correction_text or "").strip(),
            "target": dict(target) if isinstance(target, dict) else {},
            "splitter": "|",
            "ts": time.time(),
        }
        return [
            Event(
                topic="control/trainer_correction",
                payload=payload,
                source=self.name,
                correlation_id=correlation_id,
                meta={"control": True, "kind": "trainer_correction"},
            )
        ]

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
    # /camera: control webcam capture without generating command percept/text
    # ------------------------------------------------------------------
    async def _handle_camera_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=2)  # "/camera", "<subcmd>", "<rest...>"

        if len(parts) == 1:
            enabled = bool(await ctx.get_kv("camera:enabled", False))
            selected = await ctx.get_kv("camera:selected", None)
            selected_text = "-"
            if isinstance(selected, dict):
                selected_text = f"{selected.get('index', '?')}: {selected.get('name', 'Camera')}"
            msg = (
                f"camera: {'on' if enabled else 'off'} | selected={selected_text}\n"
                "Usage: /camera list | /camera select <index> | /camera on|off | /camera preview on|off | /camera status"
            )
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        subcmd = (parts[1] or "").strip().lower()
        rest = (parts[2] or "").strip() if len(parts) > 2 else ""

        if subcmd in ("status", "state"):
            enabled = bool(await ctx.get_kv("camera:enabled", False))
            preview = bool(await ctx.get_kv("camera:preview", False))
            selected = await ctx.get_kv("camera:selected", None)
            selected_text = "-"
            if isinstance(selected, dict):
                selected_text = f"{selected.get('index', '?')}: {selected.get('name', 'Camera')}"
            fps = await ctx.get_kv("camera:fps", 2.0)
            msg = f"camera: {'on' if enabled else 'off'} | preview={'on' if preview else 'off'} | selected={selected_text} | fps={fps}"
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        if subcmd == "list":
            try:
                from microbrain.utils.mb_vision.camera_grabber import list_cameras
                cams = list_cameras()
                try:
                    await ctx.set_kv("camera:devices_last", [c.as_dict() for c in cams])
                except Exception:
                    pass
            except Exception as e:
                return [self._speech_error(
                    f"Camera list failed: {e!r}",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            if not cams:
                return [self._speech_control(
                    "No cameras found. Install/check opencv-python and camera permissions if this seems wrong.",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            lines = ["Cameras:"]
            for cam in cams[:25]:
                desc = f"  [{cam.index}] {cam.name}"
                if cam.width and cam.height:
                    desc += f" ({cam.width}x{cam.height})"
                if cam.backend:
                    desc += f" [{cam.backend}]"
                lines.append(desc)
            if len(cams) > 25:
                lines.append(f"...and {len(cams) - 25} more")
            lines.append("Pick one with: /camera select <index>")
            return [self._speech_control("\n".join(lines), channel=channel, correlation_id=correlation_id)]

        if subcmd == "select":
            if not rest:
                return [self._speech_error(
                    "Usage: /camera select <index>",
                    channel=channel,
                    correlation_id=correlation_id,
                )]
            try:
                idx = int(rest)
            except Exception:
                return [self._speech_error(
                    "Usage: /camera select <index>  (index must be a number)",
                    channel=channel,
                    correlation_id=correlation_id,
                )]
            if idx < 0 or idx > 32:
                return [self._speech_error(
                    "Camera index out of safe probe range 0..32.",
                    channel=channel,
                    correlation_id=correlation_id,
                )]

            chosen = {"index": idx, "name": f"Camera {idx}", "ts": time.time()}
            try:
                cached = await ctx.get_kv("camera:devices_last", None)
                if isinstance(cached, list):
                    for row in cached:
                        if isinstance(row, dict) and int(row.get("index", -1)) == idx:
                            chosen.update(row)
                            break
            except Exception:
                pass

            await ctx.set_kv("camera:selected", chosen)
            await ctx.set_kv("camera:selected_index", idx)
            return [
                Event(
                    topic="control/camera",
                    payload={"action": "select", "index": idx, "camera": chosen},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "camera_control"},
                ),
                self._speech_control(
                    f"Selected camera: {idx} ({chosen.get('name', 'Camera')}).",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        if subcmd in ("on", "off"):
            return [
                Event(
                    topic="control/camera",
                    payload={"action": subcmd},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "camera_control"},
                ),
                self._speech_control(
                    f"Camera {subcmd}.",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]

        if subcmd == "preview":
            if rest.lower() in ("on", "off"):
                act = "preview_on" if rest.lower() == "on" else "preview_off"
                return [
                    Event(
                        topic="control/camera",
                        payload={"action": act},
                        source=self.name,
                        correlation_id=correlation_id,
                        meta={"control": True, "kind": "camera_control"},
                    ),
                    self._speech_control(
                        f"Camera preview {rest.lower()}.",
                        channel=channel,
                        correlation_id=correlation_id,
                    ),
                ]
            return [self._speech_error(
                "Usage: /camera preview on|off",
                channel=channel,
                correlation_id=correlation_id,
            )]

        return [self._speech_error(
            "Unknown /camera subcommand. Try: list, select, on, off, preview on|off, status",
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

    # ------------------------------------------------------------------
    # /power: simulated battery + sleep/charge gate controls
    # ------------------------------------------------------------------
    async def _handle_power_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=2)  # "/power", "<subcmd>", "<rest...>"

        state = _coerce_power_state(await ctx.get_kv("power:state", None))

        if len(parts) == 1:
            msg = (
                f"power: {_display_power_pct(state.get('pct', 100.0))}% | "
                f"mode={str(state.get('mode', 'active'))} | "
                f"charging={bool(state.get('charging', False))} | "
                f"sleep={bool(state.get('sleep', False))} | "
                f"entropy_allowed={bool(await ctx.get_kv('entropy:allowed', False))}\n"
                "Usage:\n"
                "  /power status\n"
                "  /power pct <0-100>\n"
                "  /power charging on|off\n"
                "  /power sleep on|off\n"
                "  /cookie"
            )
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        subcmd = (parts[1] or "").strip().lower()
        rest = (parts[2] or "").strip() if len(parts) > 2 else ""

        if subcmd in ("status", "state"):
            msg = (
                f"power: {_display_power_pct(state.get('pct', 100.0))}% | "
                f"mode={str(state.get('mode', 'active'))} | "
                f"charging={bool(state.get('charging', False))} | "
                f"sleep={bool(state.get('sleep', False))} | "
                f"entropy_allowed={bool(await ctx.get_kv('entropy:allowed', False))}"
            )
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        if subcmd in ("pct", "set"):
            try:
                v = float(rest)
            except Exception:
                return [self._speech_error("Usage: /power pct <0-100> or /power set <0-100>", channel=channel, correlation_id=correlation_id)]
            v = max(0.0, min(100.0, v))

            # Immediately persist state so /power status reflects it right away
            state["pct"] = v
            state["last_ts"] = time.time()
            await ctx.set_kv("power:state", state)
            await ctx.set_kv("power:battery_pct", float(state.get("pct", 100.0)))
            await ctx.set_kv("power:charging", bool(state.get("charging", False)))
            await ctx.set_kv("power:sleep", bool(state.get("sleep", False)))
            await ctx.set_kv("power:mode", str(state.get("mode", "active")))
            await ctx.set_kv("entropy:allowed", bool(state.get("charging", False) and state.get("sleep", False)))

            return [
                Event(
                    topic="control/power",
                    payload={"set_pct": v},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "power_control"},
                ),
                self._speech_control(
                    f"Set battery to {v:.1f}% (display bucket {_display_power_pct(v)}%).",
                    channel=channel,
                    correlation_id=correlation_id,
                ),
            ]
        
        if subcmd == "charging":
            if rest.lower() not in ("on", "off", "true", "false", "1", "0"):
                return [self._speech_error("Usage: /power charging on|off", channel=channel, correlation_id=correlation_id)]
            val = rest.lower() in ("on", "true", "1")

            # Immediately persist state so /power status reflects it right away
            state["charging"] = val
            state["last_ts"] = time.time()
            await ctx.set_kv("power:charging_last_set_ts", state["last_ts"])
            await ctx.set_kv("power:state", state)
            await ctx.set_kv("power:battery_pct", float(state.get("pct", 100.0)))
            await ctx.set_kv("power:charging", bool(state.get("charging", False)))
            await ctx.set_kv("power:sleep", bool(state.get("sleep", False)))
            await ctx.set_kv("power:mode", str(state.get("mode", "active")))
            await ctx.set_kv("entropy:allowed", bool(state.get("charging", False) and state.get("sleep", False)))

            return [
                Event(
                    topic="control/power",
                    payload={"charging": val},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "power_control"},
                ),
                self._speech_control(f"Charging {'on' if val else 'off'}.", channel=channel, correlation_id=correlation_id),
            ]
        
        if subcmd == "sleep":
            if rest.lower() not in ("on", "off", "true", "false", "1", "0"):
                return [self._speech_error("Usage: /power sleep on|off", channel=channel, correlation_id=correlation_id)]
            val = rest.lower() in ("on", "true", "1")

            # Immediately persist state so /power status reflects it right away
            state["sleep"] = val
            state["last_ts"] = time.time()
            await ctx.set_kv("power:sleep_last_set_ts", state["last_ts"])
            await ctx.set_kv("power:state", state)
            await ctx.set_kv("power:battery_pct", float(state.get("pct", 100.0)))
            await ctx.set_kv("power:charging", bool(state.get("charging", False)))
            await ctx.set_kv("power:sleep", bool(state.get("sleep", False)))
            await ctx.set_kv("power:mode", str(state.get("mode", "active")))
            await ctx.set_kv("entropy:allowed", bool(state.get("charging", False) and state.get("sleep", False)))

            return [
                Event(
                    topic="control/power",
                    payload={"sleep": val},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "power_control"},
                ),
                self._speech_control(f"Sleep {'on' if val else 'off'}.", channel=channel, correlation_id=correlation_id),
            ]
        
        return [self._speech_error("Unknown /power subcommand. Try: status, pct, set, charging, sleep", channel=channel, correlation_id=correlation_id)]

    async def _handle_cookie_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        state = _coerce_power_state(await ctx.get_kv("power:state", None))
        before = float(state.get("pct", 100.0) or 100.0)
        after = max(0.0, min(100.0, before + 5.0))

        state["pct"] = after
        state["last_ts"] = time.time()
        await ctx.set_kv("power:state", state)
        await ctx.set_kv("power:battery_pct", after)
        await ctx.set_kv("power:charging", bool(state.get("charging", False)))
        await ctx.set_kv("power:sleep", bool(state.get("sleep", False)))
        await ctx.set_kv("power:mode", str(state.get("mode", "active")))
        await ctx.set_kv("entropy:allowed", bool(state.get("charging", False) and state.get("sleep", False)))

        delta = after - before
        if delta <= 0.0:
            msg = f"Cookie received, but power is already full at {_display_power_pct(after)}%."
        else:
            msg = (
                f"Cookie received. Power +{delta:.1f}% → {after:.1f}% "
                f"(display bucket {_display_power_pct(after)}%)."
            )

        return [
            Event(
                topic="control/power",
                payload={"set_pct": after, "reason": "cookie"},
                source=self.name,
                correlation_id=correlation_id,
                meta={"control": True, "kind": "power_control", "feed": "cookie"},
            ),
            self._speech_control(msg, channel=channel, correlation_id=correlation_id),
        ]


    async def _handle_read_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        parts = (cmd_text or "").strip().split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else "status"
        sub = arg.lower()

        if sub in ("", "status", "state"):
            enabled = bool(await ctx.get_kv("read:enabled", False))
            active_file = str(await ctx.get_kv("read:active_file", "") or "")
            active_kind = str(await ctx.get_kv("read:active_kind", "") or "")
            chunk_index = int(await ctx.get_kv("read:chunk_index", 0) or 0)
            idle_after_s = float(await ctx.get_kv("read:idle_after_s", 90.0) or 90.0)
            read_dir = str(await ctx.get_kv("read:dir", "") or "")
            last = await ctx.get_kv("read:last_result", {})
            progress = "idle"
            if isinstance(last, dict) and last:
                progress = str(last.get("summary", progress) or progress)
            msg = (
                f"read: {'on' if enabled else 'off'} | file={Path(active_file).name if active_file else '-'} | "
                f"kind={active_kind or '-'} | chunk={chunk_index} | idle_after={int(idle_after_s)}s\n"
                f"dir: {read_dir or '-'}\n"
                f"last: {progress}"
            )
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        if sub in ("on", "start"):
            await ctx.set_kv("read:enabled", True)
            return [
                Event(
                    topic="control/read",
                    payload={"command": "on"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "read_control"},
                ),
                self._speech_control("Read mode on.", channel=channel, correlation_id=correlation_id),
            ]

        if sub in ("off", "stop"):
            await ctx.set_kv("read:enabled", False)
            return [
                Event(
                    topic="control/read",
                    payload={"command": "off"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "read_control"},
                ),
                self._speech_control("Read mode off.", channel=channel, correlation_id=correlation_id),
            ]

        if sub in ("next", "step"):
            return [
                Event(
                    topic="control/read",
                    payload={"command": "next"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta={"control": True, "kind": "read_control"},
                ),
                self._speech_control("Read next chunk requested.", channel=channel, correlation_id=correlation_id),
            ]

        return [self._speech_error("Unknown /read subcommand. Try: on, off, status, next", channel=channel, correlation_id=correlation_id)]


    def _speech_user_profile(self, text: str, channel: str, correlation_id: str) -> Event:
        return self._control_status(
            text,
            channel=channel,
            correlation_id=correlation_id,
            kind="user_profile",
        )



    # ------------------------------------------------------------------
    # /slearn: structured CAPS learning sheets
    # ------------------------------------------------------------------
    async def _handle_slearn_command(
        self,
        cmd_text: str,
        ctx,
        channel: str,
        correlation_id: str,
    ) -> List[Event]:
        line = (cmd_text or "").strip()
        parts = line.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) >= 2 else "status"
        rest = parts[2].strip() if len(parts) >= 3 else ""

        if sub in ("status", "stat", "state"):
            enabled = bool(await ctx.get_kv("slearn:enabled", False))
            directory = str(await ctx.get_kv("slearn:dir", "") or "")
            weight = int(await ctx.get_kv("slearn:default_weight", 3) or 3)
            last = await ctx.get_kv("slearn:last_result", None)
            sidecar_started = bool(await ctx.get_kv("slearn:sidecar_started", False))
            sidecar_error = str(await ctx.get_kv("slearn:sidecar_error", "") or "")
            active_file = str(await ctx.get_kv("slearn:active_file", "") or "")
            chunk_index = int(await ctx.get_kv("slearn:chunk_index", 0) or 0)
            emitted_total = int(await ctx.get_kv("slearn:rules_emitted_total", 0) or 0)
            applied_total = int(await ctx.get_kv("slearn:rules_applied_total", 0) or 0)
            done_count = int(await ctx.get_kv("slearn:files_completed_count", 0) or 0)
            audit_path = str(await ctx.get_kv("slearn:audit_path", "") or "")

            msg = (
                f"Structured learning: {'on' if enabled else 'off'} | "
                f"sidecar={'running' if sidecar_started else 'not started'} | "
                f"weight=+{weight} | dir={directory or '(default Z:\\memory\\slearn_dir)'}"
            )
            if sidecar_error:
                msg += f"\nSidecar error: {sidecar_error}"
            if active_file:
                msg += f"\nActive: {active_file} @ chunk {chunk_index}"
            msg += f"\nProgress: files_done={done_count} emitted_total={emitted_total} applied_total={applied_total}"
            if audit_path:
                msg += f"\nAudit: {audit_path}"
            if isinstance(last, dict):
                msg += f"\nLast: {last.get('summary', last)}"
            msg += "\nUsage: /slearn on|off|status|next|dir <path>|weight <1-5>|template"
            return [self._speech_control(msg, channel=channel, correlation_id=correlation_id)]

        if sub in ("on", "start"):
            await ctx.set_kv("slearn:enabled", True)
            return [
                Event(
                    topic="control/slearn",
                    payload={"command": "on"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta=self._control_meta(kind="slearn_control"),
                ),
                self._speech_control("Structured learning on. Drop .txt/.md/.slearn files into the slearn_dir.", channel=channel, correlation_id=correlation_id),
            ]

        if sub in ("off", "stop"):
            await ctx.set_kv("slearn:enabled", False)
            return [
                Event(
                    topic="control/slearn",
                    payload={"command": "off"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta=self._control_meta(kind="slearn_control"),
                ),
                self._speech_control("Structured learning off.", channel=channel, correlation_id=correlation_id),
            ]

        if sub in ("next", "step"):
            return [
                Event(
                    topic="control/slearn",
                    payload={"command": "next"},
                    source=self.name,
                    correlation_id=correlation_id,
                    meta=self._control_meta(kind="slearn_control"),
                ),
                self._speech_control("Structured learning next chunk requested.", channel=channel, correlation_id=correlation_id),
            ]

        if sub == "dir":
            if not rest:
                return [self._speech_error("Usage: /slearn dir <folder>", channel=channel, correlation_id=correlation_id)]
            await ctx.set_kv("slearn:dir", rest)
            return [self._speech_control(f"Structured learning directory set: {rest}", channel=channel, correlation_id=correlation_id)]

        if sub == "weight":
            try:
                weight = max(1, min(5, int(rest)))
            except Exception:
                return [self._speech_error("Usage: /slearn weight <1-5>", channel=channel, correlation_id=correlation_id)]
            await ctx.set_kv("slearn:default_weight", weight)
            return [self._speech_control(f"Structured learning default weight set to +{weight}.", channel=channel, correlation_id=correlation_id)]

        if sub == "template":
            template = (
                "Structured learning sheet format: one rule per line. Lines beginning with # are ignored.\n"
                "IF USER says moin THEN CLASSIFY social_greeting, warmth, friendly AND REPLY good morning\n"
                "IF USER says thanks THEN CLASSIFY gratitude, warmth, friendly AND REPLY you're welcome\n"
                "IF USER says stop THEN CLASSIFY boundary, user_serious AND NOT REPLY playful\n"
                "IF POWER is low THEN CLASSIFY need_power, energy_deficit, homeostasis\n"
                "IF OBJECT detected THEN CLASSIFY base_object, noticed_thing\n"
                "IF scene_delta detected THEN CLASSIFY surprise, investigation_candidate"
            )
            return [self._speech_control(template, channel=channel, correlation_id=correlation_id)]

        return [self._speech_error("Unknown /slearn subcommand. Try: on, off, status, next, dir, weight, template", channel=channel, correlation_id=correlation_id)]

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
          /r +3 2 "IF USER says moin THEN CLASSIFY social_greeting, warmth AND REPLY good morning"
                         -> score + send structured teaching note to syntax_learning
          /r clear      -> clear snapshot and resume
        """
        line = (cmd_text or "").strip()
        parts = line.split()

        # parts[0] is "/r"
        if len(parts) == 1:
            return [
                self._speech_control(
                    "Usage:\n  /r u 5   (last 5 user)\n  /r a 5   (last 5 assistant)\n  /r +3 2  (score index)\n  /r +3 2 \"IF USER says moin THEN CLASSIFY social_greeting, warmth AND REPLY good morning\"\n  /r clear",
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

            items = await self._recent_reinforcement_items(ctx=ctx, want_role=want_role, n=n)

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
                label = self._reinforcement_item_label(it)
                lines.append(f"{i}) {label}  {preview}" if label else f"{i}) {preview}")
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

        # /r +W I ["teaching note"]   or   /r -W I ["teaching note"]
        score_parts = line.split(maxsplit=3)
        try:
            weight = int(score_parts[1])  # works for "+3" and "-2"
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

        if len(score_parts) < 3:
            return [
                self._speech_control(
                    "Missing index. Example: `/r +3 2`",
                    channel=channel,
                    correlation_id=correlation_id,
                )
            ]

        try:
            which = int(score_parts[2])
        except Exception:
            which = -1

        teaching_note = ""
        if len(score_parts) >= 4:
            teaching_note = score_parts[3].strip()
            if (len(teaching_note) >= 2) and (
                (teaching_note[0] == teaching_note[-1] == '"')
                or (teaching_note[0] == teaching_note[-1] == "'")
            ):
                teaching_note = teaching_note[1:-1].strip()

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
            {"ts": time.time(), "weight": weight, "target": target, "nonce": snap.get("nonce"), "teaching_note": teaching_note},
        )

        reinforce_event = Event(
            topic="control/reinforce",
            payload={
                "ts": time.time(),
                "weight": weight,
                "target_role": snap.get("role"),
                "target": target,
                "nonce": snap.get("nonce"),
                "teaching_note": teaching_note,
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
                f"Applied {weight:+d} to item #{which}"
                + (" with teaching note" if teaching_note else "")
                + ". Snapshot cleared. Resuming.",
                channel=channel,
                correlation_id=correlation_id,
            ),
        ]

    async def _recent_reinforcement_items(self, ctx, want_role: str, n: int) -> List[Dict[str, Any]]:
        """Return recent visible conversation turns for /r snapshots.

        /r is an operator UI affordance: the menu should show what the user
        actually saw in the conversation pane, not whatever survived HRM's
        long-term-memory filters.  HRM may intentionally skip system-ish or
        low-trust utterances, so using HRM as the primary source can make
        /r a/u point at stale items.  Conversation scene is the better live
        source; HRM remains a fallback for older sessions where the scene has
        not been built yet.
        """
        n = max(1, min(20, int(n or 5)))
        scene_items = self._conversation_scene_recent_items(
            scene=await ctx.get_kv("conversation:scene", None),
            want_role=want_role,
            n=n,
        )
        if scene_items:
            return scene_items

        hrm = await ctx.get_kv("hrm:core", None)
        return self._hrm_recent_items(hrm=hrm, want_role=want_role, n=n) if hrm else []

    def _conversation_scene_recent_items(self, scene: Any, want_role: str, n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(scene, dict):
            return out
        turns = scene.get("turns", [])
        if not isinstance(turns, list):
            return out

        for pos, turn in enumerate(reversed(turns)):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "") or "")
            if role != want_role:
                continue
            text = str(turn.get("text", "") or "").strip()
            if not text or text.lstrip().startswith("/"):
                continue
            out.append(
                {
                    "source": "conversation_scene",
                    "turn_idx": len(turns) - 1 - pos,
                    "ts": float(turn.get("ts", 0.0) or 0.0),
                    "text": text,
                    "correlation_id": str(turn.get("correlation_id", "") or ""),
                }
            )
            if len(out) >= n:
                break
        return out

    def _reinforcement_item_label(self, item: Dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        if item.get("hrm_idx") is not None:
            return f"hrm_idx={item.get('hrm_idx')}"
        if item.get("turn_idx") is not None:
            return f"turn_idx={item.get('turn_idx')}"
        return str(item.get("source", "") or "").strip()

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
                    "source": "hrm",
                    "hrm_idx": int(getattr(node, "idx", idx)),
                    "ts": float(getattr(node, "ts", 0.0)),
                    "text": text,
                }
            )
            if len(out) >= n:
                break
        return out

    def _control_meta(self, *, kind: str) -> Dict[str, Any]:
        return {
            "control": True,
            "kind": kind,
            "memory_source": "system_telemetry",
            "store_in_memory": False,
            "reinforcement_eligible": False,
            "self_output_track": False,
            "cognitive_visible": False,
        }

    def _control_status(
        self,
        text: str,
        *,
        channel: str,
        correlation_id: str,
        kind: str = "command_status",
    ) -> Event:
        return Event(
            topic="ui/status",
            payload={"text": text, "style": "system", "channel": channel},
            source=self.name,
            correlation_id=correlation_id,
            meta=self._control_meta(kind=kind),
        )

    def _speech_control(self, text: str, channel: str, correlation_id: str) -> Event:
        return self._control_status(
            text,
            channel=channel,
            correlation_id=correlation_id,
            kind="command_status",
        )

    def _speech_error(self, text: str, channel: str, correlation_id: str) -> Event:
        return Event(
            topic="ui/error",
            payload={"text": text, "style": "system", "channel": channel},
            source=self.name,
            correlation_id=correlation_id,
            meta=self._control_meta(kind="command_error"),
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
