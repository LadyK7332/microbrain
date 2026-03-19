from __future__ import annotations

import time
import re
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    s = s.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return hh, mm


def _time_in_window(now_h: int, now_m: int, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    now = now_h * 60 + now_m
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start == end:
        return True  # whole day
    if start < end:
        return start <= now < end
    # Overnight window (e.g. 22:00 -> 06:00)
    return now >= start or now < end


class TextRouterNeuron(BaseNeuron):
    def __init__(self, cfg: NeuronConfig):
        super().__init__(cfg)
        # Trace for causal explanations
        self._last_user_text: str = ""
        self._last_user_meta: Dict[str, Any] = {}
        self._last_assistant_text: str = ""
        self._last_assistant_meta: Dict[str, Any] = {}
    # --- debug roll call (only active when --debug is passed) ----

    """
    Router for perceptual text.

    Listens on:
        - "percept/text"   (normalized text input)

    Emits:
        - "act/speech"     for fast, low-latency responses
        - "reason/request" for the downstream responder path

    Routing rules (v1, simple):

    1) Slash commands (text starting with "/"):
       - /echo <msg>     → act/speech with <msg>
       - /help           → act/speech with a help blurb
       - unknown /cmd    → small error message

    2) Short greetings:
       - "hi", "hello", "hey", "yo" (case-insensitive)
         → quick friendly reply via act/speech

    3) Everything else:
       → forward to the responder path via "reason/request"
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
            )

        payload = event.payload

        # Track last external input time for idle/sleep scheduling.
        try:
            if event.topic == "percept/text":
                await ctx.set_kv("power:last_external_ts", time.time())
        except Exception:
            pass

        # If this is a speech event, record the last assistant reply for /why.
        if event.topic == "act/speech":
            if isinstance(payload, dict):
                text = str(payload.get("text", "") or "")
                style = str(payload.get("style", "") or "")
                channel = str(payload.get("channel", "default"))
                # Only track assistant/system outputs, not echoes
                if style in ("assistant", "system"):
                    self._last_assistant_text = text
                    self._last_assistant_meta = {
                        "channel": channel,
                        "style": style,
                    }
                    await ctx.log_debug(
                        f"[{self.name}] Updated last assistant reply",
                        channel=channel,
                        style=style,
                    )
            return []

        if not isinstance(payload, dict) or "text" not in payload:
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload for percept/text",
                payload_type=str(type(payload)),
            )
            return []

        text = str(payload.get("text", "")).strip()
        if not text:
            await ctx.log_debug(
                f"[{self.name}] Empty text in percept, ignoring",
                topic=event.topic,
            )
            return []

        source = str(payload.get("source", "user") or "user")
        channel = str(payload.get("channel", "default") or "default")
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        transport_source = str(raw_meta.get("transport_source", raw_meta.get("source", source)) or source)
        text = str(payload.get("text", "")).strip()

        lowered = text.lower()

        # Track last user input for explanations, but avoid pure /commands and "why?" probes
        if not text.startswith("/") and not self._looks_like_why_question(lowered):
            self._last_user_text = text
            self._last_user_meta = {
                "channel": channel,
                "source": source,
                "transport_source": transport_source,
            }
            
        # ------------------------------
        # 1) Slash commands
        # ------------------------------
        if text.startswith("/"):
            cmd_line = text[1:].strip()
            if not cmd_line:
                return [self._speech(
                    "Empty command. Try /help.",
                    channel=channel,
                    style="system",
                    event=event,
                )]

            parts = cmd_line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("echo", "say"):
                message = arg or "(nothing to echo)"
                await ctx.log_debug(
                    f"[{self.name}] Handling /echo locally",
                    channel=channel,
                )
                return [self._speech(
                    message,
                    channel=channel,
                    style="default",
                    event=event,
                )]

            if cmd in ("help", "?"):
                help_text = (
                    "Available commands:\n"
                    "  /echo <text>  - echo back text quickly\n"
                    "  /help         - show this help\n"
                    "  /status         - describe my current brain state\n"
                    "  /reflect        - reflect on my own status\n"
                    "  /why            - explain why I answered the way I did\n"
                    "Other messages are sent to my reasoning core."
                )
                await ctx.log_debug(
                    f"[{self.name}] Handling /help locally",
                    channel=channel,
                )
                return [self._speech(
                    help_text,
                    channel=channel,
                    style="system",
                    event=event,
                )]
            if cmd in ("status", "brain", "who"):
                await ctx.log_debug(
                    f"[{self.name}] Routing /status to introspection neuron",
                    channel=channel,
                )
                # Ask the status_introspect neuron for a report
                introspect_payload = {
                    "command": cmd,
                    "text": text,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                }
                return [Event(
                    topic="introspect/status",
                    payload=introspect_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "introspect_request"},
                )]
            if cmd in ("reflect", "introspect", "self"):
                await ctx.log_debug(
                    f"[{self.name}] Routing /reflect to introspection neuron",
                    channel=channel,
                )
                introspect_payload = {
                    "command": "reflect",
                    "text": text,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                }
                return [Event(
                    topic="introspect/status",
                    payload=introspect_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "introspect_request"},
                )]
            if cmd in ("why", "explain", "because"):
                await ctx.log_debug(
                    f"[{self.name}] Handling /why via introspect/why",
                    channel=channel,
                )

                last_user = (self._last_user_text or "").strip()
                last_reply = (self._last_assistant_text or "").strip()

                introspect_payload = {
                    "last_user": last_user,
                    "last_reply": last_reply,
                    "source": source,
                    "channel": channel,
                    "raw_meta": raw_meta,
                    "command": cmd,
                }

                return [Event(
                    topic="introspect/why",
                    payload=introspect_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "why_explain_request"},
                )]

            if cmd in ("er", "evidence"):
                # /er on|off|status|arm|disarm|threshold <n>
                sub = (arg or "").strip().lower()
                parts2 = sub.split()

                if not sub or sub == "status":
                    enabled = bool(await ctx.get_kv("er:enabled", True))
                    armed = bool(await ctx.get_kv("er:armed", False))
                    manual = bool(await ctx.get_kv("er:manual_hold", False))
                    sess = str(await ctx.get_kv("er:session_id", "") or "")
                    thr = int(await ctx.get_kv("er:hazard_threshold", 3) or 3)
                    last_lvl = int(await ctx.get_kv("er:last_level", 0) or 0)
                    return [self._speech(
                        f"ER enabled={enabled} armed={armed} manual_hold={manual} threshold={thr} session={sess or 'none'} last_level={last_lvl}",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                action = parts2[0]

                if action in ("on", "enable", "true"):
                    await ctx.set_kv("er:enabled", True)
                    return [self._speech("Evidence recorder: ON.", channel=channel, style="system", event=event)]

                if action in ("off", "disable", "false"):
                    await ctx.set_kv("er:enabled", False)
                    await ctx.set_kv("er:armed", False)
                    await ctx.set_kv("er:manual_hold", False)
                    return [self._speech("Evidence recorder: OFF.", channel=channel, style="system", event=event)]

                if action == "arm":
                    await ctx.set_kv("er:enabled", True)
                    await ctx.set_kv("er:armed", True)
                    await ctx.set_kv("er:manual_hold", True)
                    await ctx.set_kv("er:last_trigger_ts", time.time())
                    await ctx.set_kv("er:last_reason", "manual_arm")
                    await ctx.set_kv("er:last_level", 3)
                    await ctx.set_kv("er:last_source", self.name)
                    return [self._speech("Evidence recorder: ARMED.", channel=channel, style="system", event=event)]

                if action in ("disarm", "stop"):
                    await ctx.set_kv("er:armed", False)
                    await ctx.set_kv("er:manual_hold", False)
                    return [self._speech("Evidence recorder: DISARMED.", channel=channel, style="system", event=event)]

                if action == "threshold" and len(parts2) > 1:
                    try:
                        thr = max(1, min(9, int(parts2[1])))
                        await ctx.set_kv("er:hazard_threshold", thr)
                    except Exception:
                        pass
                    thr = int(await ctx.get_kv("er:hazard_threshold", 3) or 3)
                    return [self._speech(f"ER hazard threshold set to {thr}.", channel=channel, style="system", event=event)]

                return [self._speech("Usage: /er on|off|status|arm|disarm|threshold <n>", channel=channel, style="system", event=event)]

            if cmd in ("power", "pwr"):
                # /power sleep on|off|toggle|status|now
                # /power charging on|off|toggle|status
                # /power window HH:MM-HH:MM
                sub = (arg or "").strip()
                if not sub:
                    return [self._speech(
                        "Usage: /power sleep on|off|toggle|status|now | /power charging on|off|toggle|status | /power window HH:MM-HH:MM",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                parts2 = sub.split()
                domain = parts2[0].lower()
                action = parts2[1].lower() if len(parts2) > 1 else "status"

                if domain == "sleep":
                    cur = bool(await ctx.get_kv("power:sleep", False))
                    idle_s = float(await ctx.get_kv("power:sleep_idle_s", 20.0) or 20.0)
                    period_s = float(await ctx.get_kv("power:sleep_period_s", 30.0) or 30.0)

                    if action in ("on", "enable", "true"):
                        await ctx.set_kv("power:sleep", True)
                        await ctx.set_kv("power:sleep_last_set_ts", time.time())
                        return [self._speech(
                            f"Sleep consolidation: ON (idle>= {idle_s:.0f}s, period≈ {period_s:.0f}s).",
                            channel=channel,
                            style="system",
                            event=event,
                        )]

                    if action in ("off", "disable", "false"):
                        await ctx.set_kv("power:sleep", False)
                        await ctx.set_kv("power:sleep_last_set_ts", time.time())
                        return [self._speech(
                            "Sleep consolidation: OFF.",
                            channel=channel,
                            style="system",
                            event=event,
                        )]

                    if action in ("toggle", "swap"):
                        newv = not cur
                        await ctx.set_kv("power:sleep", newv)
                        await ctx.set_kv("power:sleep_last_set_ts", time.time())
                        return [self._speech(
                            f"Sleep consolidation: {'ON' if newv else 'OFF'}.",
                            channel=channel,
                            style="system",
                            event=event,
                        )]

                    if action in ("now", "run", "cycle"):
                        if not cur:
                            return [self._speech(
                                "Sleep consolidation is OFF. Use: /power sleep on",
                                channel=channel,
                                style="system",
                                event=event,
                            )]
                        await ctx.set_kv("power:sleep_kick", True)
                        return [self._speech(
                            "Sleep consolidation: requested one cycle.",
                            channel=channel,
                            style="system",
                            event=event,
                        )]

                    last_cycle = float(await ctx.get_kv("power:sleep_last_cycle_ts", 0.0) or 0.0)
                    return [self._speech(
                        f"Sleep is {'ON' if cur else 'OFF'}. idle>= {idle_s:.0f}s period≈ {period_s:.0f}s last_cycle_ts={last_cycle:.0f}",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                if domain == "charging":
                    cur = bool(await ctx.get_kv("power:charging", False))
                    if action in ("on", "enable", "true"):
                        await ctx.set_kv("power:charging", True)
                        await ctx.set_kv("power:charging_last_set_ts", time.time())
                        return [self._speech("Charging state: ON.", channel=channel, style="system", event=event)]
                    if action in ("off", "disable", "false"):
                        await ctx.set_kv("power:charging", False)
                        await ctx.set_kv("power:charging_last_set_ts", time.time())
                        return [self._speech("Charging state: OFF.", channel=channel, style="system", event=event)]
                    if action in ("toggle", "swap"):
                        newv = not cur
                        await ctx.set_kv("power:charging", newv)
                        await ctx.set_kv("power:charging_last_set_ts", time.time())
                        return [self._speech(f"Charging state: {'ON' if newv else 'OFF'}.", channel=channel, style="system", event=event)]
                    return [self._speech(f"Charging is {'ON' if cur else 'OFF'}.", channel=channel, style="system", event=event)]


                if domain == "idle":
                    cur = bool(await ctx.get_kv("power:idle_enabled", True))
                    after_s = float(await ctx.get_kv("power:idle_after_s", 60.0) or 60.0)
                    cpu_thr = float(await ctx.get_kv("power:idle_cpu_threshold", 15.0) or 15.0)

                    if action in ("on", "enable", "true"):
                        await ctx.set_kv("power:idle_enabled", True)
                        return [self._speech(f"Idle mode: ON (after={after_s:.0f}s, cpu<={cpu_thr:.0f}%).", channel=channel, style="system", event=event)]
                    if action in ("off", "disable", "false"):
                        await ctx.set_kv("power:idle_enabled", False)
                        return [self._speech("Idle mode: OFF.", channel=channel, style="system", event=event)]
                    if action in ("after", "delay") and len(parts2) > 2:
                        try:
                            v = float(parts2[2])
                            await ctx.set_kv("power:idle_after_s", max(5.0, v))
                        except Exception:
                            pass
                        after_s = float(await ctx.get_kv("power:idle_after_s", 60.0) or 60.0)
                        return [self._speech(f"Idle after set to {after_s:.0f}s.", channel=channel, style="system", event=event)]
                    if action in ("cpu", "threshold") and len(parts2) > 2:
                        try:
                            v = float(parts2[2])
                            await ctx.set_kv("power:idle_cpu_threshold", max(1.0, min(95.0, v)))
                        except Exception:
                            pass
                        cpu_thr = float(await ctx.get_kv("power:idle_cpu_threshold", 15.0) or 15.0)
                        return [self._speech(f"Idle CPU threshold set to {cpu_thr:.0f}%.", channel=channel, style="system", event=event)]

                    last_cpu = await ctx.get_kv("power:last_cpu_percent", None)
                    status = "ON" if cur else "OFF"
                    return [self._speech(f"Idle is {status} (after={after_s:.0f}s, cpu<={cpu_thr:.0f}%, last_cpu={last_cpu}).", channel=channel, style="system", event=event)]

                if domain == "window":
                    spec = (parts2[1] if len(parts2) > 1 else "").strip()
                    if "-" not in spec:
                        return [self._speech("Usage: /power window HH:MM-HH:MM", channel=channel, style="system", event=event)]
                    a, b = spec.split("-", 1)
                    pa = _parse_hhmm(a)
                    pb = _parse_hhmm(b)
                    if not pa or not pb:
                        return [self._speech("Bad time format. Use HH:MM-HH:MM (24h).", channel=channel, style="system", event=event)]
                    await ctx.set_kv("power:charge_window_start", f"{pa[0]:02d}:{pa[1]:02d}")
                    await ctx.set_kv("power:charge_window_end", f"{pb[0]:02d}:{pb[1]:02d}")
                    return [self._speech(f"Charge window set: {pa[0]:02d}:{pa[1]:02d}-{pb[0]:02d}:{pb[1]:02d}", channel=channel, style="system", event=event)]

                return [self._speech(
                    "Usage: /power sleep … | /power charging … | /power window HH:MM-HH:MM",
                    channel=channel,
                    style="system",
                    event=event,
                )]


            if cmd in ("as_audio", "audio"):
                # Treat the rest of the line as if it came from microphone STT.
                spoken_text = arg.strip()
                if not spoken_text:
                    return [self._speech(
                        "Usage: /as_audio <what you would have said out loud>",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                await ctx.log_debug(
                    f"[{self.name}] Injecting audio transcription as percept/audio",
                    channel=channel,
                    text_preview=spoken_text[:80],
                )

                audio_payload = {
                    "text": spoken_text,
                    "confidence": 1.0,
                    "speaker": "user",
                    "channel": channel,
                    "raw_meta": {
                        "source": "cli",
                        "input_modality": "audio",
                    },
                }

                audio_event = Event(
                    topic="percept/audio",
                    payload=audio_payload,
                    source=self.name,
                    correlation_id=event.correlation_id,
                )

                return [audio_event]

            
            if cmd == "vision":
                sub = (arg or "").strip()
                if not sub:
                    return [self._speech(
                        "Usage: /vision list | /vision select <n|title> | /vision on|off | /vision preview on|off",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                parts2 = sub.split()
                subcmd = parts2[0].lower()
                rest = " ".join(parts2[1:]).strip()

                if subcmd == "list":
                    try:
                        from microbrain.utils.mb_vision.window_grabber import list_windows
                        wins = list_windows()
                    except Exception as e:
                        return [self._speech(
                            f"Vision list failed: {e!r}",
                            channel=channel,
                            style="system",
                            event=event,
                        )]

                    if not wins:
                        return [self._speech(
                            "No windows found.",
                            channel=channel,
                            style="system",
                            event=event,
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
                    return [self._speech("\n".join(lines), channel=channel, style="system", event=event)]

                if subcmd == "select":
                    if not rest:
                        return [self._speech(
                            "Usage: /vision select <index|title_substring>",
                            channel=channel,
                            style="system",
                            event=event,
                        )]
                    return [
                        Event(
                            topic="control/vision",
                            payload={"action": "select", "selector": rest},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={"kind": "vision_control"},
                        ),
                        self._speech(
                            f"Selecting window: {rest!r}",
                            channel=channel,
                            style="system",
                            event=event,
                        ),
                    ]

                if subcmd in ("on", "off"):
                    return [
                        Event(
                            topic="control/vision",
                            payload={"action": subcmd},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={"kind": "vision_control"},
                        ),
                        self._speech(
                            f"Vision {subcmd}.",
                            channel=channel,
                            style="system",
                            event=event,
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
                                correlation_id=event.correlation_id,
                                meta={"kind": "vision_control"},
                            ),
                            self._speech(
                                f"Vision preview {rest.lower()}.",
                                channel=channel,
                                style="system",
                                event=event,
                            ),
                        ]
                    return [self._speech(
                        "Usage: /vision preview on|off",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                return [self._speech(
                    "Unknown /vision subcommand. Try: list, select, on, off, preview on|off",
                    channel=channel,
                    style="system",
                    event=event,
                )]

            if cmd == "focus":
                sub = (arg or "").strip()
                if not sub:
                    return [self._speech(
                        "Usage: /focus center | /focus <x> <y> (normalized 0..1)",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                if sub.lower() == "center":
                    return [
                        Event(
                            topic="control/focus",
                            payload={"action": "center"},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={"kind": "focus_control"},
                        ),
                        self._speech("Focus set to center.", channel=channel, style="system", event=event),
                    ]

                pts = sub.split()
                if len(pts) != 2:
                    return [self._speech(
                        "Usage: /focus <x> <y> (normalized 0..1)",
                        channel=channel,
                        style="system",
                        event=event,
                    )]
                try:
                    x = float(pts[0])
                    y = float(pts[1])
                except Exception:
                    return [self._speech(
                        "Couldn't parse numbers. Example: /focus 0.5 0.5",
                        channel=channel,
                        style="system",
                        event=event,
                    )]

                return [
                    Event(
                        topic="control/focus",
                        payload={"action": "set", "x": x, "y": y},
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={"kind": "focus_control"},
                    ),
                    self._speech(f"Focus set to ({x:.3f}, {y:.3f}).", channel=channel, style="system", event=event),
                ]

# Unknown command
            await ctx.log_debug(
                f"[{self.name}] Unknown command, responding locally",
                cmd=cmd,
                channel=channel,
            )
            return [self._speech(
                f"I don't know the command '/{cmd}'. Try /help.",
                channel=channel,
                style="system",
                event=event,
            )]

        # ------------------------------
        # Natural-language introspection requests ("why did you say that?")
        # ------------------------------
        if self._looks_like_why_question(lowered):
            last_user = (self._last_user_text or "").strip()
            last_reply = (self._last_assistant_text or "").strip()

            introspect_payload = {
                "last_user": last_user,
                "last_reply": last_reply,
                "source": source,
                "channel": channel,
                "raw_meta": raw_meta,
            }

            return [Event(
                topic="introspect/why",
                payload=introspect_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "why_explain_request"},
            )]

        # ------------------------------
        # Natural-language recollection requests ("remember that thing...")
        # ------------------------------
        if self._looks_like_recollection_request(lowered):
            recollect_payload = {
                "query_text": text,
                "raw_user_text": text,
                "source": source,
                "channel": channel,
                "raw_meta": raw_meta,
            }

            return [Event(
                topic="memory/recollect",
                payload=recollect_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "recollection_request"},
            )]
        # ------------------------------
        # 2) Greetings: warmup fast-path, then defer to learned behavior
        # ------------------------------
        if lowered in ("hi", "hello", "hey", "yo"):
            # Prefer the /user display name when available
            user_name = await ctx.get_kv("profile:user_name", None)
            user_label = str(user_name).strip() if user_name else "there"

            # Phase out canned greetings as semantic memory grows
            mem_store = await ctx.get_kv("memory:store", None)
            try:
                semantic_n = len(getattr(mem_store, "semantic", [])) if mem_store else 0
            except Exception:
                semantic_n = 0

            warmup_max = int(await ctx.get_kv("router:greet_warmup_semantic_max", 50) or 50)

            if semantic_n < warmup_max:
                await ctx.log_debug(
                    f"[{self.name}] Handling greeting locally (warmup)",
                    text=text,
                    channel=channel,
                    semantic_n=semantic_n,
                    warmup_max=warmup_max,
                )
                return [self._speech(
                    f"Hey, {user_label}! What's up?",
                    channel=channel,
                    style="assistant",
                    event=event,
                )]

            # Past warmup: let the reasoner handle it (learned/personalized)
            await ctx.log_debug(
                f"[{self.name}] Greeting forwarded to responder path (post-warmup)",
                text=text,
                channel=channel,
                semantic_n=semantic_n,
                warmup_max=warmup_max,
            )
            # fall through to default forwarding

        # ------------------------------
        # 3) Default: forward to responder path
        # ------------------------------
        await ctx.log_debug(
            f"[{self.name}] Forwarding to responder path",
            channel=channel,
        )

        reason_payload = {
            "text": text,
            "source": source,
            "channel": channel,
            "raw_meta": raw_meta,
        }

        reason_event = Event(
            topic="reason/request",
            payload=reason_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"routed": True},
        )

        return [reason_event]

    # ------------------------------
    # Helper: detect "why did you say that?" style questions
    # ------------------------------
    def _looks_like_why_question(self, lowered: str) -> bool:
        stripped = lowered.strip()
        if not stripped:
            return False

        # Very short forms
        if stripped in ("why?", "why", "why tho", "why though"):
            return True

        phrases = [
            # External "you" forms
            "why did you say",
            "why did you respond",
            "why did you reply",
            "why that answer",
            "why this answer",
            "why this response",
            "why that response",
            "what made you say that",
            "what made you respond",
            "explain that answer",
            "explain your answer",
            "explain your response",
            # Inner-thought "i" forms
            "why did i say",
            "why did i respond",
            "why did i reply",
            "what made me say that",
            "what made me respond",
            "explain my answer",
            "explain my response",
        ]
        return any(p in stripped for p in phrases)


    # ------------------------------
    # Helper: detect "remember that thing..." recollection requests
    # ------------------------------
    def _looks_like_recollection_request(self, lowered: str) -> bool:
        stripped = lowered.strip()
        if not stripped:
            return False

        if "remember" not in stripped:
            return False

        phrases = [
            "remember that time",
            "remember that thing",
            "remember the thing",
            "remember when you said",
            "remember what you said",
            "remember that funny thing",
            "remember that joke",
            "remember that story",
        ]
        return any(p in stripped for p in phrases)

    # ------------------------------
    # Helper to build speech events
    # ------------------------------
    
    def _speech(self, text: str, channel: str, style: str, event: Event) -> Event:
        return Event(
            topic="act/speech",
            payload={
                "text": text,
                "channel": channel,
                "style": style,
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"kind": "router_reply"},
        )
        
def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="text_router",
        subscribed_topics=["percept/text", "act/speech"],
        output_topics=["reason/request", "act/speech"],
        priority=5,
    )
    yield TextRouterNeuron(cfg)
