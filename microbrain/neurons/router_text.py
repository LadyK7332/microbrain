from __future__ import annotations

import time

from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


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
        - "context/request" for the contextual/thought pipeline
        - "ui/status" / "ui/error" for command-plane diagnostics

    Routing rules:

    1) Slash commands (text starting with "/") stay on the command plane.

    2) Natural language, including greetings, goes through context/thought.
       The old fast canned greeting scaffold was removed once the line was live.
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

        source = str(payload.get("source", "user"))
        channel = str(payload.get("channel", "default"))
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        raw_meta = payload.get("raw_meta", {}) or {}
        source = raw_meta.get("source", "cli")
        channel = raw_meta.get("channel", "default")
        text = str(payload.get("text", "")).strip()

        lowered = text.lower()

        # Track last user input for explanations, but avoid pure /commands and "why?" probes
        if not text.startswith("/") and not self._looks_like_why_question(lowered):
            self._last_user_text = text
            self._last_user_meta = {
                "channel": channel,
                "source": source,
            }
            
        # ------------------------------
        # 1) Slash commands
        # ------------------------------
        if text.startswith("/"):
            cmd_line = text[1:].strip()
            if not cmd_line:
                return [self._ui_error(
                    "Empty command. Try /help.",
                    channel=channel,
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
                return [self._ui_status(
                    message,
                    channel=channel,
                    event=event,
                    kind="command_echo",
                )]

            if cmd in ("help", "?"):
                help_text = (
                    "Available commands:\n"
                    "  /echo <text>            - echo back text quickly\n"
                    "  /help                   - show this help\n"
                    "  /quit or /exit          - close the Textual UI\n"
                    "  /status                 - describe my current brain state\n"
                    "  /reflect                - reflect on my own status\n"
                    "  /why                    - explain why I answered the way I did\n"
                    "  /cookie                 - feed me a virtual snack (+5% power)\n"
                    "  /power                  - show current power status\n"
                    "  /power pct <0-100>      - set battery percent\n"
                    "  /power set <0-100>      - alias for /power pct\n"
                    "  /power charging on|off  - toggle charging state\n"
                    "  /power sleep on|off     - toggle sleep state\n"
                    "  /read on|off            - toggle reading mode\n"
                    "  /read status            - show reading progress\n"
                    "  /read next              - force one reading chunk\n"
                    "  /slearn on|off          - toggle structured CAPS learning sheets\n"
                    "  /slearn status          - show structured learning progress\n"
                    "  /slearn next            - force one structured learning chunk\n"
                    "  /slearn dir <path>      - set structured learning sheet folder\n"
                    "  /slearn template        - show sheet format examples\n"
                    "  /user <name>            - set your display name\n"
                    "  /user clear             - clear saved display name\n"
                    "  /vision list            - list capturable windows\n"
                    "  /vision select <x>      - select window by index or title\n"
                    "  /vision on|off          - enable or disable window vision capture\n"
                    "  /vision preview on|off  - toggle window vision preview\n"
                    "  /camera list            - list webcam/camera devices\n"
                    "  /camera select <index>  - select webcam by index\n"
                    "  /camera on|off          - enable or disable webcam capture\n"
                    "  /camera preview on|off  - toggle webcam preview\n"
                    "  /focus center           - center the vision focus point\n"
                    "  /focus <x> <y>          - set normalized focus coordinates\n"
                    "  /as_audio <text>        - inject text as microphone-style input\n"
                    "  /acc <n> <text>         - send text with signed tone (-10..10: +=positive, -=correction)\n"
                    "  /t                      - arm trainer correction for the last assistant utterance\n"
                    "  /t status               - show trainer latch status\n"
                    "  /t cancel               - cancel trainer correction\n"
                    "  /r u <n>                - snapshot recent user items for scoring\n"
                    "  /r a <n>                - snapshot recent assistant items for scoring\n"
                    "  /r +W <i> or /r -W <i>  - score snapshot item with weight\n"
                    "  /r +W <i> \"IF ...\"     - score item and attach structured teaching note\n"
                    "  /r clear                - close reinforcement snapshot\n"
                    "Other messages are sent to my reasoning core.\n"
                    "Note: /vision is window/screen capture; /camera is webcam capture."
                )
                await ctx.log_debug(
                    f"[{self.name}] Handling /help locally",
                    channel=channel,
                )
                return [self._ui_status(
                    help_text,
                    channel=channel,
                    event=event,
                    kind="command_help",
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
            if cmd == "cookie":
                await ctx.log_debug(
                    f"[{self.name}] Handling /cookie locally",
                    channel=channel,
                )
                return [
                    Event(
                        topic="control/power",
                        payload={"add_pct": 5.0, "reason": "cookie"},
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={**self._control_meta(kind="power_control"), "feed": "cookie"},
                    ),
                    self._ui_status(
                        "Cookie received. Power +5.0%.",
                        channel=channel,
                        event=event,
                        kind="command_power",
                    ),
                ]


            if cmd in ("as_audio", "audio"):
                # Treat the rest of the line as if it came from microphone STT.
                spoken_text = arg.strip()
                if not spoken_text:
                    return [self._ui_error(
                        "Usage: /as_audio <what you would have said out loud>",
                        channel=channel,
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

                return [
                    audio_event,
                    self._ui_status(
                        "Injected text as microphone-style input.",
                        channel=channel,
                        event=event,
                        kind="command_as_audio",
                    ),
                ]

            
            if cmd == "vision":
                sub = (arg or "").strip()
                if not sub:
                    return [self._ui_error(
                        "Usage: /vision list | /vision select <n|title> | /vision on|off | /vision preview on|off",
                        channel=channel,
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
                        return [self._ui_error(
                            f"Vision list failed: {e!r}",
                            channel=channel,
                            event=event,
                        )]

                    if not wins:
                        return [self._ui_status(
                            "No windows found.",
                            channel=channel,
                            event=event,
                            kind="vision_status",
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
                    return [self._ui_status("\n".join(lines), channel=channel, event=event, kind="vision_status")]

                if subcmd == "select":
                    if not rest:
                        return [self._ui_error(
                            "Usage: /vision select <index|title_substring>",
                            channel=channel,
                            event=event,
                        )]
                    return [
                        Event(
                            topic="control/vision",
                            payload={"action": "select", "selector": rest},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta=self._control_meta(kind="vision_control"),
                        ),
                        self._ui_status(
                            f"Selecting window: {rest!r}",
                            channel=channel,
                            event=event,
                            kind="vision_status",
                        ),
                    ]

                if subcmd in ("on", "off"):
                    return [
                        Event(
                            topic="control/vision",
                            payload={"action": subcmd},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta=self._control_meta(kind="vision_control"),
                        ),
                        self._ui_status(
                            f"Vision {subcmd}.",
                            channel=channel,
                            event=event,
                            kind="vision_status",
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
                                meta=self._control_meta(kind="vision_control"),
                            ),
                            self._ui_status(
                                f"Vision preview {rest.lower()}.",
                                channel=channel,
                                event=event,
                                kind="vision_status",
                            ),
                        ]
                    return [self._ui_error(
                        "Usage: /vision preview on|off",
                        channel=channel,
                        event=event,
                    )]

                return [self._ui_error(
                    "Unknown /vision subcommand. Try: list, select, on, off, preview on|off",
                    channel=channel,
                    event=event,
                )]

            if cmd == "camera":
                sub = (arg or "").strip()
                if not sub:
                    enabled = bool(await ctx.get_kv("camera:enabled", False))
                    selected = await ctx.get_kv("camera:selected", None)
                    selected_text = "-"
                    if isinstance(selected, dict):
                        selected_text = f"{selected.get('index', '?')}: {selected.get('name', 'Camera')}"
                    msg = (
                        f"camera: {'on' if enabled else 'off'} | selected={selected_text}\n"
                        "Usage: /camera list | /camera select <index> | /camera on|off | /camera preview on|off | /camera status"
                    )
                    return [self._ui_status(msg, channel=channel, event=event, kind="camera_status")]

                parts2 = sub.split()
                subcmd = parts2[0].lower()
                rest = " ".join(parts2[1:]).strip()

                if subcmd in ("status", "state"):
                    enabled = bool(await ctx.get_kv("camera:enabled", False))
                    preview = bool(await ctx.get_kv("camera:preview", False))
                    selected = await ctx.get_kv("camera:selected", None)
                    selected_text = "-"
                    if isinstance(selected, dict):
                        selected_text = f"{selected.get('index', '?')}: {selected.get('name', 'Camera')}"
                    fps = await ctx.get_kv("camera:fps", 2.0)
                    msg = f"camera: {'on' if enabled else 'off'} | preview={'on' if preview else 'off'} | selected={selected_text} | fps={fps}"
                    return [self._ui_status(msg, channel=channel, event=event, kind="camera_status")]

                if subcmd == "list":
                    try:
                        from microbrain.utils.mb_vision.camera_grabber import list_cameras
                        cams = list_cameras()
                        try:
                            await ctx.set_kv("camera:devices_last", [c.as_dict() for c in cams])
                        except Exception:
                            pass
                    except Exception as e:
                        return [self._ui_error(f"Camera list failed: {e!r}", channel=channel, event=event)]

                    if not cams:
                        return [self._ui_status(
                            "No cameras found. Install/check opencv-python and camera permissions if this seems wrong.",
                            channel=channel,
                            event=event,
                            kind="camera_status",
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
                    return [self._ui_status("\n".join(lines), channel=channel, event=event, kind="camera_status")]

                if subcmd == "select":
                    if not rest:
                        return [self._ui_error("Usage: /camera select <index>", channel=channel, event=event)]
                    try:
                        idx = int(rest)
                    except Exception:
                        return [self._ui_error("Usage: /camera select <index>  (index must be a number)", channel=channel, event=event)]
                    if idx < 0 or idx > 32:
                        return [self._ui_error("Camera index out of safe probe range 0..32.", channel=channel, event=event)]

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
                            correlation_id=event.correlation_id,
                            meta=self._control_meta(kind="camera_control"),
                        ),
                        self._ui_status(
                            f"Selected camera: {idx} ({chosen.get('name', 'Camera')}).",
                            channel=channel,
                            event=event,
                            kind="camera_status",
                        ),
                    ]

                if subcmd in ("on", "off"):
                    return [
                        Event(
                            topic="control/camera",
                            payload={"action": subcmd},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta=self._control_meta(kind="camera_control"),
                        ),
                        self._ui_status(f"Camera {subcmd}.", channel=channel, event=event, kind="camera_status"),
                    ]

                if subcmd == "preview":
                    if rest.lower() in ("on", "off"):
                        act = "preview_on" if rest.lower() == "on" else "preview_off"
                        return [
                            Event(
                                topic="control/camera",
                                payload={"action": act},
                                source=self.name,
                                correlation_id=event.correlation_id,
                                meta=self._control_meta(kind="camera_control"),
                            ),
                            self._ui_status(f"Camera preview {rest.lower()}.", channel=channel, event=event, kind="camera_status"),
                        ]
                    return [self._ui_error("Usage: /camera preview on|off", channel=channel, event=event)]

                return [self._ui_error(
                    "Unknown /camera subcommand. Try: list, select, on, off, preview on|off, status",
                    channel=channel,
                    event=event,
                )]

            if cmd == "focus":
                sub = (arg or "").strip()
                if not sub:
                    return [self._ui_error(
                        "Usage: /focus center | /focus <x> <y> (normalized 0..1)",
                        channel=channel,
                        event=event,
                    )]

                if sub.lower() == "center":
                    return [
                        Event(
                            topic="control/focus",
                            payload={"action": "center"},
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta=self._control_meta(kind="focus_control"),
                        ),
                        self._ui_status("Focus set to center.", channel=channel, event=event),
                    ]

                pts = sub.split()
                if len(pts) != 2:
                    return [self._ui_error(
                        "Usage: /focus <x> <y> (normalized 0..1)",
                        channel=channel,
                        event=event,
                    )]
                try:
                    x = float(pts[0])
                    y = float(pts[1])
                except Exception:
                    return [self._ui_error(
                        "Couldn't parse numbers. Example: /focus 0.5 0.5",
                        channel=channel,
                        event=event,
                    )]

                return [
                    Event(
                        topic="control/focus",
                        payload={"action": "set", "x": x, "y": y},
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta=self._control_meta(kind="focus_control"),
                    ),
                    self._ui_status(f"Focus set to ({x:.3f}, {y:.3f}).", channel=channel, event=event),
                ]

# Unknown command
            await ctx.log_debug(
                f"[{self.name}] Unknown command, responding locally",
                cmd=cmd,
                channel=channel,
            )
            return [self._ui_error(
                f"I don't know the command '/{cmd}'. Try /help.",
                channel=channel,
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
        # 2) Natural language, including greetings, goes through thought/context.
        # ------------------------------

        # ------------------------------
        # 3) Default: forward to contextual builder
        # ------------------------------
        await ctx.log_debug(
            f"[{self.name}] Forwarding to contextual builder",
            channel=channel,
        )

        reason_payload = {
            "text": text,
            "source": source,
            "channel": channel,
            "raw_meta": raw_meta,
        }

        reason_event = Event(
            topic="context/request",
            payload=reason_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"routed": True},
        )

        return [reason_event]

    # ------------------------------
    def _normalize_token(self, text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

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

    def _ui_status(self, text: str, channel: str, event: Event, kind: str = "command_status") -> Event:
        return Event(
            topic="ui/status",
            payload={
                "text": text,
                "channel": channel,
                "style": "system",
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta=self._control_meta(kind=kind),
        )

    def _ui_error(self, text: str, channel: str, event: Event) -> Event:
        return Event(
            topic="ui/error",
            payload={
                "text": text,
                "channel": channel,
                "style": "system",
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta=self._control_meta(kind="command_error"),
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="text_router",
        subscribed_topics=["percept/text", "act/speech"],
        output_topics=["context/request", "ui/status", "ui/error"],
        priority=5,
    )
    yield TextRouterNeuron(cfg)
