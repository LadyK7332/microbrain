from __future__ import annotations

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
        - "act/speech"     for fast, low-latency responses
        - "reason/request" for heavier LLM reasoning

    Routing rules (v1, simple):

    1) Slash commands (text starting with "/"):
       - /echo <msg>     → act/speech with <msg>
       - /help           → act/speech with a help blurb
       - unknown /cmd    → small error message

    2) Short greetings:
       - "hi", "hello", "hey", "yo" (case-insensitive)
         → quick friendly reply via act/speech

    3) Everything else:
       → forward to LLMReasoner via "reason/request"
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
                    'Available commands:\n  /echo <text>            - echo back text quickly\n  /help                   - show this help\n  /status                 - describe my current brain state\n  /reflect                - reflect on my own status\n  /why                    - explain why I answered the way I did\n  /cookie                 - feed me a virtual snack (+5% power)\n  /power                  - show current power status\n  /power pct <0-100>      - set battery percent\n  /power set <0-100>      - alias for /power pct\n  /power charging on|off  - toggle charging state\n  /power sleep on|off     - toggle sleep state\n  /read on|off            - toggle reading mode\n  /read status            - show reading progress\n  /read next              - force one reading chunk\n  /user <name>            - set your display name\n  /user clear             - clear saved display name\n  /vision list            - list capturable windows\n  /vision select <x>      - select window by index or title\n  /vision on|off          - enable or disable vision capture\n  /vision preview on|off  - toggle vision preview\n  /focus center           - center the vision focus point\n  /focus <x> <y>          - set normalized focus coordinates\n  /as_audio <text>        - inject text as microphone-style input\n  /acc <n> <text>         - send text with explicit tone/intensity (-10..10)\n  /r u <n>                - snapshot recent user items for scoring\n  /r a <n>                - snapshot recent assistant items for scoring\n  /r +W <i>               - reinforce snapshot item with weight\n  /r clear                - close reinforcement snapshot\nOther messages are sent to my reasoning core.'
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
                        meta={"control": True, "kind": "power_control", "feed": "cookie"},
                    ),
                    self._speech(
                        "Cookie received. Power +5.0%.",
                        channel=channel,
                        style="system",
                        event=event,
                    ),
                ]


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
        greeting_match = await self._match_greeting(ctx, lowered)
        if bool(greeting_match.get("matched", False)):
            # Prefer the /user display name when available
            user_name = await ctx.get_kv("profile:user_name", None)
            user_label = str(user_name).strip() if user_name else "there"

            # Learned greeting aliases should stay snappy even after warmup.
            if bool(greeting_match.get("learned", False)):
                await ctx.log_debug(
                    f"[{self.name}] Handling learned greeting alias locally",
                    text=text,
                    channel=channel,
                    concept=str(greeting_match.get("concept", "") or ""),
                )
                return [self._speech(
                    f"Hey, {user_label}! What's up?",
                    channel=channel,
                    style="assistant",
                    event=event,
                )]

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
                f"[{self.name}] Greeting forwarded to LLM (post-warmup)",
                text=text,
                channel=channel,
                semantic_n=semantic_n,
                warmup_max=warmup_max,
            )
            # fall through to default forwarding

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

    async def _match_greeting(self, ctx, lowered: str) -> Dict[str, Any]:
        norm = self._normalize_token(lowered)
        builtins = {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}
        if norm in builtins:
            return {"matched": True, "learned": False, "normalized": norm}

        aliases = await ctx.get_kv("router:greeting_aliases", {}) or {}
        if isinstance(aliases, dict) and norm in aliases:
            return {"matched": True, "learned": True, "normalized": norm, "concept": str(aliases.get(norm, "") or "")}

        return {"matched": False, "learned": False, "normalized": norm}

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
