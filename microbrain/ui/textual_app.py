"""Textual UI frontend for MicroBrain.

This module is UI-only. The orchestrator integration lives in `textual_bridge.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from microbrain.utils.memdir import resolve_memdir_cli

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static


@dataclass
class UIMessage:
    topic: str
    payload: object
    source: str = ""
    meta: dict | None = None


SendCallback = Callable[[str], Awaitable[None]]


class MicroBrainUI(App):
    """A small Textual UI with a raw event pane, conversation pane, and input.

    Parameters
    ----------
    send_cb:
        Async callback called whenever the user submits a line.
    recv_q:
        Async queue of UIMessage items coming from the orchestrator.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #raw {
        height: 2fr;
        border: round $accent;
        margin: 1 2 0 2;
        padding: 1 1;
        overflow-x: hidden;
        overflow-y: auto;
    }
    #pressure {
        height: 4;
        border: round $warning;
        margin: 0 2 0 2;
        padding: 0 1;
    }
    #conversation {
        height: 3fr;
        border: round $primary;
        margin: 0 2 0 2;
        padding: 1 1;
        overflow-x: hidden;
        overflow-y: auto;
    }
    #input {
        margin: 0 2 1 2;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        send_cb: Optional[SendCallback] = None,
        recv_q: Optional[asyncio.Queue[UIMessage]] = None,
        memdir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._send_cb = send_cb
        self._recv_q = recv_q
        self._memdir = memdir

        # Speaker labels (loaded from memdir on mount)
        self._assistant_label = "MB"
        self._user_label = "you"

        # Transcript files are initialized on mount after memdir resolution.
        self._raw_log_path: Path | None = None
        self._conversation_log_path: Path | None = None

        # Two-line live teaching/body strip between raw trace and conversation.
        # This is UI instrumentation only: slow body condition + fast pressure pulse.
        self._pressure_prev: dict[str, float] = {}
        self._pressure_last_payload: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="raw", highlight=True, markup=True, wrap=True, min_width=1)
            yield Static("[b]body>[/b] starting | [b]pulse>[/b] waiting", id="pressure")
            yield RichLog(id="conversation", highlight=True, markup=True, wrap=True, min_width=1)
            yield Input(placeholder="Type here…", id="input")
        yield Footer()

    def _resolve_memdir(self) -> Path:
        try:
            return Path(self._memdir) if self._memdir else resolve_memdir_cli(None)
        except Exception:
            return Path.cwd() / "memory"

    def _init_transcript_paths(self) -> None:
        try:
            memdir = self._resolve_memdir()
            log_dir = memdir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._raw_log_path = log_dir / "textual_raw.jsonl"
            self._conversation_log_path = log_dir / "textual_conversation.log"
        except Exception:
            self._raw_log_path = None
            self._conversation_log_path = None

    def _append_text(self, path: Path | None, line: str) -> None:
        if path is None:
            return
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line.rstrip("\n") + "\n")
        except Exception:
            # UI logging should never break the face.
            pass

    def _safe_json(self, value: object) -> object:
        """Return JSON-safe-ish values without throwing on odd payload objects."""
        try:
            json.dumps(value)
            return value
        except Exception:
            return repr(value)

    def _append_raw_event(self, msg: UIMessage) -> None:
        record = {
            "ts": time.time(),
            "topic": msg.topic,
            "source": msg.source,
            "payload": self._safe_json(msg.payload),
            "meta": self._safe_json(msg.meta or {}),
        }
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        except Exception:
            line = repr(record)
        self._append_text(self._raw_log_path, line)

    def _append_conversation_line(self, line: str) -> None:
        self._append_text(self._conversation_log_path, line)

    def _load_labels_from_memdir(self) -> None:
        """Load assistant/user display labels from memdir JSON files."""
        memdir = self._resolve_memdir()

        # Assistant label from PDNA profile
        try:
            pdna_path = memdir / "pdna_profile.json"
            if pdna_path.exists():
                data = json.loads(pdna_path.read_text(encoding="utf-8"))
                name = str(data.get("name", "") or "").strip()
                if name:
                    self._assistant_label = name
        except Exception:
            pass

        # User label from /user persistent profile (optional)
        try:
            user_path = memdir / "state" / "user_profile.json"
            if user_path.exists():
                data = json.loads(user_path.read_text(encoding="utf-8"))
                uname = str(data.get("user_name", "") or "").strip()
                if uname:
                    self._user_label = uname
        except Exception:
            pass

        # Tiny sanitization so Textual markup can't get weird
        self._assistant_label = self._assistant_label.replace("[", "(").replace("]", ")")
        self._user_label = self._user_label.replace("[", "(").replace("]", ")")


    def _safe_label(self, value: object, default: str = "") -> str:
        text = str(value if value is not None else default).strip() or default
        return text.replace("[", "(").replace("]", ")")

    def _safe_num(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _trend(self, key: str, value: float) -> str:
        prev = self._pressure_prev.get(key)
        self._pressure_prev[key] = value
        if prev is None:
            return "[dim]~[/dim]"
        delta = value - prev
        if delta > 0.015:
            return "[green]↑[/green]"
        if delta < -0.015:
            return "[red]↓[/red]"
        return "[dim]~[/dim]"

    def _fmt_metric(self, key: str, label: str, value: object) -> str:
        num = max(0.0, min(1.0, self._safe_num(value, 0.0)))
        return f"{label} {num:.2f}{self._trend(key, num)}"

    def _render_pressure_band(self, payload: dict) -> str:
        body = payload.get("body", {}) if isinstance(payload.get("body"), dict) else {}
        pulse = payload.get("pulse", {}) if isinstance(payload.get("pulse"), dict) else {}

        power_mode = self._safe_label(body.get("power_mode"), "awake")
        charging = "chg:on" if bool(body.get("charging", False)) else "chg:off"
        sleep = "sleep:on" if bool(body.get("sleep", False)) else "sleep:off"
        maint = self._safe_label(body.get("maintenance"), "idle")
        mem = self._safe_label(body.get("memory_composer"), "off")
        read = self._safe_label(body.get("read_sidecar"), "off")
        pending = int(self._safe_num(body.get("memory_pending"), 0.0))
        cap_available = int(self._safe_num(body.get("cap_available"), 0.0))
        cap_total = int(self._safe_num(body.get("cap_total"), 0.0))
        drawer_ready = int(self._safe_num(body.get("drawer_ready"), 0.0))
        drawer_waiting = int(self._safe_num(body.get("drawer_waiting"), 0.0))

        body_line = (
            f"[b]body>[/b] pwr {power_mode} {charging} {sleep} | "
            f"maint {maint} | mem {mem} pend {pending} | read {read} | "
            f"cap {cap_available}/{cap_total} | drawer r{drawer_ready}/w{drawer_waiting}"
        )

        thought_intent = self._safe_label(pulse.get("thought_intent"), "idle")
        thought_status = self._safe_label(pulse.get("thought_status"), "idle")
        pulse_line = (
            "[b]pulse>[/b] "
            + " | ".join(
                [
                    self._fmt_metric("salience", "sal", pulse.get("salience")),
                    self._fmt_metric("reward", "dop", pulse.get("reward")),
                    self._fmt_metric("boredom", "bored", pulse.get("boredom")),
                    self._fmt_metric("curiosity", "cur", pulse.get("curiosity")),
                    self._fmt_metric("expression", "expr", pulse.get("expression")),
                    self._fmt_metric("trainer", "train", pulse.get("trainer")),
                    self._fmt_metric("thought_pressure", "think", pulse.get("thought_pressure")),
                ]
            )
            + f" | {thought_intent}/{thought_status}"
        )

        return body_line + "\n" + pulse_line

    def _write_pressure_state(self, msg: UIMessage) -> None:
        payload = msg.payload if isinstance(msg.payload, dict) else {}
        self._pressure_last_payload = payload
        try:
            band = self.query_one("#pressure", Static)
            band.update(self._render_pressure_band(payload))
        except Exception:
            # The face should never crash because an internal state packet is odd.
            pass

    async def on_mount(self) -> None:
        # Poll for inbound messages without blocking the UI loop.
        self.set_interval(0.05, self._drain_recv_queue)

        self._init_transcript_paths()
        self._load_labels_from_memdir()

        raw = self.query_one("#raw", RichLog)
        conv = self.query_one("#conversation", RichLog)
        raw.write("[b]raw bus/event trace[/b]  (top pane, x-ray view)")
        self.query_one("#pressure", Static).update(
            "[b]body>[/b] pwr starting | maint unknown | mem unknown | read unknown\n"
            "[b]pulse>[/b] sal -- | dop -- | bored -- | cur -- | expr -- | train -- | think --"
        )
        conv.write(f"[b]{self._assistant_label} UI online.[/b]  (/quit to close)")
        self._append_conversation_line(f"{self._assistant_label} UI online. (/quit to close)")

    def _extract_text_and_channels(self, msg: UIMessage) -> tuple[str | None, str, str]:
        payload = msg.payload
        text = None
        if isinstance(payload, dict) and "text" in payload:
            text = str(payload.get("text", ""))
        elif isinstance(payload, str):
            text = payload

        meta = msg.meta or {}
        channel = str(meta.get("channel", "") or "")
        payload_channel = str(payload.get("channel", "") or "") if isinstance(payload, dict) else ""
        payload_source = str(payload.get("source", "") or "") if isinstance(payload, dict) else ""
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload, dict) and isinstance(payload.get("raw_meta"), dict) else {}
        transport_source = str(raw_meta.get("transport_source", raw_meta.get("source", "")) or "")
        effective_channel = payload_channel or channel
        effective_source = payload_source or transport_source or str(msg.source or "")
        return text, effective_channel, effective_source

    def _raw_preview(self, msg: UIMessage) -> str:
        payload = msg.payload
        meta = msg.meta or {}
        try:
            preview = json.dumps(
                {
                    "payload": self._safe_json(payload),
                    "meta": self._safe_json(meta),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            preview = str({"payload": payload, "meta": meta})
        # The raw pane is meant to be the x-ray view. Keep it visually full by
        # default, while allowing a safety cap for giant payloads if needed.
        try:
            max_preview = int(os.environ.get("MB_UI_RAW_PREVIEW_MAX", "0") or "0")
        except Exception:
            max_preview = 0
        if max_preview > 0 and len(preview) > max_preview:
            preview = preview[:max_preview] + "…"
        source = msg.source or ""
        source_part = f" src={source}" if source else ""
        return f"[dim]{msg.topic}[/dim]{source_part} {preview}"

    def _write_raw_message(self, msg: UIMessage) -> None:
        raw = self.query_one("#raw", RichLog)
        self._append_raw_event(msg)
        raw.write(self._raw_preview(msg))

    def _should_show_in_conversation(self, msg: UIMessage, text: str | None) -> bool:
        if text is None:
            return False
        meta = msg.meta or {}
        if bool(meta.get("ui_hidden", False)):
            return False
        if meta.get("ui_visible") is False:
            return False

        payload = msg.payload
        if isinstance(payload, dict):
            if bool(payload.get("ui_hidden", False)):
                return False
            if payload.get("ui_visible") is False:
                return False

        # Main conversation pane is skin, not x-ray. Internal reasoning stays raw-only
        # unless a neuron explicitly opts into ui_visible=True.
        if msg.topic in {"reason/request", "reason/output"} and not bool(meta.get("ui_visible", False)):
            return False
        return True

    def _write_conversation_message(self, msg: UIMessage) -> None:
        conv = self.query_one("#conversation", RichLog)
        text, effective_channel, effective_source = self._extract_text_and_channels(msg)
        if not self._should_show_in_conversation(msg, text):
            return

        if msg.topic in ("ui/error", "control/error"):
            rendered = f"[red]error>[/red] {text}"
            plain = f"error> {text}"
        elif msg.topic in ("ui/status", "control/status"):
            rendered = f"[dim]status>[/dim] {text}"
            plain = f"status> {text}"
        elif msg.topic == "thought/internal":
            rendered = f"[magenta]thought>[/magenta] {text}"
            plain = f"thought> {text}"
        elif msg.topic == "act/speech" and effective_channel == "thought":
            rendered = f"[magenta]thought>[/magenta] {text}"
            plain = f"thought> {text}"
        elif msg.topic == "act/speech":
            rendered = f"[cyan]{self._assistant_label}>[/cyan] {text}"
            plain = f"{self._assistant_label}> {text}"
        elif msg.topic in {"reason/request", "reason/output"} and effective_source == "internal":
            rendered = f"[magenta]think>[/magenta] {text}"
            plain = f"think> {text}"
        else:
            # Status-like non-speech messages may still show if explicit ui_visible=True.
            preview = str(msg.payload)
            if len(preview) > 240:
                preview = preview[:240] + "…"
            rendered = f"[dim]{msg.topic}[/dim] {preview}"
            plain = f"{msg.topic} {preview}"

        conv.write(rendered)
        self._append_conversation_line(plain)

    async def _drain_recv_queue(self) -> None:
        if self._recv_q is None:
            return
        drained = 0
        while drained < 50:
            try:
                msg = self._recv_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
            if msg.topic == "ui/pressure_state":
                self._write_pressure_state(msg)
                continue
            self._write_raw_message(msg)
            self._write_conversation_message(msg)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        self.query_one("#input", Input).value = ""
        if not text:
            return

        conv = self.query_one("#conversation", RichLog)
        raw = self.query_one("#raw", RichLog)

        # Local UI label update for /user so it feels immediate (no restart needed)
        if text.lower().startswith("/user "):
            new_name = text.split(" ", 1)[1].strip().strip('"').strip("'")
            if new_name and new_name.lower() not in ("clear", "reset", "none", "off"):
                self._user_label = new_name.replace("[", "(").replace("]", ")")
            elif new_name.lower() in ("clear", "reset", "none", "off"):
                self._user_label = "you"

        conv_line = f"{self._user_label}> {text}"
        conv.write(f"[green]{self._user_label}>[/green] {text}")
        self._append_conversation_line(conv_line)

        # Also write local user input into the raw pane/log immediately. The bus will
        # echo the resulting input/text event after send_cb, but this makes keystrokes
        # visible even if the brain stalls.
        local_msg = UIMessage(
            topic="ui/input_submitted",
            payload={"text": text},
            source="textual",
            meta={"source": "ui", "channel": "textual", "local_echo": True},
        )
        self._append_raw_event(local_msg)
        raw.write(self._raw_preview(local_msg))

        # Local quit command so you can always bail out, even if neurons are weird.
        if text.lower() in {"/quit", "/exit"}:
            self.exit()
            return

        if self._send_cb is None:
            conv.write("[red]No send callback wired.[/red]")
            self._append_conversation_line("error> No send callback wired.")
            return

        # Fire-and-forget: don't block the UI thread.
        async def _send() -> None:
            try:
                await self._send_cb(text)
            except Exception as e:  # noqa: BLE001
                conv.write(f"[red]send error:[/red] {e!r}")
                self._append_conversation_line(f"send error> {e!r}")

        asyncio.create_task(_send())
