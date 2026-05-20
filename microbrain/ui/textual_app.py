"""Textual UI frontend for MicroBrain.

This module is UI-only. The orchestrator integration lives in `textual_bridge.py`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from microbrain.utils.memdir import resolve_memdir_cli

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog


@dataclass
class UIMessage:
    topic: str
    payload: object
    source: str = ""
    meta: dict | None = None


SendCallback = Callable[[str], Awaitable[None]]


class MicroBrainUI(App):
    """A small Textual UI: log + input.

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
    #log {
        height: 1fr;
        border: round $primary;
        margin: 1 2;
        padding: 1 1;
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="log", highlight=True, markup=True)
            yield Input(placeholder="Type here…", id="input")
        yield Footer()

    def _load_labels_from_memdir(self) -> None:
        """Load assistant/user display labels from memdir JSON files."""
        try:
            memdir = Path(self._memdir) if self._memdir else resolve_memdir_cli(None)
        except Exception:
            memdir = Path.cwd() / "memory"

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

    async def on_mount(self) -> None:
        # Poll for inbound messages without blocking the UI loop.
        self.set_interval(0.05, self._drain_recv_queue)

        # A tiny hello so it's obvious the UI started.
        self._load_labels_from_memdir()

        # A tiny hello so it's obvious the UI started.
        self.query_one("#log", RichLog).write(
            f"[b]{self._assistant_label} UI online.[/b]  (/quit to close)"
        )

    async def _drain_recv_queue(self) -> None:
        if self._recv_q is None:
            return
        log = self.query_one("#log", RichLog)
        drained = 0
        while drained < 50:
            try:
                msg = self._recv_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1

            # Common patterns: act/speech payload can be dict or string.
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
            effective_source = payload_source or transport_source

            if msg.topic in ("ui/error", "control/error") and text is not None:
                log.write(f"[red]error>[/red] {text}")
            elif msg.topic in ("ui/status", "control/status") and text is not None:
                log.write(f"[dim]status>[/dim] {text}")
            elif msg.topic == "act/speech" and effective_channel == "thought" and text is not None:
                log.write(f"[magenta]thought>[/magenta] {text}")
            elif msg.topic == "act/speech" and text is not None:
                log.write(f"[cyan]{self._assistant_label}>[/cyan] {text}")
            elif msg.topic == "reason/output" and text is not None:
                log.write(f"[magenta]thought>[/magenta] {text}")
            elif msg.topic == "reason/request" and effective_source == "internal" and text is not None:
                log.write(f"[magenta]think?[/magenta] {text}")
            else:
                # Keep it lightweight; show topic and a short payload preview.
                preview = str(payload)
                if len(preview) > 240:
                    preview = preview[:240] + "…"
                log.write(f"[dim]{msg.topic}[/dim] {preview}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        self.query_one("#input", Input).value = ""
        if not text:
            return

        log = self.query_one("#log", RichLog)
        
        # Local UI label update for /user so it feels immediate (no restart needed)
        if text.lower().startswith("/user "):
            new_name = text.split(" ", 1)[1].strip().strip('"').strip("'")
            if new_name and new_name.lower() not in ("clear", "reset", "none", "off"):
                self._user_label = new_name.replace("[", "(").replace("]", ")")
            elif new_name.lower() in ("clear", "reset", "none", "off"):
                self._user_label = "you"

        log.write(f"[green]{self._user_label}>[/green] {text}")

        # Local quit command so you can always bail out, even if neurons are weird.
        if text.lower() in {"/quit", "/exit"}:
            self.exit()
            return

        if self._send_cb is None:
            log.write("[red]No send callback wired.[/red]")
            return

        # Fire-and-forget: don't block the UI thread.
        async def _send() -> None:
            try:
                await self._send_cb(text)
            except Exception as e:  # noqa: BLE001
                log.write(f"[red]send error:[/red] {e!r}")

        asyncio.create_task(_send())
