"""Textual UI frontend for MicroBrain.

This module is UI-only. The orchestrator integration lives in `textual_bridge.py`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog


@dataclass
class UIMessage:
    topic: str
    payload: object
    source: str = ""


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
    ) -> None:
        super().__init__()
        self._send_cb = send_cb
        self._recv_q = recv_q

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="log", highlight=True, markup=True)
            yield Input(placeholder="Type here…", id="input")
        yield Footer()

    async def on_mount(self) -> None:
        # Poll for inbound messages without blocking the UI loop.
        self.set_interval(0.05, self._drain_recv_queue)

        # A tiny hello so it's obvious the UI started.
        self.query_one("#log", RichLog).write("[b]MicroBrain UI online.[/b]  (/quit to close)")

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

            if msg.topic == "act/speech" and text is not None:
                log.write(f"[cyan]mb>[/cyan] {text}")
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
        log.write(f"[green]you>[/green] {text}")

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
