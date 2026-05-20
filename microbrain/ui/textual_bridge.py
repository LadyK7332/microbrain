"""Bridge between the orchestrator and the Textual UI."""

from __future__ import annotations

import asyncio
from typing import Any

from microbrain.orchestrator.event_bus import Event
from microbrain.orchestrator.orchestrator import Orchestrator

from .textual_app import MicroBrainUI, UIMessage


async def run_textual_frontend(orch: Orchestrator, *, memdir: str | None = None) -> None:
    """Run Textual UI connected to an already-started orchestrator."""

    recv_q: asyncio.Queue[UIMessage] = asyncio.Queue(maxsize=500)

    async def _ui_tap(ev: Event) -> list[Event]:
        # Drop noisy internal ticks by default; UI would spam.
        if ev.topic == "clock/tick":
            return []
        try:
            recv_q.put_nowait(
                UIMessage(topic=ev.topic, payload=ev.payload, source=ev.source, meta=dict(ev.meta or {}))
            )
        except asyncio.QueueFull:
            # Best-effort: if UI can't keep up, drop oldest by draining a little.
            try:
                _ = recv_q.get_nowait()
                recv_q.put_nowait(
                    UIMessage(topic=ev.topic, payload=ev.payload, source=ev.source, meta=dict(ev.meta or {}))
                )
            except Exception:
                pass
        return []

    # Subscribe to the stuff a human cares about.
    # - act/speech: assistant output
    # - vision/status: window grabber status messages
    # - control/vision: confirmations can be emitted elsewhere; still useful
    topics = [
        "act/speech",
        "ui/status",
        "ui/error",
        "control/status",
        "control/error",
        "reason/request",
        "reason/output",
        "vision/status",
        "vision/focus",
        "control/vision",
        "control/focus",
    ]
    # EventBus signature is: subscribe(name, topics, handler, priority=0)
    sub_id = orch.bus.subscribe(
        "ui.textual.tap",
        topics,
        _ui_tap,
        priority=0,
    )
    async def _send_text(text: str) -> None:
        await orch.push_event(
            "input/text",
            text,
            meta={"source": "ui", "channel": "textual"},
        )
        # Let neurons chew; if something is stuck, UI should remain responsive anyway.
        await orch.wait_for_idle(timeout=30.0)

    app = MicroBrainUI(send_cb=_send_text, recv_q=recv_q, memdir=memdir)
    await app.run_async()

    # When UI closes, best-effort unsubscribe.
    try:
        orch.bus.unsubscribe(sub_id)
    except Exception:
        pass
