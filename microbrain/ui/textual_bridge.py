"""Bridge between the orchestrator and the Textual UI."""

from __future__ import annotations

import asyncio

from microbrain.orchestrator.event_bus import Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import is_infrastructure_event

from .textual_app import MicroBrainUI, UIMessage


from .frontend_common import pressure_snapshot as _pressure_snapshot


async def run_textual_frontend(orch: Orchestrator, *, memdir: str | None = None) -> None:
    """Run Textual UI connected to an already-started orchestrator."""

    recv_q: asyncio.Queue[UIMessage] = asyncio.Queue(maxsize=500)

    async def _ui_tap(ev: Event) -> list[Event]:
        # Drop noisy internal ticks by default; UI would spam.
        if is_infrastructure_event(ev):
            return []
        meta = dict(ev.meta or {})
        # The Textual face should not show internal reasoning/request plumbing
        # unless a specific event opts into UI visibility. The log inspector can
        # still watch the firehose from microbrain.log.
        if meta.get("ui_hidden") is True or meta.get("ui_visible") is False:
            return []
        if ev.topic in {"reason/request", "reason/output"} and meta.get("ui_visible") is not True:
            return []
        try:
            recv_q.put_nowait(
                UIMessage(topic=ev.topic, payload=ev.payload, source=ev.source, meta=meta)
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

    async def _pressure_pump() -> None:
        while True:
            try:
                recv_q.put_nowait(
                    UIMessage(
                        topic="ui/pressure_state",
                        payload=_pressure_snapshot(orch),
                        source="ui.pressure_sampler",
                        meta={"ui_hidden": True, "store_in_memory": False},
                    )
                )
            except asyncio.QueueFull:
                try:
                    _ = recv_q.get_nowait()
                except Exception:
                    pass
            except Exception:
                pass
            await asyncio.sleep(0.25)

    pressure_task = asyncio.create_task(_pressure_pump(), name="ui_pressure_band_sampler")

    app = MicroBrainUI(send_cb=_send_text, recv_q=recv_q, memdir=memdir)
    try:
        await app.run_async()
    finally:
        pressure_task.cancel()
        try:
            await pressure_task
        except asyncio.CancelledError:
            pass

    # When UI closes, best-effort unsubscribe.
    try:
        orch.bus.unsubscribe(sub_id)
    except Exception:
        pass
