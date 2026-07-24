"""Qt/asyncio bootstrap for the native MicroBrain dashboard."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# No behavioral knobs are currently required here.  Qt timing lives with the
# dashboard app/bridge where it can be inspected beside the behavior it affects.

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

MISSING_QT_MESSAGE = (
    "Dashboard UI requires optional packages PySide6 and qasync. "
    "Install with: python -m pip install -r requirements-dashboard.txt"
)


def run_qt_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an asyncio coroutine on Qt's event loop in the main thread."""

    try:
        from PySide6.QtWidgets import QApplication
        from qasync import QEventLoop
    except ModuleNotFoundError as exc:
        try:
            coro.close()
        except Exception:
            pass
        raise SystemExit(MISSING_QT_MESSAGE) from exc

    app = QApplication.instance() or QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        return loop.run_until_complete(coro)


async def run_dashboard_frontend(orch, *, memdir: str | None = None) -> None:
    """Start the two native windows and wait until the Qt application exits."""

    from PySide6.QtWidgets import QApplication

    from .app import create_dashboard
    from .bridge import DashboardBridge

    bridge = DashboardBridge(orch, memdir=memdir)
    await bridge.start()
    controller = create_dashboard(bridge, memdir=memdir)
    # Keep an explicit reference for the lifetime of the coroutine.
    _ = controller

    loop = asyncio.get_running_loop()
    closed = loop.create_future()

    def _quit() -> None:
        if not closed.done():
            closed.set_result(None)

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication was not created before dashboard startup")
    app.aboutToQuit.connect(_quit)
    try:
        await closed
    finally:
        await bridge.stop()
