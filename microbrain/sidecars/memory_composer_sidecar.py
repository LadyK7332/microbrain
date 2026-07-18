from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from microbrain.memory.mem_cell_composer import MemCellComposer

logger = logging.getLogger(__name__)


class MemoryComposerSidecar:
    """Background single-writer mem-cell composer.

    Other organs stage pending memory updates.  This sidecar periodically drains
    those updates and owns the canonical mem_cell/<tier>/<tier>.jsonl writes.
    """

    def __init__(self, orchestrator: Any, *, memdir: str | Path, interval_s: float = 2.0):
        self.orchestrator = orchestrator
        self.memdir = Path(memdir)
        self.interval_s = max(0.25, float(interval_s))
        self.composer = MemCellComposer(self.memdir)
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run_loop(), name="memory_composer_sidecar")
        logger.info("Memory composer sidecar started.")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while not self._stopping:
            try:
                status = await asyncio.to_thread(self.composer.compose_once)
                await self._publish_status(status)
            except TimeoutError:
                # Another composer is active. Fine; that is the point.
                pass
            except Exception:
                logger.exception("Memory composer cycle failed")
                try:
                    self.orchestrator.kv_store["mem_cell:composer:last_error"] = repr(time.time())
                except Exception:
                    pass
            await asyncio.sleep(self.interval_s)

    async def _publish_status(self, status: dict[str, Any]) -> None:
        try:
            kv = self.orchestrator.kv_store
            kv["mem_cell:composer:last_status"] = status
            kv["mem_cell:composer:last_seen_ts"] = time.time()
            kv["mem_cell:composer:pending_count"] = self.composer.pending_count()
        except Exception:
            pass
