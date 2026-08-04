from __future__ import annotations

import asyncio
import time
from typing import Any

from microbrain.utils.heartbeat_stream import (
    HEARTBEAT_HZ,
    HEARTBEAT_INTERVAL_S,
    PRIMARY_HEARTBEAT_TOPIC,
    heartbeat_meta,
    heartbeat_payload,
)

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Exponential smoothing weights for dashboard pacemaker telemetry. These do not
# alter the 20-TPS timing contract; they only make the displayed health values
# readable instead of flickering with every scheduler jitter sample.
HEARTBEAT_STATS_KEEP = 0.90
HEARTBEAT_STATS_NEW = 0.10

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

HEARTBEAT_STATS_SCHEMA = "body.heartbeat_stats.v2"
HEARTBEAT_TASK_NAME = "microbrain_body_heartbeat"


class BodyHeartbeatPacemaker:
    """One fixed-rate body pacemaker feeding the isolated infrastructure bus.

    Tick number is only a scheduling coordinate. Actual elapsed time is always
    measured from the event loop's monotonic clock. If the host stalls, missed
    opportunities are summarized in telemetry and discarded; no catch-up burst
    is emitted.
    """

    def __init__(self, orchestrator: Any) -> None:
        self.orch = orchestrator
        self._task: asyncio.Task | None = None

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run(), name=HEARTBEAT_TASK_NAME)
        self.orch.kv_store["body:heartbeat:pacemaker_started"] = True
        return self._task

    async def stop(self) -> None:
        task = self._task
        self._task = None
        self.orch.kv_store["body:heartbeat:pacemaker_started"] = False
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        tick = 0
        last_mono = loop.time()
        next_deadline = last_mono + HEARTBEAT_INTERVAL_S
        missed_total = int(self.orch.kv_store.get("body:heartbeat:missed_total", 0) or 0)
        hz_ema = HEARTBEAT_HZ
        jitter_ema = 0.0

        try:
            while True:
                await asyncio.sleep(max(0.0, next_deadline - loop.time()))
                now_mono = loop.time()
                epoch_s = time.time()
                delta_s = max(0.0, now_mono - last_mono)
                drift_s = now_mono - next_deadline

                # Estimate opportunities skipped by a host stall. They are
                # telemetry only and are never replayed onto the body bus.
                missed = max(0, int(delta_s / HEARTBEAT_INTERVAL_S) - 1)
                missed_total += missed

                tick += 1
                packet = heartbeat_payload(
                    tick=tick,
                    epoch_s=epoch_s,
                    monotonic_s=now_mono,
                    delta_s=delta_s,
                    drift_s=drift_s,
                    missed_estimate=missed,
                )

                actual_hz = (1.0 / delta_s) if delta_s > 1e-9 else HEARTBEAT_HZ
                hz_ema = (HEARTBEAT_STATS_KEEP * hz_ema) + (HEARTBEAT_STATS_NEW * actual_hz)
                jitter_ms = abs(delta_s - HEARTBEAT_INTERVAL_S) * 1000.0
                jitter_ema = (HEARTBEAT_STATS_KEEP * jitter_ema) + (HEARTBEAT_STATS_NEW * jitter_ms)
                stats = {
                    "schema": HEARTBEAT_STATS_SCHEMA,
                    "tick": tick,
                    "nominal_hz": HEARTBEAT_HZ,
                    "nominal_interval_s": HEARTBEAT_INTERVAL_S,
                    "actual_hz_ema": round(hz_ema, 3),
                    "jitter_ms_ema": round(jitter_ema, 3),
                    "drift_ms": round(drift_s * 1000.0, 3),
                    "delta_s": round(delta_s, 6),
                    "missed_last": missed,
                    "missed_total": missed_total,
                    "last_epoch_s": epoch_s,
                    "last_monotonic_s": now_mono,
                    "alive": True,
                }
                self.orch.kv_store["body:heartbeat:last"] = packet
                self.orch.kv_store["body:heartbeat:stats"] = stats
                self.orch.kv_store["body:heartbeat:missed_total"] = missed_total

                await self.orch.push_body_event(
                    PRIMARY_HEARTBEAT_TOPIC,
                    packet,
                    meta=heartbeat_meta(),
                    source="body.heartbeat_pacemaker",
                )

                last_mono = now_mono
                next_deadline += HEARTBEAT_INTERVAL_S
                if next_deadline <= now_mono:
                    # Advance to one future deadline; never emit missed pulses.
                    next_deadline = now_mono + HEARTBEAT_INTERVAL_S
        finally:
            previous = self.orch.kv_store.get("body:heartbeat:stats", {})
            stats = dict(previous) if isinstance(previous, dict) else {}
            stats["alive"] = False
            stats["last_epoch_s"] = time.time()
            self.orch.kv_store["body:heartbeat:stats"] = stats
