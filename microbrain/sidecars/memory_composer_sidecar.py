from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from microbrain.memory.mem_cell_composer import MemCellComposer
from microbrain.memory.mem_cell_store import TIERS
from microbrain.sidecars.slearn_workbench import SLEARN_COMPOSER_FLUSH_BATCHES

logger = logging.getLogger(__name__)

# Behavioral tuning
COMPOSER_HEALTH_PULSE_S = 1.0
COMPOSER_LONG_CYCLE_WARN_S = 60.0
COMPOSER_QUEUE_SCAN_WARN_S = 10.0


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
        self._health_task: Optional[asyncio.Task] = None
        self._queue_scan_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._cycle_index = 0
        self._queue_scan_started_ts = 0.0
        self._queue_scan_finished_ts = 0.0
        self._queue_scan_error = ""
        self._cached_pending = {tier: 0 for tier in TIERS}
        self._cached_processing = {tier: 0 for tier in TIERS}
        self._cached_lock_exists = False
        self._cached_lock_age_s = 0.0
        self._cached_scan_tiers = list(TIERS)
        self._cached_receipt_focused = False
        self._cached_receipts_observed = 0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run_loop(), name="memory_composer_sidecar")
        self._health_task = asyncio.create_task(self._health_loop(), name="memory_composer_health")
        self.orchestrator.kv_store["mem_cell:composer:started"] = True

        # Startup must never wait on a disk/network-directory walk.  Z:\memory
        # may contain a large backlog or a temporarily unavailable share.  Seed a
        # lightweight snapshot immediately; the health task refreshes queue counts
        # in one detached worker thread after the event loop is already alive.
        await self._publish_health(scan_queues=False)
        logger.info("Memory composer sidecar started.")

    async def stop(self) -> None:
        self._stopping = True
        for task in (self._health_task, self._task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._queue_scan_task is not None and not self._queue_scan_task.done():
            self._queue_scan_task.cancel()
        self.orchestrator.kv_store["mem_cell:composer:started"] = False
        await self._publish_health(scan_queues=False)

    def _tiers_for_cycle(self) -> list[str]:
        """Return tiers eligible for this composer pass.

        Bulk SLEARN uses disk-backed pending files as a coalescing buffer. While
        a bucket file is still being staged, defer the learned tier until either
        EOF or the configured flush threshold is reached. When a learned flush is
        due, target learned *only* so the commit path cannot stall on unrelated
        hot/now/short/long queue scans before it services the SLEARN receipts.
        """
        kv = self.orchestrator.kv_store
        active_file = str(kv.get("slearn:active_file", "") or "")
        mode = str(kv.get("slearn:mode", "") or "").lower()
        eof = bool(kv.get("slearn:eof", False))
        phase = str(kv.get("slearn:phase", kv.get("slearn:status", "")) or "").lower()
        status = str(kv.get("slearn:status", "") or "").lower()
        outstanding = int(kv.get("slearn:outstanding_batches", 0) or 0)
        flush_batches = max(1, int(kv.get("slearn:composer_flush_batches", SLEARN_COMPOSER_FLUSH_BATCHES) or SLEARN_COMPOSER_FLUSH_BATCHES))
        bucket_active = bool(active_file and mode == "bucket")
        waiting_for_commit = phase in {"waiting_commit", "waiting_composer"} or status in {"waiting_commit", "waiting_composer"}
        learned_flush_due = bool(active_file and outstanding > 0 and (waiting_for_commit or eof or (bucket_active and outstanding >= flush_batches)))
        defer_learned = bool(bucket_active and outstanding > 0 and not learned_flush_due)

        kv["mem_cell:composer:learned_deferred"] = defer_learned
        kv["mem_cell:composer:learned_deferred_batches"] = outstanding if defer_learned else 0
        kv["mem_cell:composer:learned_flush_due"] = learned_flush_due
        kv["mem_cell:composer:learned_flush_batches"] = flush_batches
        kv["mem_cell:composer:target_tiers_reason"] = (
            "slearn_waiting_commit" if learned_flush_due and waiting_for_commit else
            "slearn_flush_threshold" if learned_flush_due else
            "slearn_deferred" if defer_learned else
            "normal"
        )

        if learned_flush_due:
            # A SLEARN commit should never be blocked by an unrelated tier.  The
            # regular composer loop will return to the other tiers after the
            # learned receipts have been acknowledged.
            return ["learned"]
        if defer_learned:
            return [tier for tier in TIERS if tier != "learned"]
        return list(TIERS)

    def _slearn_receipts_for_cycle(self) -> list[str]:
        """Return exact SLEARN composer receipts for a focused learned commit."""
        kv = self.orchestrator.kv_store
        if not bool(kv.get("mem_cell:composer:learned_flush_due", False)):
            return []
        raw = kv.get("slearn:receipt_paths", [])
        out: list[str] = []
        seen: set[str] = set()
        for item in list(raw or []):
            receipt = str(item or "").strip()
            if not receipt or receipt in seen:
                continue
            seen.add(receipt)
            out.append(receipt)
        return out

    def _raw_receipt_counts(self, receipts: list[str], *, tier: str = "learned") -> tuple[dict[str, int], dict[str, int]]:
        """Count only exact SLEARN receipts, not the whole learned backlog."""
        pending: dict[str, int] = {t: 0 for t in TIERS}
        processing: dict[str, int] = {t: 0 for t in TIERS}
        for path in self.composer.receipt_paths(receipts, tier=tier):
            if str(path.name).endswith(".processing"):
                processing[tier] += 1
            else:
                pending[tier] += 1
        return pending, processing

    async def _run_loop(self) -> None:
        while not self._stopping:
            kv = self.orchestrator.kv_store
            self._cycle_index += 1
            cycle_started = time.time()
            kv["mem_cell:composer:cycle_index"] = self._cycle_index
            kv["mem_cell:composer:cycle_started_ts"] = cycle_started
            try:
                tiers = self._tiers_for_cycle()
                kv["mem_cell:composer:busy"] = True
                kv["mem_cell:composer:active_tiers"] = list(tiers)
                await self._publish_health(scan_queues=False)
                receipts = self._slearn_receipts_for_cycle()
                kv["mem_cell:composer:target_receipts_count"] = len(receipts)
                if receipts and tiers == ["learned"]:
                    status = await asyncio.to_thread(self.composer.compose_receipts, receipts, tier="learned")
                else:
                    status = await asyncio.to_thread(self.composer.compose_once, tiers=tiers)
                kv["mem_cell:composer:last_success_ts"] = time.time()
                kv["mem_cell:composer:consecutive_errors"] = 0
                kv["mem_cell:composer:last_error"] = ""
                kv["mem_cell:composer:last_error_type"] = ""
                await self._publish_status(status)
            except TimeoutError as exc:
                # A live lock is diagnostic information, not a silent no-op.
                kv["mem_cell:composer:last_lock_timeout_ts"] = time.time()
                kv["mem_cell:composer:last_lock_timeout"] = str(exc)
                kv["mem_cell:composer:last_error"] = str(exc)
                kv["mem_cell:composer:last_error_type"] = "TimeoutError"
                kv["mem_cell:composer:last_error_ts"] = time.time()
                kv["mem_cell:composer:consecutive_errors"] = int(kv.get("mem_cell:composer:consecutive_errors", 0) or 0) + 1
            except Exception as exc:
                logger.exception("Memory composer cycle failed")
                kv["mem_cell:composer:last_error"] = repr(exc)
                kv["mem_cell:composer:last_error_type"] = type(exc).__name__
                kv["mem_cell:composer:last_error_ts"] = time.time()
                kv["mem_cell:composer:consecutive_errors"] = int(kv.get("mem_cell:composer:consecutive_errors", 0) or 0) + 1
            finally:
                finished = time.time()
                kv["mem_cell:composer:busy"] = False
                kv["mem_cell:composer:active_tiers"] = []
                kv["mem_cell:composer:cycle_finished_ts"] = finished
                kv["mem_cell:composer:last_cycle_elapsed_s"] = max(0.0, finished - cycle_started)
                await self._publish_health(scan_queues=False)
            await asyncio.sleep(self.interval_s)

    async def _health_loop(self) -> None:
        while not self._stopping:
            self._kick_or_harvest_queue_scan()
            await self._publish_health(scan_queues=False)
            await asyncio.sleep(COMPOSER_HEALTH_PULSE_S)

    def _kick_or_harvest_queue_scan(self) -> None:
        """Maintain at most one non-blocking queue-directory scan.

        The scan can be slow on a network-backed memdir.  It therefore runs in
        a worker thread and is never awaited by startup, the composer cycle, or
        the dashboard heartbeat.  A hung scan remains visible as telemetry
        instead of freezing the whole mind or spawning an unbounded thread pile.
        """
        task = self._queue_scan_task
        if task is not None:
            if not task.done():
                return
            try:
                probe = task.result()
                self._cached_pending = dict(probe.get("pending", {}))
                self._cached_processing = dict(probe.get("processing", {}))
                self._cached_lock_exists = bool(probe.get("lock_exists", False))
                self._cached_lock_age_s = float(probe.get("lock_age_s", 0.0) or 0.0)
                self._cached_scan_tiers = [str(t) for t in list(probe.get("scan_tiers", []) or []) if str(t) in TIERS]
                self._cached_receipt_focused = bool(probe.get("receipt_focused", False))
                self._cached_receipts_observed = int(probe.get("receipts_observed", 0) or 0)
                self._queue_scan_error = ""
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._queue_scan_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue_scan_finished_ts = time.time()
                self._queue_scan_task = None

        if self._stopping or self._queue_scan_task is not None:
            return
        self._queue_scan_started_ts = time.time()
        self._queue_scan_task = asyncio.create_task(
            asyncio.to_thread(self._raw_storage_health),
            name="memory_composer_queue_scan",
        )

    def _health_scan_tiers(self) -> list[str]:
        """Return the queue tiers the observer may safely scan right now.

        When SLEARN is waiting for learned receipts, scanning hot/now/short/long
        is unrelated to the user's blocker and can itself hang on a busy
        network-backed memdir.  Keep the observer focused on the target tier.
        """
        kv = self.orchestrator.kv_store
        active_tiers = [str(t) for t in list(kv.get("mem_cell:composer:active_tiers", []) or []) if str(t) in TIERS]
        if active_tiers:
            return list(dict.fromkeys(active_tiers))
        if bool(kv.get("mem_cell:composer:learned_flush_due", False)):
            return ["learned"]
        return list(TIERS)

    @staticmethod
    def _count_matching_files(path: Path, suffix: str) -> int:
        try:
            if not path.exists():
                return 0
            count = 0
            with __import__("os").scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file() and entry.name.endswith(suffix) and not entry.name.startswith("."):
                            count += 1
                    except OSError:
                        continue
            return count
        except OSError:
            return 0

    def _raw_queue_counts(self, tiers: list[str] | None = None) -> tuple[dict[str, int], dict[str, int]]:
        pending: dict[str, int] = {tier: 0 for tier in TIERS}
        processing: dict[str, int] = {tier: 0 for tier in TIERS}
        for tier in list(tiers or TIERS):
            if tier not in TIERS:
                continue
            pdir = self.composer.pending_root / tier
            xdir = self.composer.processing_root / tier
            pending[tier] = self._count_matching_files(pdir, ".jsonl")
            processing[tier] = self._count_matching_files(xdir, ".processing")
        return pending, processing

    def _raw_storage_health(self) -> dict[str, Any]:
        """Collect filesystem-backed health in a worker thread only."""
        tiers = self._health_scan_tiers()
        receipts = self._slearn_receipts_for_cycle()
        receipt_focused = bool(receipts and tiers == ["learned"])
        if receipt_focused:
            pending, processing = self._raw_receipt_counts(receipts, tier="learned")
        else:
            pending, processing = self._raw_queue_counts(tiers)
        lock_exists = self.composer.lock_path.exists()
        lock_age = 0.0
        if lock_exists:
            try:
                lock_age = max(0.0, time.time() - self.composer.lock_path.stat().st_mtime)
            except OSError:
                lock_age = 0.0
        return {
            "pending": pending,
            "processing": processing,
            "scan_tiers": tiers,
            "lock_exists": lock_exists,
            "lock_age_s": lock_age,
            "receipt_focused": receipt_focused,
            "receipts_observed": len(receipts) if receipt_focused else 0,
        }

    def _health_snapshot(
        self,
        *,
        now: float | None = None,
        scan_queues: bool = True,
    ) -> dict[str, Any]:
        now = float(now if now is not None else time.time())
        kv = self.orchestrator.kv_store
        if scan_queues:
            probe = self._raw_storage_health()
            pending = dict(probe.get("pending", {}))
            processing = dict(probe.get("processing", {}))
            lock_exists = bool(probe.get("lock_exists", False))
            lock_age = float(probe.get("lock_age_s", 0.0) or 0.0)
        else:
            pending = dict(self._cached_pending)
            processing = dict(self._cached_processing)
            lock_exists = self._cached_lock_exists
            lock_age = self._cached_lock_age_s

        queue_scan_running = bool(self._queue_scan_task is not None and not self._queue_scan_task.done())
        queue_scan_age = (
            max(0.0, now - self._queue_scan_started_ts)
            if queue_scan_running and self._queue_scan_started_ts
            else 0.0
        )
        busy = bool(kv.get("mem_cell:composer:busy", False))
        started_ts = float(kv.get("mem_cell:composer:cycle_started_ts", 0.0) or 0.0)
        busy_age = max(0.0, now - started_ts) if busy and started_ts else 0.0
        compose_phase = self.composer.telemetry_snapshot(now=now)
        task_alive = bool(self._task is not None and not self._task.done())
        last_success = float(kv.get("mem_cell:composer:last_success_ts", 0.0) or 0.0)
        last_error_ts = float(kv.get("mem_cell:composer:last_error_ts", 0.0) or 0.0)
        errors = int(kv.get("mem_cell:composer:consecutive_errors", 0) or 0)
        if self._stopping:
            state = "stopping"
        elif bool(kv.get("mem_cell:composer:started", False)) and not task_alive:
            state = "worker_dead"
        elif errors > 0 and last_error_ts >= last_success:
            state = "error"
        elif busy_age >= COMPOSER_LONG_CYCLE_WARN_S:
            state = "busy_long"
        elif busy:
            state = "busy"
        elif bool(kv.get("mem_cell:composer:learned_deferred", False)):
            state = "deferred"
        else:
            state = "idle"

        return {
            "schema": "mem_cell.composer.health.v1",
            "ts": now,
            "state": state,
            "started": bool(kv.get("mem_cell:composer:started", False)),
            "task_alive": task_alive,
            "busy": busy,
            "busy_age_s": round(busy_age, 3),
            "cycle_index": int(kv.get("mem_cell:composer:cycle_index", self._cycle_index) or 0),
            "last_cycle_elapsed_s": round(float(kv.get("mem_cell:composer:last_cycle_elapsed_s", 0.0) or 0.0), 3),
            "last_success_ts": last_success,
            "last_error_ts": last_error_ts,
            "last_error": str(kv.get("mem_cell:composer:last_error", "") or ""),
            "last_error_type": str(kv.get("mem_cell:composer:last_error_type", "") or ""),
            "last_lock_timeout": str(kv.get("mem_cell:composer:last_lock_timeout", "") or ""),
            "last_lock_timeout_ts": float(kv.get("mem_cell:composer:last_lock_timeout_ts", 0.0) or 0.0),
            "consecutive_errors": errors,
            "active_tiers": list(kv.get("mem_cell:composer:active_tiers", []) or []),
            "compose_phase": compose_phase,
            "pending": pending,
            "processing": processing,
            "pending_total": sum(pending.values()),
            "processing_total": sum(processing.values()),
            "scan_tiers": list(self._cached_scan_tiers),
            "target_tiers_reason": str(kv.get("mem_cell:composer:target_tiers_reason", "") or ""),
            "receipt_focused": bool(self._cached_receipt_focused),
            "receipts_observed": int(self._cached_receipts_observed or 0),
            "target_receipts_count": int(kv.get("mem_cell:composer:target_receipts_count", 0) or 0),
            "queue_scan_running": queue_scan_running,
            "queue_scan_age_s": round(queue_scan_age, 3),
            "queue_scan_stalled": bool(queue_scan_age >= COMPOSER_QUEUE_SCAN_WARN_S),
            "queue_scan_started_ts": self._queue_scan_started_ts,
            "queue_scan_finished_ts": self._queue_scan_finished_ts,
            "queue_scan_error": self._queue_scan_error,
            "lock_exists": lock_exists,
            "lock_age_s": round(lock_age, 3),
            "learned_deferred": bool(kv.get("mem_cell:composer:learned_deferred", False)),
            "last_status": kv.get("mem_cell:composer:last_status", {}),
        }

    async def _publish_health(self, *, scan_queues: bool = False) -> None:
        try:
            self.orchestrator.kv_store["mem_cell:composer:health"] = self._health_snapshot(
                scan_queues=scan_queues
            )
        except Exception:
            logger.exception("Memory composer health snapshot failed")

    async def _publish_status(self, status: dict[str, Any]) -> None:
        try:
            kv = self.orchestrator.kv_store
            kv["mem_cell:composer:last_status"] = status
            kv["mem_cell:composer:last_seen_ts"] = time.time()
            # Preserve legacy telemetry without walking the network-backed
            # queue on the asyncio/UI thread.  The detached health probe refreshes
            # these cached counts independently.
            kv["mem_cell:composer:pending_count"] = dict(self._cached_pending)
        except Exception:
            pass
