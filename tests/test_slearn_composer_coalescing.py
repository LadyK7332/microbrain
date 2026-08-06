from __future__ import annotations

import asyncio
import threading

from microbrain.memory.mem_cell_composer import MemCellComposer
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sidecars.memory_composer_sidecar import MemoryComposerSidecar
from microbrain.sidecars.slearn_workbench import (
    SLEARN_COMPOSER_FLUSH_BATCHES,
    SLEARN_MAX_INFLIGHT_BATCHES,
)


def test_composer_can_defer_learned_without_blocking_other_tiers(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=True, writer_id="coalesce-test")
    store.stage_cells([{"id": "learned:1", "kind": "test"}], tier="learned")
    store.stage_cells([{"id": "now:1", "kind": "test"}], tier="now")

    composer = MemCellComposer(tmp_path)
    status = composer.compose_once(tiers=["now", "short", "long"])

    assert "learned" not in status["tiers_requested"]
    assert (tmp_path / "mem_cell" / "now" / "now.jsonl").exists()
    assert list((tmp_path / "mem_cell" / "_pending" / "learned").glob("*.jsonl"))
    assert not (tmp_path / "mem_cell" / "learned" / "learned.jsonl").exists()

    composer.compose_once(tiers=["learned"])
    assert (tmp_path / "mem_cell" / "learned" / "learned.jsonl").exists()
    assert not list((tmp_path / "mem_cell" / "_pending" / "learned").glob("*.jsonl"))


def test_bucket_scheduler_buffers_learned_until_threshold_or_eof(tmp_path):
    orch = Orchestrator()
    orch.kv_store.update({
        "slearn:active_file": "wordnet.slearn",
        "slearn:mode": "bucket",
        "slearn:eof": False,
        "slearn:outstanding_batches": 8,
        "slearn:composer_flush_batches": SLEARN_COMPOSER_FLUSH_BATCHES,
    })
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    tiers = sidecar._tiers_for_cycle()
    assert "learned" not in tiers
    assert orch.kv_store["mem_cell:composer:learned_deferred"] is True

    orch.kv_store["slearn:outstanding_batches"] = SLEARN_COMPOSER_FLUSH_BATCHES
    assert "learned" in sidecar._tiers_for_cycle()

    orch.kv_store["slearn:outstanding_batches"] = 3
    orch.kv_store["slearn:eof"] = True
    assert "learned" in sidecar._tiers_for_cycle()


def test_bucket_backpressure_allows_one_50k_file_to_stage_before_final_commit():
    # Current WordNet chunks are ~50k lines at 1k lines per bucket.  Keeping the
    # inflight ceiling above that lets the file reach EOF before the learned
    # shard is rewritten, reducing write amplification to one final bulk commit.
    assert SLEARN_MAX_INFLIGHT_BATCHES >= 50
    assert SLEARN_COMPOSER_FLUSH_BATCHES >= 50


def test_composer_health_snapshot_exposes_queue_processing_and_long_cycle(tmp_path):
    orch = Orchestrator()
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    pending = tmp_path / "mem_cell" / "_pending" / "learned"
    processing = tmp_path / "mem_cell" / "_processing" / "learned"
    pending.mkdir(parents=True, exist_ok=True)
    processing.mkdir(parents=True, exist_ok=True)
    (pending / "one.jsonl").write_text("{}\n", encoding="utf-8")
    (processing / "two.jsonl.123.processing").write_text("{}\n", encoding="utf-8")

    class _LiveTask:
        @staticmethod
        def done() -> bool:
            return False

    sidecar._task = _LiveTask()  # type: ignore[assignment]
    orch.kv_store.update({
        "mem_cell:composer:started": True,
        "mem_cell:composer:busy": True,
        "mem_cell:composer:cycle_index": 12,
        "mem_cell:composer:cycle_started_ts": 100.0,
        "mem_cell:composer:last_success_ts": 90.0,
        "mem_cell:composer:active_tiers": ["learned"],
    })

    health = sidecar._health_snapshot(now=200.0)
    assert health["state"] == "busy_long"
    assert health["task_alive"] is True
    assert health["busy_age_s"] == 100.0
    assert health["pending"]["learned"] == 1
    assert health["processing"]["learned"] == 1
    assert health["pending_total"] == 1
    assert health["processing_total"] == 1


def test_composer_health_snapshot_exposes_real_exception_text(tmp_path):
    orch = Orchestrator()
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    class _LiveTask:
        @staticmethod
        def done() -> bool:
            return False

    sidecar._task = _LiveTask()  # type: ignore[assignment]
    orch.kv_store.update({
        "mem_cell:composer:started": True,
        "mem_cell:composer:last_success_ts": 10.0,
        "mem_cell:composer:last_error_ts": 20.0,
        "mem_cell:composer:last_error": "RuntimeError('boom')",
        "mem_cell:composer:last_error_type": "RuntimeError",
        "mem_cell:composer:consecutive_errors": 1,
    })

    health = sidecar._health_snapshot(now=21.0)
    assert health["state"] == "error"
    assert health["last_error_type"] == "RuntimeError"
    assert "boom" in health["last_error"]


def test_composer_start_does_not_wait_for_network_queue_scan(tmp_path):
    async def scenario() -> None:
        orch = Orchestrator()
        sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)
        release_scan = threading.Event()

        def slow_storage_probe():
            release_scan.wait(timeout=0.5)
            return {
                "pending": {tier: 0 for tier in ("now", "short", "long", "learned")},
                "processing": {tier: 0 for tier in ("now", "short", "long", "learned")},
                "lock_exists": False,
                "lock_age_s": 0.0,
            }

        sidecar._raw_storage_health = slow_storage_probe  # type: ignore[method-assign]
        await asyncio.wait_for(sidecar.start(), timeout=0.1)
        assert orch.kv_store["mem_cell:composer:started"] is True
        assert orch.kv_store["mem_cell:composer:health"]["pending_total"] == 0

        # Let the detached probe finish before closing the temporary directory.
        release_scan.set()
        await asyncio.sleep(0.02)
        await sidecar.stop()

    asyncio.run(scenario())


def test_cached_health_snapshot_never_touches_storage(tmp_path):
    orch = Orchestrator()
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    def fail_if_called():
        raise AssertionError("filesystem health probe ran on the event-loop path")

    sidecar._raw_storage_health = fail_if_called  # type: ignore[method-assign]
    health = sidecar._health_snapshot(now=10.0, scan_queues=False)
    assert health["pending_total"] == 0
    assert health["processing_total"] == 0
    assert health["lock_exists"] is False


def test_health_marks_a_single_long_queue_scan_without_starting_more(tmp_path):
    orch = Orchestrator()
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    class _LiveTask:
        @staticmethod
        def done() -> bool:
            return False

    sidecar._task = _LiveTask()  # type: ignore[assignment]
    sidecar._queue_scan_task = _LiveTask()  # type: ignore[assignment]
    sidecar._queue_scan_started_ts = 100.0
    orch.kv_store["mem_cell:composer:started"] = True

    health = sidecar._health_snapshot(now=111.0, scan_queues=False)
    assert health["task_alive"] is True
    assert health["queue_scan_running"] is True
    assert health["queue_scan_stalled"] is True
    assert health["queue_scan_age_s"] == 11.0


def test_bucket_waiting_commit_targets_learned_only_even_before_eof_flag(tmp_path):
    orch = Orchestrator()
    orch.kv_store.update({
        "slearn:active_file": "wordnet.slearn",
        "slearn:mode": "bucket",
        "slearn:eof": False,
        "slearn:phase": "waiting_commit",
        "slearn:status": "waiting_commit",
        "slearn:outstanding_batches": 51,
        "slearn:composer_flush_batches": 64,
    })
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)

    tiers = sidecar._tiers_for_cycle()

    assert tiers == ["learned"]
    assert orch.kv_store["mem_cell:composer:learned_flush_due"] is True
    assert orch.kv_store["mem_cell:composer:target_tiers_reason"] == "slearn_waiting_commit"


def test_slearn_learned_flush_health_scan_ignores_now_tier(tmp_path):
    orch = Orchestrator()
    orch.kv_store.update({
        "slearn:active_file": "wordnet.slearn",
        "slearn:mode": "bucket",
        "slearn:phase": "waiting_commit",
        "slearn:status": "waiting_commit",
        "slearn:outstanding_batches": 51,
    })
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)
    assert sidecar._tiers_for_cycle() == ["learned"]

    pending_now = tmp_path / "mem_cell" / "_pending" / "now"
    pending_learned = tmp_path / "mem_cell" / "_pending" / "learned"
    pending_now.mkdir(parents=True, exist_ok=True)
    pending_learned.mkdir(parents=True, exist_ok=True)
    (pending_now / "wrong.jsonl").write_text("{}\n", encoding="utf-8")
    (pending_learned / "right.jsonl").write_text("{}\n", encoding="utf-8")

    probe = sidecar._raw_storage_health()

    assert probe["scan_tiers"] == ["learned"]
    assert probe["pending"]["learned"] == 1
    assert probe["pending"]["now"] == 0


def test_receipt_focused_compose_ignores_unrelated_learned_backlog(tmp_path):
    stale = MemCellStore(tmp_path, composer_enabled=True, writer_id="old-slearn-job")
    for idx in range(7):
        stale.stage_cells([{"id": f"old:{idx}", "kind": "stale"}], tier="learned")

    current = MemCellStore(tmp_path, composer_enabled=True, writer_id="slearn-sidecar-job-current")
    current.stage_cells([
        {"id": "current:1", "kind": "slearn"},
        {"id": "current:2", "kind": "slearn"},
    ], tier="learned")
    receipts = current.take_staged_paths("learned")
    assert len(receipts) == 1

    composer = MemCellComposer(tmp_path)
    status = composer.compose_receipts(receipts, tier="learned")

    assert status["receipt_focused"] is True
    assert status["rows_applied"] == 2
    assert status["tiers"]["learned"]["pending_remaining"] == 0

    direct = MemCellStore(tmp_path, composer_enabled=False)
    ids = {row.get("id") for row in direct._read_shard("learned")}
    assert ids == {"current:1", "current:2"}
    assert list((tmp_path / "mem_cell" / "_pending" / "learned").glob("*.jsonl"))
    assert not any("current" in path.read_text(encoding="utf-8") for path in (tmp_path / "mem_cell" / "_pending" / "learned").glob("*.jsonl"))


def test_slearn_receipt_health_counts_exact_receipts_not_backlog(tmp_path):
    stale = MemCellStore(tmp_path, composer_enabled=True, writer_id="old-slearn-job")
    for idx in range(5):
        stale.stage_cells([{"id": f"old:{idx}", "kind": "stale"}], tier="learned")

    current = MemCellStore(tmp_path, composer_enabled=True, writer_id="slearn-sidecar-job-current")
    current.stage_cells([{"id": "current:1", "kind": "slearn"}], tier="learned")
    receipts = current.take_staged_paths("learned")

    orch = Orchestrator()
    orch.kv_store.update({
        "slearn:active_file": "wordnet.slearn",
        "slearn:mode": "bucket",
        "slearn:phase": "waiting_commit",
        "slearn:status": "waiting_commit",
        "slearn:outstanding_batches": len(receipts),
        "slearn:receipt_paths": receipts,
    })
    sidecar = MemoryComposerSidecar(orch, memdir=tmp_path)
    assert sidecar._tiers_for_cycle() == ["learned"]

    probe = sidecar._raw_storage_health()

    assert probe["receipt_focused"] is True
    assert probe["receipts_observed"] == len(receipts)
    assert probe["pending"]["learned"] == len(receipts)
    assert probe["processing"]["learned"] == 0
