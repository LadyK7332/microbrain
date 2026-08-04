from __future__ import annotations

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
