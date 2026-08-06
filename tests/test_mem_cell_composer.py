from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from microbrain.memory.mem_cell_composer import MemCellComposer
from microbrain.memory.mem_cell_store import MemCellStore


def test_mem_cell_writers_stage_pending_and_composer_writes_single_canonical_shard(tmp_path):
    store = MemCellStore(tmp_path)

    def write_cell(idx: int) -> None:
        store.upsert_cell(
            {
                "id": f"cell:{idx % 7}",
                "kind": "test",
                "anchor": {"text": f"row {idx}"},
                "meta": {"idx": idx},
            },
            tier="now",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_cell, range(64)))

    path = tmp_path / "mem_cell" / "now" / "now.jsonl"
    assert not path.exists()
    pending = list((tmp_path / "mem_cell" / "_pending" / "now").glob("*.jsonl"))
    assert pending

    composer = MemCellComposer(tmp_path)
    status = composer.compose_once()
    assert status["rows_applied"] >= 7
    assert path.exists()
    assert not list((tmp_path / "mem_cell" / "_pending" / "now").glob("*.tmp"))

    direct_store = MemCellStore(tmp_path, composer_enabled=False)
    rows = direct_store._read_shard("now")
    ids = {row.get("id") for row in rows}
    assert ids == {f"cell:{idx}" for idx in range(7)}


def test_direct_mem_cell_store_can_still_write_without_composer(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    store.upsert_cell({"id": "direct:1", "kind": "test"}, tier="now")
    path = tmp_path / "mem_cell" / "now" / "now.jsonl"
    assert path.exists()
    assert {row.get("id") for row in store._read_shard("now")} == {"direct:1"}


def test_composer_exposes_fine_grained_phase_telemetry(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=True, writer_id="phase-test")
    store.stage_cells([
        {"id": "phase:1", "kind": "test", "tier": "learned"},
        {"id": "phase:2", "kind": "test", "tier": "learned"},
    ], tier="learned")

    composer = MemCellComposer(tmp_path)
    initial = composer.telemetry_snapshot()
    assert initial["phase"] == "idle"

    status = composer.compose_once(tiers=["learned"])
    snap = composer.telemetry_snapshot()
    assert status["rows_applied"] == 2
    assert snap["phase"] == "idle"
    assert snap["files_processed"] >= 1
    assert snap["rows_applied"] == 2
    assert "phase_age_s" in snap



def test_composer_skips_global_lock_when_selected_tiers_have_no_pending_work(tmp_path):
    composer = MemCellComposer(tmp_path, lock_timeout_s=0.1)
    lock = tmp_path / "mem_cell" / "_composer.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pid=999999 owner_id=foreign ts=1\n", encoding="utf-8")

    status = composer.compose_once(tiers=["now", "short", "long"])

    assert status["skipped_lock_no_work"] is True
    assert status["files_processed"] == 0
    assert lock.exists(), "no-work preflight should not touch a foreign lock"


def test_composer_recovers_dead_owner_lock_before_processing_work(tmp_path):
    store = MemCellStore(tmp_path, composer_enabled=True, writer_id="dead-lock-test")
    store.stage_cells([{"id": "deadlock:1", "kind": "test"}], tier="learned")
    lock = tmp_path / "mem_cell" / "_composer.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pid=999999 owner_id=dead-test ts=1\n", encoding="utf-8")

    composer = MemCellComposer(tmp_path, lock_timeout_s=0.2, stale_lock_after_s=999999.0)
    status = composer.compose_once(tiers=["learned"])

    assert status["rows_applied"] == 1
    snap = composer.telemetry_snapshot()
    assert snap["phase"] == "idle"
    assert int(snap.get("lock_recoveries", 0) or 0) >= 1
