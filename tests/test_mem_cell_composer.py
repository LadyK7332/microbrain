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
