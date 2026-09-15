import json
import os

from microbrain.memory.mem_cell_store import MemCellStore


def test_search_text_cells_unloads_tier_rows_after_lookup(tmp_path):
    os.environ["MB_ENABLE_MEM_CELL_V1"] = "1"
    store = MemCellStore(tmp_path, composer_enabled=False)
    path = tmp_path / "mem_cell" / "long" / "long.jsonl"
    row = {
        "schema": "mem_cell.v1",
        "id": "cell:apple",
        "kind": "object_descriptor",
        "tier": "long",
        "anchor": {"kind": "object/apple", "ref": "apple red fruit"},
        "refs": [],
        "ts": 1.0,
        "last_seen": 1.0,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    hits = store.search_text_cells("apple", tiers=("long",), limit=3)
    summary = store.memory_load_summary()

    assert hits and hits[0]["cell_id"] == "cell:apple"
    assert summary["loaded_row_total"] == 0
    assert "long" in summary["unloaded_tiers"]


def test_find_cells_by_kind_unloads_tier_rows_after_lookup(tmp_path):
    os.environ["MB_ENABLE_MEM_CELL_V1"] = "1"
    store = MemCellStore(tmp_path, composer_enabled=False)
    path = tmp_path / "mem_cell" / "learned" / "learned.jsonl"
    row = {
        "schema": "mem_cell.v1",
        "id": "cell:pattern",
        "kind": "thought_template",
        "tier": "learned",
        "anchor": {"kind": "thought/template", "ref": "if question then answer"},
        "refs": [],
        "usage_count": 2,
        "ts": 1.0,
        "last_seen": 1.0,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    hits = store.find_cells_by_kind("thought_template", tiers=("learned",), limit=3)
    summary = store.memory_load_summary()

    assert hits and hits[0]["id"] == "cell:pattern"
    assert summary["loaded_row_total"] == 0
    assert "learned" in summary["unloaded_tiers"]


def test_mem_cell_v1_is_disabled_by_default(tmp_path):
    os.environ.pop("MB_ENABLE_MEM_CELL_V1", None)
    os.environ.pop("MB_MEMORY_V1_ENABLED", None)

    store = MemCellStore(tmp_path, composer_enabled=False)

    assert store.memory_load_summary()["disabled"] is True
    assert store.search_text_cells("apple") == []
    assert store.stage_cells([{"id": "cell:x"}], tier="now") == 0
    assert not (tmp_path / "mem_cell").exists()
