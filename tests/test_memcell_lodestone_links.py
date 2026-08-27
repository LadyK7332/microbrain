from __future__ import annotations

from pathlib import Path

from microbrain.memory.lodestone_links import (
    MEMCELL_LINK_LODESTONE_SCHEMA,
    MEMCELL_LINK_PACK_SCHEMA,
    RETRIEVAL_DIRECTED,
    RETRIEVAL_SCATTER,
    RETRIEVAL_WALK,
    MemCellLinkLodestoneStore,
    normalize_link_entry,
    select_lodestone_retrieval_mode,
)


def test_normalize_link_entry_preserves_reason_weight_and_ref() -> None:
    entry = normalize_link_entry(
        {"cell_id": "concept:soft", "kind": "attribute", "weight": 2.0, "confidence": 0.7, "reason": "touch summary"},
        bucket="links_explicit",
        source="unit_test",
    )

    assert entry["ref"] == "concept:soft"
    assert entry["link_type"] == "attribute"
    assert entry["weight"] == 1.0
    assert entry["confidence"] == 0.7
    assert entry["reason"] == "touch summary"
    assert entry["bucket"] == "links_explicit"


def test_pack_links_keeps_small_buckets_inline(tmp_path: Path) -> None:
    store = MemCellLinkLodestoneStore(tmp_path)
    pack = store.pack_links(
        cell_id="concept:fabric",
        bucket="links_explicit",
        links=["object:a", "object:b", "object:c"],
        max_inline_links=4,
    )

    assert pack["schema"] == MEMCELL_LINK_PACK_SCHEMA
    assert pack["count"] == 3
    assert len(pack["links"]) == 3
    assert pack["lodestone"] == {}
    assert not (tmp_path / "mem_cell_links" / "concept:fabric" / "links_explicit.jsonl").exists()


def test_pack_links_spills_dense_bucket_to_append_only_lodestone(tmp_path: Path) -> None:
    store = MemCellLinkLodestoneStore(tmp_path)
    links = [
        {"ref": f"cell:{idx}", "weight": idx / 10, "confidence": 0.4 + idx / 100, "reason": "hub growth"}
        for idx in range(9)
    ]

    pack = store.pack_links(
        cell_id="concept:clock_tick",
        bucket="links_explicit",
        links=links,
        max_inline_links=4,
        source="unit_test",
        reason="dense node overflow",
        timestamp=1787133192.1,
    )

    assert pack["schema"] == MEMCELL_LINK_PACK_SCHEMA
    assert pack["count"] == 9
    assert pack["links"] == []
    pointer = pack["lodestone"]
    assert pointer["schema"] == MEMCELL_LINK_LODESTONE_SCHEMA
    assert pointer["overflowed"] is True
    assert pointer["hub"] is True
    assert pointer["degree_estimate"] == 9
    assert pointer["ledger_ref"].endswith("links_explicit.jsonl")
    assert pointer["index_ref"].endswith("links_explicit.idx.json")
    assert pointer["query_weight_hint"] == "broad_traversal_not_specific_answer"

    ledger_path = tmp_path / pointer["ledger_ref"]
    index_path = tmp_path / pointer["index_ref"]
    assert ledger_path.exists()
    assert index_path.exists()
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 9
    index = store.read_index("concept:clock_tick", "links_explicit")
    assert index["schema"] == "memcell.link_ledger_index.v1"
    assert index["degree_estimate"] == 9
    assert index["retrieval_modes"] == ["directed", "walk", "scatter"]


def test_shape_link_field_returns_list_or_lodestone_pointer(tmp_path: Path) -> None:
    store = MemCellLinkLodestoneStore(tmp_path)

    inline = store.shape_link_field(
        cell_id="concept:small",
        bucket="links_lang",
        links=["noun:object", "attribute:soft"],
        max_inline_links=4,
    )
    assert isinstance(inline, list)
    assert len(inline) == 2

    lodestone = store.shape_link_field(
        cell_id="concept:large",
        bucket="links_lang",
        links=[f"usage:{idx}" for idx in range(6)],
        max_inline_links=3,
    )
    assert isinstance(lodestone, dict)
    assert lodestone["schema"] == MEMCELL_LINK_LODESTONE_SCHEMA
    assert lodestone["overflowed"] is True


def test_retrieve_links_supports_directed_walk_and_scatter_modes(tmp_path: Path) -> None:
    store = MemCellLinkLodestoneStore(tmp_path)
    links = [
        {"ref": "cell:soft_touch", "confidence": 0.9, "weight": 0.8, "reason": "touch fuzzy soft"},
        {"ref": "cell:green_vision", "confidence": 0.7, "weight": 0.3, "reason": "vision green"},
        {"ref": "cell:voice_rough", "confidence": 0.5, "weight": 0.5, "reason": "audio rough voice"},
        {"ref": "cell:old_low", "confidence": 0.1, "weight": 0.1, "reason": "low trust"},
    ]
    store.append_links(cell_id="person:demo", bucket="links_explicit", links=links, timestamp=1787133192.1)

    directed = store.retrieve_links(cell_id="person:demo", bucket="links_explicit", mode="directed", query="soft touch", limit=2)
    assert directed["schema"] == "memcell.link_retrieval.v1"
    assert directed["mode"] == RETRIEVAL_DIRECTED
    assert directed["results"][0]["ref"] == "cell:soft_touch"

    walk = store.retrieve_links(cell_id="person:demo", bucket="links_explicit", mode="walk", limit=2)
    assert walk["mode"] == RETRIEVAL_WALK
    assert walk["results"][0]["ref"] == "cell:soft_touch"

    scatter_a = store.retrieve_links(cell_id="person:demo", bucket="links_explicit", mode="scatter", limit=3, seed=42)
    scatter_b = store.retrieve_links(cell_id="person:demo", bucket="links_explicit", mode="scatter", limit=3, seed=42)
    assert scatter_a["mode"] == RETRIEVAL_SCATTER
    assert scatter_a["results"] == scatter_b["results"]


def test_select_lodestone_retrieval_mode_uses_existing_drive_and_hypothesis_state() -> None:
    assert select_lodestone_retrieval_mode({"direct_question": True}) == RETRIEVAL_DIRECTED
    assert select_lodestone_retrieval_mode({"hypothesis_response_demand": 0.81}) == RETRIEVAL_DIRECTED
    assert select_lodestone_retrieval_mode({"trainer_correction": True}) == RETRIEVAL_WALK
    assert select_lodestone_retrieval_mode({"contradiction": True}) == RETRIEVAL_WALK
    assert select_lodestone_retrieval_mode({"boredom": 0.8, "curiosity": 0.6, "hypothesis_uncertainty": 0.1}) == RETRIEVAL_SCATTER
