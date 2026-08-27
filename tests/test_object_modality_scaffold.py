from __future__ import annotations

import json
from pathlib import Path

from microbrain.evidence.object_modality_scaffold import (
    OBJECT_MODALITY_SCAFFOLD_SCHEMA,
    ObjectModalityScaffoldStore,
    attach_scaffold_to_memcell,
)


def test_ensure_scaffold_preseeds_default_modality_files(tmp_path: Path):
    store = ObjectModalityScaffoldStore(tmp_path)
    packet = store.ensure_scaffold("object:soft_fabric_042", source="unit_test")

    assert packet["schema"] == OBJECT_MODALITY_SCAFFOLD_SCHEMA
    assert set(packet["modalities"].keys()) >= {"vision", "audio", "touch", "language"}
    touch = packet["modalities"]["touch"]
    assert touch["next_ref_source"] == "index_file_not_memcell"
    assert (tmp_path / touch["ledger_ref"]).exists()
    assert (tmp_path / touch["index_ref"]).exists()

    idx = json.loads((tmp_path / touch["index_ref"]).read_text(encoding="utf-8"))
    assert idx["entry_count"] == 0
    assert idx["next_ref_number"] == 1


def test_reserve_ref_appends_to_ledger_and_updates_index(tmp_path: Path):
    store = ObjectModalityScaffoldStore(tmp_path)
    packet = store.ensure_scaffold("object:soft_fabric_042", modalities=["touch"])

    first = store.reserve_ref(
        packet,
        "touch",
        artifact_ref="evidence/touch/a.jsonl",
        fossil_ref="touch_fossil:a",
        summary="soft fuzzy contact",
        claims_supported=["texture.soft", "texture.fuzzy"],
        confidence=0.61,
        source="unit_test",
    )
    second = store.reserve_ref(packet, "touch", artifact_ref="evidence/touch/b.jsonl", summary="low slip")

    assert first["ref_id"].endswith(":000001")
    assert second["ref_id"].endswith(":000002")

    idx = store.read_index(packet, "touch")
    assert idx["entry_count"] == 2
    assert idx["next_ref_number"] == 3
    assert idx["last_ref_id"] == second["ref_id"]
    assert "texture.soft" in idx["claim_hints"]

    rows = list(store.iter_ledger_entries(packet, "touch"))
    assert [row["ref_id"] for row in rows] == [first["ref_id"], second["ref_id"]]


def test_attach_scaffold_to_memcell_keeps_packet_compact(tmp_path: Path):
    store = ObjectModalityScaffoldStore(tmp_path)
    packet = store.ensure_scaffold("person:unknown_123")
    cell = {"id": "person:unknown_123", "kind": "person", "text": "unknown person"}

    out = attach_scaffold_to_memcell(cell, packet)
    assert out["id"] == cell["id"]
    assert out["meta"]["evidence_scaffolded"] is True
    assert out["evidence_scaffold"]["schema"] == OBJECT_MODALITY_SCAFFOLD_SCHEMA
    assert "touch" in out["evidence_scaffold"]["modalities"]

    encoded = json.dumps(out["evidence_scaffold"], sort_keys=True)
    assert "soft fuzzy contact" not in encoded
    assert len(encoded) < 5000


def test_scaffold_does_not_store_mutable_next_counter_in_memcell_packet(tmp_path: Path):
    store = ObjectModalityScaffoldStore(tmp_path)
    packet = store.ensure_scaffold("object:counter_test", modalities=["vision"])
    store.reserve_ref(packet, "vision", artifact_ref="evidence/vision/a.json")

    # The scaffold packet is stable. The index owns next_ref_number.
    vision_packet = packet["modalities"]["vision"]
    assert "next_ref_number" not in vision_packet
    assert vision_packet["next_ref_source"] == "index_file_not_memcell"
    assert store.read_index(packet, "vision")["next_ref_number"] == 2


def test_custom_modalities_are_normalized_and_deduped(tmp_path: Path):
    store = ObjectModalityScaffoldStore(tmp_path)
    packet = store.ensure_scaffold("object:x", modalities=["Touch", "touch", "thermal sense", ""])
    assert list(packet["modalities"].keys()) == ["touch", "thermal_sense"]
