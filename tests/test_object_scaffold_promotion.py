from __future__ import annotations

import json
from pathlib import Path

from microbrain.evidence.object_modality_scaffold import OBJECT_MODALITY_SCAFFOLD_FIELD
from microbrain.memory.object_scaffold_promotion import (
    extract_modalities_for_object,
    looks_like_object_memcell,
    maybe_attach_object_modality_scaffold,
    should_attach_object_modality_scaffold,
)


def test_language_token_cell_is_not_scaffolded(tmp_path: Path):
    cell = {"id": "token:soft", "kind": "token", "tier": "long", "text": "soft"}
    out = maybe_attach_object_modality_scaffold(cell, base_dir=tmp_path, tier="long")
    assert OBJECT_MODALITY_SCAFFOLD_FIELD not in out
    assert not looks_like_object_memcell(cell)


def test_durable_object_gets_modality_scaffold_and_files(tmp_path: Path):
    cell = {
        "id": "object:soft_fabric_042",
        "kind": "object.memory_frame",
        "tier": "short",
        "encounter_count": 4,
        "modalities": {"touch": {"summary": "soft"}, "visual": {"summary": "fuzzy"}},
        "text": "soft fuzzy object",
    }
    out = maybe_attach_object_modality_scaffold(cell, base_dir=tmp_path, tier="short", source="unit_test")

    scaffold = out[OBJECT_MODALITY_SCAFFOLD_FIELD]
    assert scaffold["schema"] == "object.modality_scaffold.v1"
    assert set(scaffold["modalities"].keys()) >= {"touch", "vision", "language"}
    assert out["meta"]["evidence_scaffolded"] is True
    assert out["meta"]["evidence_scaffold_tier"] == "short"

    touch = scaffold["modalities"]["touch"]
    assert "next_ref_number" not in touch
    assert touch["next_ref_source"] == "index_file_not_memcell"
    assert (tmp_path / touch["ledger_ref"]).exists()
    assert (tmp_path / touch["index_ref"]).exists()
    idx = json.loads((tmp_path / touch["index_ref"]).read_text(encoding="utf-8"))
    assert idx["next_ref_number"] == 1


def test_hot_object_is_not_scaffolded_until_used_enough(tmp_path: Path):
    cell = {"id": "object:new", "kind": "object", "tier": "hot", "encounter_count": 20}
    assert not should_attach_object_modality_scaffold(cell, tier="hot")
    out = maybe_attach_object_modality_scaffold(cell, base_dir=tmp_path, tier="hot")
    assert OBJECT_MODALITY_SCAFFOLD_FIELD not in out


def test_now_object_scaffolds_after_enough_encounters(tmp_path: Path):
    cell = {"id": "object:known", "kind": "object", "tier": "now", "encounter_count": 3}
    out = maybe_attach_object_modality_scaffold(cell, base_dir=tmp_path, tier="now")
    assert OBJECT_MODALITY_SCAFFOLD_FIELD in out


def test_existing_scaffold_is_left_alone(tmp_path: Path):
    existing = {
        "schema": "object.modality_scaffold.v1",
        "object_id": "object:x",
        "modalities": {"touch": {"ledger_ref": "old", "index_ref": "old"}},
    }
    cell = {"id": "object:x", "kind": "object", "tier": "long", OBJECT_MODALITY_SCAFFOLD_FIELD: existing}
    out = maybe_attach_object_modality_scaffold(cell, base_dir=tmp_path, tier="long")
    assert out[OBJECT_MODALITY_SCAFFOLD_FIELD] == existing


def test_modalities_are_extracted_from_legacy_sense_fields():
    cell = {
        "id": "person:unknown_1",
        "kind": "person",
        "senses_present": {"vision": True, "audio": False, "touch": True},
        "sense_tags": {"audio": {"labels": ["rough_voice"]}},
        "classifiers": ["sensor.camera_0", "visual_proto_object"],
        "text": "unknown person",
    }
    mods = extract_modalities_for_object(cell)
    assert "vision" in mods
    assert "touch" in mods
    assert "audio" in mods
    assert "language" in mods
