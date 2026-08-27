from __future__ import annotations

from pathlib import Path

from microbrain.evidence.object_modality_scaffold import OBJECT_MODALITY_SCAFFOLD_FIELD
from microbrain.memory.mem_cell_store import MemCellStore


def test_memcell_store_upsert_scaffolds_durable_object(tmp_path: Path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    out = store.upsert_cell(
        {
            "id": "object:integration_soft_fabric",
            "kind": "object.memory_frame",
            "tier": "short",
            "encounter_count": 5,
            "modalities": {"touch": {"summary": "soft"}},
            "text": "soft fabric object",
        },
        tier="short",
        touch=False,
        flush=True,
    )

    assert OBJECT_MODALITY_SCAFFOLD_FIELD in out
    scaffold = out[OBJECT_MODALITY_SCAFFOLD_FIELD]
    assert "touch" in scaffold["modalities"]
    touch = scaffold["modalities"]["touch"]
    assert (tmp_path / touch["ledger_ref"]).exists()
    assert (tmp_path / touch["index_ref"]).exists()

    found = store.find_cell("object:integration_soft_fabric", tier_hint="short")
    assert found is not None
    assert OBJECT_MODALITY_SCAFFOLD_FIELD in found


def test_memcell_store_does_not_scaffold_token(tmp_path: Path):
    store = MemCellStore(tmp_path, composer_enabled=False)
    out = store.upsert_cell(
        {"id": "token:integration_soft", "kind": "token", "tier": "long", "text": "soft"},
        tier="long",
        touch=False,
        flush=True,
    )
    assert OBJECT_MODALITY_SCAFFOLD_FIELD not in out
