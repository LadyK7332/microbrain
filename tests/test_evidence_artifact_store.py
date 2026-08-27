from __future__ import annotations

from pathlib import Path

from microbrain.evidence.artifact_store import EvidenceArtifactStore
from microbrain.evidence.evidence_card import build_evidence_card, evidence_ref_card


def test_write_jsonl_artifact_returns_compact_card_without_raw_records(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    records = [
        {"t": idx * 0.01, "pressure": idx / 100.0, "sensor": "left_pad"}
        for idx in range(100)
    ]

    card = store.write_jsonl_artifact(
        modality="touch",
        records=records,
        prefix="touch_delta",
        summary="soft fuzzy contact, light compression, low slip",
        claims_supported=["texture.soft", "texture.fuzzy", "shape.compressible"],
        confidence=0.61,
        timestamp=1787133192.1,
        time_range=[1787133192.1, 1787133195.8],
        fossil_ref="touch_fossil:abc123",
        source="touch_artifact_neuron",
    )

    assert card["schema"] == "evidence.card.v1"
    assert card["modality"] == "touch"
    assert card["artifact_ref"].startswith("evidence/touch/2026-08-19/touch_delta_")
    assert card["sample_count"] == 100
    assert card["byte_count"] > 0
    assert card["checksum"].startswith("blake2b:")
    assert card["claims_supported"] == ["texture.soft", "texture.fuzzy", "shape.compressible"]
    assert card["confidence"] == 0.61
    assert card["fossil_ref"] == "touch_fossil:abc123"

    # The card is the handle, not the cursed proof pile.
    assert "pressure" not in repr(card)
    artifact_path = tmp_path / card["artifact_ref"]
    assert artifact_path.exists()
    assert "pressure" in artifact_path.read_text(encoding="utf-8")


def test_read_jsonl_artifact_round_trips_from_card_or_ref(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    records = [{"idx": 1}, {"idx": 2}, {"idx": 3}]
    card = store.write_jsonl_artifact(modality="touch", records=records, timestamp=1787133192.1)

    assert store.read_jsonl_artifact(card) == records
    assert store.read_jsonl_artifact(card["artifact_ref"], limit=2) == records[:2]


def test_write_json_artifact_uses_portable_refs_and_compact_ref(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    card = store.write_json_artifact(
        modality="vision",
        payload={"object": "button", "confidence": 0.7},
        prefix="vision_obj",
        summary="green button candidate",
        claims_supported=["kind.button", "color.green"],
        confidence=0.7,
        timestamp=1787133192.1,
    )

    assert not card["artifact_ref"].startswith(str(tmp_path))
    assert card["artifact_ref"].startswith("evidence/vision/2026-08-19/")
    assert store.read_json_artifact(card) == {"object": "button", "confidence": 0.7}

    compact = store.compact_ref(card)
    assert compact == evidence_ref_card(card)
    assert compact["schema"] == "evidence.ref.v1"
    assert compact["artifact_ref"] == card["artifact_ref"]
    assert compact["summary"] == "green button candidate"
    assert "byte_count" not in compact
    assert "checksum" not in compact


def test_build_evidence_card_clamps_and_cleans_values() -> None:
    card = build_evidence_card(
        modality="Touch Sensor!!",
        artifact_ref="/evidence/touch/example.jsonl",
        summary="x" * 500,
        claims_supported=["soft", "soft", "fuzzy"],
        confidence=9.0,
        time_range=[20.0, 10.0],
    )

    assert card["modality"] == "touch_sensor"
    assert card["artifact_ref"] == "evidence/touch/example.jsonl"
    assert card["confidence"] == 1.0
    assert card["claims_supported"] == ["soft", "fuzzy"]
    assert card["time_range"] == [10.0, 20.0]
    assert len(card["summary"]) <= 280


def test_missing_artifact_reads_are_safe(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    assert store.read_jsonl_artifact("evidence/touch/missing.jsonl") == []
    assert store.read_json_artifact("evidence/touch/missing.json") is None



def test_pack_refs_keeps_small_reference_sets_inline(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    refs = [f"evidence/touch/day/sample_{idx}.jsonl" for idx in range(3)]

    pack = store.pack_refs(modality="touch", refs=refs, max_inline_refs=4)

    assert pack["schema"] == "evidence.ref_pack.v1"
    assert pack["count"] == 3
    assert len(pack["refs"]) == 3
    assert pack["index_ref"] == ""
    assert not (tmp_path / "evidence" / "touch").exists()


def test_pack_refs_writes_index_when_reference_set_is_large(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    refs = [f"evidence/touch/day/sample_{idx}.jsonl" for idx in range(9)]

    pack = store.pack_refs(
        modality="touch",
        refs=refs,
        max_inline_refs=4,
        timestamp=1787133192.1,
        source="unit_test",
    )

    assert pack["schema"] == "evidence.ref_pack.v1"
    assert pack["count"] == 9
    assert pack["refs"] == []
    assert pack["index_ref"].startswith("evidence/touch/2026-08-19/ref_index_")
    assert pack["index_card"]["schema"] == "evidence.ref.v1"
    assert pack["index_card"]["claims_supported"] == ["evidence.ref_index"]

    index_payload = store.read_json_artifact(pack["index_ref"])
    assert index_payload["schema"] == "evidence.ref_index.v1"
    assert index_payload["count"] == 9
    assert [row["artifact_ref"] for row in index_payload["refs"]] == refs
    assert store.read_ref_index(pack["index_card"]) == index_payload["refs"]


def test_write_ref_index_deduplicates_and_normalizes_refs(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    card = store.write_ref_index(
        modality="vision",
        refs=[
            {"artifact_ref": "/evidence/vision/a.json", "summary": "A"},
            {"artifact_ref": "evidence/vision/a.json", "summary": "A"},
            {"ref": "evidence/vision/b.json", "kind": "snapshot"},
        ],
        timestamp=1787133192.1,
    )

    refs = store.read_ref_index(card)
    assert len(refs) == 2
    assert refs[0]["artifact_ref"] == "evidence/vision/a.json"
    assert refs[1] == {"artifact_ref": "evidence/vision/b.json", "kind": "snapshot"}


def test_pack_multimodal_refs_keeps_small_total_inline(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    pack = store.pack_multimodal_refs(
        refs_by_modality={
            "vision": ["evidence/vision/day/img_1.json"],
            "touch": ["evidence/touch/day/touch_1.jsonl"],
            "audio": ["evidence/audio/day/voice_1.aud"],
        },
        max_inline_refs=4,
    )

    assert pack["schema"] == "evidence.multimodal_ref_pack.v1"
    assert pack["count"] == 3
    assert pack["index_ref"] == ""
    assert sorted(pack["refs_by_modality"].keys()) == ["audio", "touch", "vision"]
    assert pack["refs_by_modality"]["touch"][0]["artifact_ref"] == "evidence/touch/day/touch_1.jsonl"
    assert pack["refs_by_modality"]["touch"][0]["modality"] == "touch"


def test_pack_multimodal_refs_uses_one_index_when_total_is_large(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    refs_by_modality = {
        "vision": [f"evidence/vision/day/img_{idx}.json" for idx in range(4)],
        "touch": [f"evidence/touch/day/touch_{idx}.jsonl" for idx in range(4)],
        "audio": [f"evidence/audio/day/voice_{idx}.aud" for idx in range(4)],
    }

    pack = store.pack_multimodal_refs(
        refs_by_modality=refs_by_modality,
        max_inline_refs=8,
        timestamp=1787133192.1,
        source="unit_test",
    )

    assert pack["schema"] == "evidence.multimodal_ref_pack.v1"
    assert pack["count"] == 12
    assert pack["refs_by_modality"] == {}
    assert pack["index_ref"].startswith("evidence/multimodal/2026-08-19/multimodal_ref_index_")
    assert pack["index_card"]["schema"] == "evidence.ref.v1"
    assert pack["index_card"]["claims_supported"] == ["evidence.multimodal_ref_index"]

    index_payload = store.read_json_artifact(pack["index_ref"])
    assert index_payload["schema"] == "evidence.multimodal_ref_index.v1"
    assert index_payload["count"] == 12
    assert index_payload["modalities"] == ["audio", "touch", "vision"]
    grouped = store.read_multimodal_ref_index(pack["index_card"])
    assert sorted(grouped.keys()) == ["audio", "touch", "vision"]
    assert len(grouped["vision"]) == 4
    assert len(grouped["touch"]) == 4
    assert len(grouped["audio"]) == 4


def test_pack_multimodal_refs_accepts_flat_evidence_cards(tmp_path: Path) -> None:
    store = EvidenceArtifactStore(tmp_path)
    cards = [
        build_evidence_card(modality="vision", artifact_ref="evidence/vision/a.json", confidence=0.7),
        build_evidence_card(modality="touch", artifact_ref="evidence/touch/a.jsonl", confidence=0.6),
    ]

    pack = store.pack_multimodal_refs(refs_by_modality=cards, max_inline_refs=4)

    assert pack["count"] == 2
    assert sorted(pack["refs_by_modality"].keys()) == ["touch", "vision"]
    assert pack["refs_by_modality"]["vision"][0]["modality"] == "vision"
    assert pack["refs_by_modality"]["touch"][0]["modality"] == "touch"
