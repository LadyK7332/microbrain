from __future__ import annotations

from pathlib import Path

from microbrain.orchestrator.event_bus import EventBus
from microbrain.orchestrator.neuron_base import Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.ui.dashboard.config_catalog import scan_file
from microbrain.ui.frontend_common import extract_evidence_refs, pressure_snapshot, runtime_tuning_candidates


def test_pressure_snapshot_preserves_textual_pressure_contract() -> None:
    orch = Orchestrator()
    orch.kv_store.update(
        {
            "power:state": {"mode": "awake", "charging": True, "sleep": False},
            "drive:boredom": {"level": 0.4},
            "drive:social_interaction": {"level": 0.2},
            "thought:momentum": {"pressure": 0.3, "dominant_intent": "curiosity"},
            "mem_cell:composer:started": True,
            "read:sidecar_started": True,
        }
    )
    snap = pressure_snapshot(orch)
    assert snap["schema"] == "ui.pressure_band.v1"
    assert snap["body"]["charging"] is True
    assert snap["body"]["memory_composer"] == "on"
    assert snap["pulse"]["boredom"] == 0.4
    assert snap["pulse"]["curiosity"] >= 0.3


def test_runtime_tuning_candidates_only_exposes_scalar_tuning_keys() -> None:
    kv = {
        "vision:fps": 2.0,
        "thought:turn:ttl_s": 30.0,
        "drive:boredom": {"level": 0.2},
        "policy:enabled": True,
        "vision:last_frame": {"path": "x"},
    }
    found = runtime_tuning_candidates(kv)
    assert found == {"thought:turn:ttl_s": 30.0, "vision:fps": 2.0}


def test_evidence_refs_are_carried_references_not_disk_guesses() -> None:
    payload = {
        "image_ref": r"C:\memory\vision\frame.jpg",
        "memory_cell_ids": ["abc", "def"],
        "nested": {"audio_ref": r"C:\memory\audio\clip.wav", "note": "frame maybe elsewhere"},
    }
    refs = extract_evidence_refs(payload)
    assert {r["ref"] for r in refs} == {
        r"C:\memory\vision\frame.jpg",
        r"C:\memory\audio\clip.wav",
        "abc",
        "def",
    }


def test_config_catalog_classifies_tune_and_law(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "# ---------------------------------------------------------------------------\n"
        "# Behavioral tuning\n"
        "# ---------------------------------------------------------------------------\n"
        "FOO = 0.5\n"
        "# ---------------------------------------------------------------------------\n"
        "# Required static constants\n"
        "# ---------------------------------------------------------------------------\n"
        "BAR = 'schema.v1'\n",
        encoding="utf-8",
    )
    entries = scan_file(source, root=tmp_path)
    assert [(e.category, e.name) for e in entries] == [("tune", "FOO"), ("law", "BAR")]
