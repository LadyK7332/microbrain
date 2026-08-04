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


def test_dashboard_snapshot_exposes_slearn_engineering_state(tmp_path: Path) -> None:
    from microbrain.ui.dashboard.bridge import DashboardBridge

    orch = Orchestrator()
    orch.kv_store.update(
        {
            "slearn:sidecar_started": True,
            "slearn:enabled": True,
            "slearn:active_file": str(tmp_path / "wordnet_12.slearn"),
            "slearn:chunk_index": 17,
            "slearn:files_completed_count": 9,
            "slearn:rules_emitted_total": 42000,
            "slearn:rules_applied_total": 41950,
            "slearn:mode": "bucket",
            "slearn:workspace": {"clean": True},
            "slearn:last_result": {"summary": "bucket 17 committed"},
            "visual:current": {
                "schema": "visual.current.v1",
                "frame_ref": "frame.jpg",
                "object_count": 1,
                "objects": [{"track_id": "chair-1", "label": "chair", "confidence": 0.9}],
            },
        }
    )
    bridge = DashboardBridge(orch, memdir=str(tmp_path))
    snap = bridge.runtime_snapshot()
    assert snap["slearn"]["sidecar_started"] is True
    assert snap["slearn"]["active_file"].endswith("wordnet_12.slearn")
    assert snap["slearn"]["chunk_index"] == 17
    assert snap["slearn"]["mode"] == "bucket"
    assert snap["slearn"]["workspace"] == {"clean": True}
    assert snap["slearn"]["last_result"]["summary"] == "bucket 17 committed"
    assert snap["vision"]["object_count"] == 1
    assert snap["vision"]["objects"][0]["track_id"] == "chair-1"


def test_dashboard_bridge_attaches_ram_frame_bytes_without_persisting_them(tmp_path: Path) -> None:
    import asyncio

    from microbrain.ui.dashboard.bridge import DashboardBridge
    from microbrain.orchestrator.neuron_base import Event

    async def run():
        orch = Orchestrator()
        bridge = DashboardBridge(orch, memdir=str(tmp_path))
        orch.kv_store["vision:frame:latest"] = {
            "ref": "ram:vision:camera:7",
            "jpeg_bytes": b"jpeg-data",
        }
        await bridge._tap_event(
            Event(
                topic="percept/vision",
                payload={"frame_id": 7, "data_ref": "ram:vision:camera:7", "width": 10, "height": 10},
                source="camera_capture_neuron",
            )
        )
        msg = bridge.recv_q.get_nowait()
        assert msg.payload["jpeg_bytes"] == b"jpeg-data"
        assert msg.payload["data_ref"] == "ram:vision:camera:7"

    asyncio.run(run())
