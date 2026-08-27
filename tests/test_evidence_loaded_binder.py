from __future__ import annotations

from microbrain.evidence.evidence_loaded_binder import build_evidence_observation, route_topic_for_observation


def test_build_observation_caps_items_and_routes_hypothesis() -> None:
    loaded = {
        "schema": "evidence.loaded.v1",
        "ok": True,
        "artifact_ref": "evidence/touch/day/contact.jsonl",
        "artifact_kind": "jsonl",
        "mode": "directed",
        "query": "soft fuzzy contact",
        "item_count": 9,
        "scanned_count": 100,
        "byte_count": 4096,
        "matched_terms": ["soft", "fuzzy"],
        "items": [{"index": i, "summary": f"soft sample {i}", "extra": list(range(20))} for i in range(9)],
        "request_id": "evidence_req:abc",
        "trigger_topic": "hypothesis/contradiction",
        "route_reason": "hypothesis_contradiction_needs_ordered_evidence",
    }
    obs = build_evidence_observation(loaded)
    assert obs["schema"] == "evidence.observation.v1"
    assert obs["ok"] is True
    assert obs["status"] == "bounded_sample_not_truth"
    assert obs["raw_policy"] == "observation_only_no_raw_memory_ingest"
    assert obs["item_count"] == 9
    assert len(obs["items_sample"]) == 4
    assert "loaded 9 matching jsonl evidence" in obs["summary"]
    assert route_topic_for_observation(obs) == "hypothesis/evidence_observation"


def test_error_observation_stays_bounded() -> None:
    obs = build_evidence_observation(
        {
            "schema": "evidence.load_error.v1",
            "ok": False,
            "artifact_ref": "evidence/touch/missing.jsonl",
            "mode": "walk",
            "error": "artifact not found",
            "trigger_topic": "safety/uncertain_action",
        }
    )
    assert obs["ok"] is False
    assert obs["confidence_hint"] == 0.0
    assert obs["error"] == "artifact not found"
    assert route_topic_for_observation(obs) == "safety/evidence_observation"


def test_unknown_trigger_only_gets_generic_observation() -> None:
    obs = build_evidence_observation({"ok": True, "item_count": 1, "items": [{"summary": "x"}], "trigger_topic": "other/topic"})
    assert route_topic_for_observation(obs) == ""
