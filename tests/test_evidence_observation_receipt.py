from microbrain.evidence.evidence_observation_receipt import (
    RAW_POLICY,
    build_evidence_observation_receipt,
    build_memcell_for_evidence_receipt,
    should_stage_observation_receipt,
    tier_for_observation_receipt,
)


def _observation(**updates):
    obs = {
        "schema": "evidence.observation.v1",
        "observation_id": "evidence_obs:test",
        "ok": True,
        "artifact_ref": "evidence/touch/thing.jsonl",
        "artifact_kind": "touch",
        "mode": "directed",
        "query": "soft pressure",
        "item_count": 2,
        "scanned_count": 10,
        "byte_count": 2048,
        "items_sample": [{"pressure": [1, 2, 3], "note": "bounded sample"}],
        "summary": "loaded 2 touch evidence items for query soft pressure",
        "request_id": "req-1",
        "trigger_topic": "hypothesis/contradiction",
        "route_reason": "hypothesis_contradiction",
        "priority": 0.8,
        "confidence_hint": 0.57,
    }
    obs.update(updates)
    return obs


def test_receipt_never_carries_items_sample():
    receipt = build_evidence_observation_receipt(_observation(), now=123.0)
    assert receipt["schema"] == "evidence.observation_receipt.v1"
    assert receipt["raw_policy"] == RAW_POLICY
    assert receipt["sample_digest"].startswith("sample:")
    assert "items_sample" not in receipt
    assert receipt["summary"].startswith("loaded 2 touch")


def test_receipt_staging_gate_and_tier():
    obs = _observation()
    assert should_stage_observation_receipt(obs) is True
    assert tier_for_observation_receipt(obs) == "short"

    low_scatter = _observation(mode="scatter", priority=0.2, route_reason="", trigger_topic="thought/probe")
    assert should_stage_observation_receipt(low_scatter) is False
    assert tier_for_observation_receipt(low_scatter) == "now"


def test_memcell_receipt_has_refs_not_samples():
    receipt = build_evidence_observation_receipt(_observation(), now=123.0)
    row = build_memcell_for_evidence_receipt(receipt)
    assert row["schema"] == "mem_cell.evidence_observation_receipt.v1"
    assert row["kind"] == "evidence.observation_receipt"
    assert row["artifact_ref"] == "evidence/touch/thing.jsonl"
    assert row["sample_digest"] == receipt["sample_digest"]
    assert "items_sample" not in row
    assert "items_sample" not in row.get("evidence_receipt", {})
    assert "evidence/touch/thing.jsonl" in row["links_explicit"]


def test_receipt_is_stable_for_same_inputs():
    a = build_evidence_observation_receipt(_observation(), now=123.0)
    b = build_evidence_observation_receipt(_observation(), now=999.0)
    assert a["receipt_id"] == b["receipt_id"]
