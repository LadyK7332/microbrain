from __future__ import annotations

import time

from microbrain.evidence_convergence import (
    WorkingBelief,
    contradiction_anomaly,
    converge_evidence_packets,
)
from microbrain.sensory_fossils import SensoryFossilStore, normalize_tags


def test_vision_fossil_uses_grayscale_shape_plus_green_branch_tag():
    store = SensoryFossilStore()
    store.store_from_payload(
        {
            "modality": "vision",
            "concept": "button",
            "branch": "green_button",
            "source_ref": "vobj:old_green_button",
            "low_res_gray": [[20, 80, 180, 220], [22, 90, 175, 218]],
            "dominant_color": "#22AA44",
            "required_color": "#22AA44",
            "confidence": 0.86,
            "objecthood_confidence": 0.92,
            "stability": 0.88,
            "tags": ["affordance:pressable", "source:user_labeled"],
        }
    )

    packets = store.query_packets(
        {
            "modality": "vision",
            "source_ref": "vobj:12",
            "low_res_gray": [[21, 82, 177, 221], [21, 91, 177, 216]],
            "dominant_color": "#25AB46",
            "importance": 0.61,
        },
        threshold=0.50,
    )

    assert packets
    packet = packets[0]
    assert packet.candidate.startswith("green_button")
    assert "#mem_cell:button" in packet.mem_cell_tags
    assert "#branch:green_button" in packet.mem_cell_tags
    assert "#color:#22AA44" in packet.mem_cell_tags
    assert packet.confidence >= 0.65


def test_touch_and_vision_converge_into_revisable_workspace_candidate():
    now = time.time()
    packets = [
        {
            "packet_id": "vision1",
            "modality": "vision",
            "source_ref": "vobj:12",
            "candidate": "green_button?",
            "similarity": 0.78,
            "confidence": 0.76,
            "importance": 0.70,
            "fossil_refs": ["vision:fossil:green_button_001"],
            "mem_cell_tags": ["#mem_cell:button", "#branch:green_button"],
            "supports": ["vision shape fossil match 0.78", "color branch #22AA44 match 0.93"],
            "uncertainty": ["not confirmed pressable"],
            "timestamp": now,
        },
        {
            "packet_id": "touch1",
            "modality": "touch",
            "source_ref": "vobj:12",
            "candidate": "green_button?",
            "similarity": 0.72,
            "confidence": 0.71,
            "importance": 0.55,
            "fossil_refs": ["touch:fossil:raised_button_004"],
            "mem_cell_tags": ["#mem_cell:button", "#affordance:pressable"],
            "supports": ["touch pressure fossil match 0.72"],
            "uncertainty": ["click response unknown"],
            "timestamp": now,
        },
    ]

    candidates = converge_evidence_packets(packets, now_ts=now, candidate_threshold=0.60, accepted_threshold=0.80)

    assert candidates
    candidate = candidates[0]
    assert candidate.target_refs == ["vobj:12"]
    assert set(candidate.modalities) == {"vision", "touch"}
    assert candidate.confidence >= 0.80
    assert candidate.accepted_working_belief is True
    assert candidate.recommended_next == "accept_working_belief"
    assert "#mem_cell:button" in candidate.mem_cell_tags


def test_recent_high_confidence_drink_contradiction_raises_anomaly():
    now = time.time()
    belief = WorkingBelief(
        subject_ref="vobj:bang_can",
        believed_as="Bang can",
        confidence=0.91,
        evidence_packet_ids=["vision1", "text_logo1"],
        mem_cell_tags=["#drink", "#ingestion", "#mem_cell:Bang_can"],
        expected_results={"taste": "Bang flavor / carbonation"},
        accepted_at=now - 120.0,
    )

    anomaly = contradiction_anomaly(
        belief,
        {
            "subject_ref": "vobj:bang_can",
            "action": "drink",
            "matches_expected": False,
            "observed_as": "wrong taste profile",
            "tags": ["maintenance", "ingestion"],
        },
        now_ts=now,
        threshold=0.60,
    )

    assert anomaly is not None
    assert anomaly.severity >= 0.60
    assert anomaly.action == "drink"
    assert any("stop ingestion" in q for q in anomaly.questions)
    assert "vision1" in anomaly.evidence_packet_ids


def test_trailing_tags_are_structured_handles_not_only_prose():
    tags = normalize_tags(["mem_cell:flower", "fossil:touch:petal_soft", "relation:petal:part_of:flower"])
    assert tags == ["#mem_cell:flower", "#fossil:touch:petal_soft", "#relation:petal:part_of:flower"]
