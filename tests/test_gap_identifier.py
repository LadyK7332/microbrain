from __future__ import annotations

from microbrain.cognition.gap_identifier import (
    build_clarification_need,
    build_evidence_need,
    build_gap_speech_payload,
    build_speech_obligation,
    identify_gap,
)


def _silence_payload(text: str = "o.o", *, crisis: bool = False, repeats: int = 0):
    return {
        "channel": "textual",
        "context": {
            "constraints": {"allow_babble": True, "crisis_mode": crisis},
            "drives": {"boredom": {"same_user_repetitions": repeats}},
            "input": {
                "channel": "textual",
                "source": "ui",
                "text": text,
                "tokens": ["o", "o"],
                "meaningful_tokens": [],
                "raw_meta": {"source": "ui", "transport_source": "ui", "frontend": "dashboard"},
            },
        },
        "hypothesis": {
            "pattern_analysis": {
                "uncertainty": 0.84,
                "statement_kind": "minimal_statement",
                "meaningful_tokens": [],
                "tokens": ["o", "o"],
            },
            "interpretations": [
                {"interpretation": "intent_ambiguous", "confidence": 0.84},
                {"interpretation": "minimal_signal", "confidence": 0.52},
            ],
            "recommended_action": "silence",
            "response_demand": 0.0,
            "should_respond": False,
        },
        "trigger": {
            "deliberate_silence": True,
            "recommended_action": "silence",
            "response_demand": 0.0,
        },
    }


def test_user_paralinguistic_silence_becomes_gap_and_clarification():
    gap = identify_gap(
        "hypothesis/action_committed",
        _silence_payload(),
        source="desire_trigger_neuron",
        event_meta={"selected_action": "silence", "deliberate_silence": True},
        now=123.0,
    )
    assert gap["identified"] is True
    assert gap["gap_kind"] == "intent_ambiguous"
    assert gap["stimulus"]["paralinguistic"] is True
    assert gap["stimulus"]["user_originated"] is True
    assert gap["silence_allowed"] is False
    assert gap["analysis"]["response_demand_recommended"] >= 0.25
    assert "intent" in gap["missing"]

    need = build_clarification_need(gap)
    assert need is not None
    assert need["question_surface"] == "Something catch your attention?"

    obligation = build_speech_obligation(gap)
    assert obligation is not None
    assert obligation["minimum_surface_complete"] is True

    speech = build_gap_speech_payload(gap)
    assert speech is not None
    assert speech["text"] == "Something catch your attention?"
    assert speech["surface_complete"] is True


def test_crisis_blocks_question_and_allows_silence():
    gap = identify_gap(
        "hypothesis/action_committed",
        _silence_payload(crisis=True),
        source="desire_trigger_neuron",
        event_meta={"selected_action": "silence", "deliberate_silence": True},
        now=123.0,
    )
    assert gap["identified"] is True
    assert gap["human_uplift_gate"]["allowed"] is False
    assert "crisis_mode" in gap["human_uplift_gate"]["blocked_reasons"]
    assert gap["silence_allowed"] is True
    assert build_clarification_need(gap) is None
    assert build_gap_speech_payload(gap) is None


def test_empty_noise_does_not_create_gap():
    gap = identify_gap("context/built", {"context": {"input": {"text": "", "channel": "textual"}}}, source="ui")
    assert gap["identified"] is False


def test_uncertain_audio_requests_direction_and_classification_evidence():
    payload = {
        "channel": "audio",
        "label": "unknown",
        "confidence": 0.31,
        "direction": "unknown",
    }
    gap = identify_gap("percept/audio", payload, source="audio_sensor", now=456.0)
    assert gap["identified"] is True
    assert gap["gap_kind"] == "audio_evidence_gap"
    evidence_need = build_evidence_need(gap)
    assert evidence_need is not None
    assert evidence_need["modality"] == "audio"
    assert evidence_need["suggested_action"] == "stereo_direction_check_and_sound_classify"
    assert "direction" in evidence_need["missing"]


def test_uncertain_vision_requests_readjust_evidence():
    payload = {"label": "unknown", "identity_confidence": 0.22, "track_id": "vobj:7"}
    gap = identify_gap("percept/vision", payload, source="vision_sensor", now=789.0)
    assert gap["identified"] is True
    assert gap["gap_kind"] == "vision_evidence_gap"
    evidence_need = build_evidence_need(gap)
    assert evidence_need is not None
    assert evidence_need["modality"] == "vision"
    assert evidence_need["suggested_action"] == "vision_readjust_refocus_or_recenter"
