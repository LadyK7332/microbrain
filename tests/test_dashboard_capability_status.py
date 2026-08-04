from microbrain.ui.dashboard.status_signals import (
    capability_counts,
    capability_short_label,
    capability_signal_map,
)


def test_actual_component_state_wins_over_alias_fallbacks():
    payload = {
        "available_components": ["audio_available", "vision_available", "not_charging"],
        "unavailable_components": ["lidar_available", "motion_available"],
        # lidar can be satisfied by vision as a fallback, but the physical lidar
        # lamp must stay red.
        "alias_available": {"lidar_available": True, "audio_available": True},
    }
    signals = capability_signal_map(payload)
    assert signals["audio_available"] is True
    assert signals["vision_available"] is True
    assert signals["lidar_available"] is False
    assert signals["motion_available"] is False
    assert capability_counts(payload) == (3, 5)


def test_alias_map_is_supported_for_older_state_payloads():
    payload = {"alias_available": {"audio_available": True, "depth_available": False}}
    assert capability_signal_map(payload) == {
        "audio_available": True,
        "depth_available": False,
    }


def test_capability_labels_are_compact():
    assert capability_short_label("textual_available") == "text"
    assert capability_short_label("some_new_available") == "some-new"
