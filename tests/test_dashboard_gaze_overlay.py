from __future__ import annotations

from microbrain.ui.dashboard.gaze_overlay import gaze_overlay_geometry


def test_maps_gaze_to_letterboxed_image():
    result = gaze_overlay_geometry(
        {"x": 0.25, "y": 0.75, "radius": 0.10, "mode": "lock", "target_motion_id": "track:17"},
        image_left=120, image_top=30, image_width=640, image_height=360,
    )
    assert result is not None
    assert result["cx"] == 280.0
    assert result["cy"] == 300.0
    assert result["aperture_radius"] == 36.0
    assert result["mode"] == "lock"
    assert result["track_id"] == "track:17"


def test_missing_or_invalid_state_has_no_fake_reticle():
    for state in (None, {}, {"mode": "roam"}, {"x": float("nan"), "y": 0.5}):
        assert gaze_overlay_geometry(
            state, image_left=0, image_top=0, image_width=640, image_height=480,
        ) is None


def test_coordinates_and_radius_are_clamped():
    result = gaze_overlay_geometry(
        {"x": -3, "y": 7, "radius": 2},
        image_left=0, image_top=0, image_width=320, image_height=240,
    )
    assert result is not None
    assert result["cx"] == 0.0
    assert result["cy"] == 240.0
    assert result["aperture_radius"] == 0.35 * 240
