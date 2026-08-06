# Vision Object Isolation + Attention Binding v1

Date: 2026-07-26
Base lineage:
- `microbrain-src-20260724-203723.zip`
- Hot Memory v1
- SLEARN bound slots v1
- Language Pattern Analyzer v2

## Purpose

Add the next-stage visual scene behavior discussed in Orch/Neur 12:

1. isolate candidate objects from coherent frame-to-frame motion rather than creating a fresh object record every frame,
2. keep stable runtime IDs and RAM-only localized snippets,
3. use direct contour/border overlays when available,
4. let motion onset recruit curiosity/gaze and bounded local reacquisition when a target is lost,
5. treat dashboard object selection as a temporary pointing/deictic attention reference for the next user turn, never as an identity assertion,
6. reduce duplicate proto-object churn around the same crop.

## Files

### New
- `microbrain/neurons/motion_object_isolation_neuron.py`
- `microbrain/neurons/visual_attention_anchor_neuron.py`
- `tests/test_motion_object_isolation.py`
- `tests/test_visual_attention_anchor.py`
- `docs/vision_object_isolation_attention_v1_20260726.md`

### Updated
- `microbrain/vision_state.py`
- `microbrain/neurons/visual_current_scene_neuron.py`
- `microbrain/neurons/proto_object_tracker_neuron.py`
- `microbrain/neurons/gaze_controller_neuron.py`
- `microbrain/neurons/input_text.py`
- `microbrain/neurons/context_builder_neuron.py`
- `microbrain/ui/dashboard/bridge.py`
- `microbrain/ui/dashboard/app.py`

## Behavior

### Motion/object isolation

`percept/vision` frames are compared in RAM. The organ:

- estimates/subtracts simple global camera translation,
- finds local coherent frame deltas,
- joins complementary leading/trailing motion regions when they plausibly belong to one translated object,
- associates those regions with existing tracks before creating new IDs,
- promotes persistent regions to `vobj:motion:*`,
- stores a bounded JPEG crop/snippet in RAM only,
- publishes object state to the ephemeral current visual scene,
- exposes a contour/polygon for direct border rendering,
- predicts a short lost-track bbox from the last velocity and attempts local reacquisition before expiry.

Motion onset emits a small `curiosity/adjust` boost and `vision/motion_attention`; gaze locks to the target. A lost target puts gaze into a bounded nearby search pattern. Reacquisition retains the same track ID.

### Current scene duplicate suppression

Near-identical, similar-area boxes are coalesced in `visual:current`. This is deliberately strict enough to preserve legitimate nested subregions such as an eye inside a face.

The proto tracker also accepts strong spatial continuity as evidence that a live crop is the same track even when its appearance hash wobbles.

### UI pointing / attention binding

Clicking a tracked object in the Window-1 inspector now does two things:

- flashes/highlights the object as before,
- sends `control/vision_attention` with the selected track ID.

The attention organ validates that the track still exists and creates a short-lived `vision.attention_ref.v1` anchor. The next normal user text input receives this reference in `raw_meta.visual_attention_ref`. Deictic words such as `this`, `that`, and `it` also receive a `deictic_binding_hint`.

This does **not** relabel the object and does **not** convert a user claim into ground truth. It only means: "the user is pointing at this current visual thing while speaking."

## Runtime tuning keys

- `vision:isolation:enabled` (default `True`)
- `vision:isolation:max_hz` (default `8.0`)
- `vision:isolation:analysis_width` (default `320`)
- `vision:isolation:pixel_threshold` (default `20`)
- `vision:isolation:min_area_frac` (default `0.0012`)
- `vision:isolation:max_area_frac` (default `0.45`)
- `vision:isolation:association_gate` (default `0.34`)
- `vision:isolation:promote_hits` (default `2`)
- `vision:isolation:lost_grace_s` (default `0.65`)
- `vision:isolation:search_timeout_s` (default `2.75`)
- `vision:isolation:max_tracks` (default `24`)
- `vision:isolation:curiosity_boost` (default `0.34`)
- `vision:current:isolation_stale_s` (default `4.0`)
- `vision:attention:ttl_s` (default `20.0`)
- `vision:attention:turns` (default `1`)
- `vision:attention:salience` (default `0.78`)

## Verification

- `python -m compileall -q microbrain` — pass
- `python -m pytest -q tests` — **85 passed**
