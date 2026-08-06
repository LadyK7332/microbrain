# Visual current-state + overlay v1 — 2026-07-24

## Design law

Frames are sensory samples. Objects are current perceptual state. Meaningful changes are experiences.

- Raw camera/window frames are RAM-first and bounded.
- `visual:current` is an ephemeral current object map in KV/RAM.
- `visual:exp` remains reserved for predicted visual state; current observations do not overwrite it.
- Ordinary frame-to-frame tracking does not write a full scene map to disk.
- `vision/object_delta` remains the meaningful change path for durable memory candidates.

## RAM frame buffer

Camera and window capture now default to `save_mode=ram`.

Recent JPEG-compressed samples live under:

- `vision:frame:latest`
- `vision:frame:ring`

Tuning:

- `vision:ram_frames_keep` — default 120 frames
- `vision:ram_frame_ttl_s` — default 10 seconds
- `vision:ram_jpeg_quality` — default 82

Disk persistence remains available explicitly with `latest`, `gated`, or `all` save modes for evidence/debug use.

## Current visual scene organ

`visual_current_scene_neuron` consumes detector/proto/delta outputs and maintains:

- `visual:current`
- `vision:current_objects`

It emits no ordinary cognitive/bus event for current-frame state. The dashboard bridge samples the RAM state directly at 10 Hz.

## Window 1

Presence / Perception now uses three top panes:

1. Vision / objects inspector
2. Live vision canvas
3. Body / proprioception

The inspector provides overlay toggles, a current object list, confidence and track IDs, plus a `Freeze tracks` diagnostic hold. Selecting an object briefly highlights its overlay and flashes a connector cue on the live feed.

Overlay states:

- green — identified with high confidence
- yellow — identified but uncertain
- blue — tracked unknown/proto-object
- red — hazard/emergency
- gray — lost/stale

High-rate raw vision samples and the dashboard's visual-state sampler are not written into dashboard transcript/history or Engineering trace. Meaningful `vision/object_delta` events remain traceable.
