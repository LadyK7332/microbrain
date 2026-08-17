# Vision Pixel Ownership v1 — Monocular Scene Projection

This patch adds the single-camera version of the pixel-mapping idea:

```text
camera frame
→ existing motion/object isolation
→ pixel ownership mask per vobj
→ object-only extraction crop/mask
→ contour/spline-ish outline
→ current scene-map projection
→ later fossil/convergence organs
```

It does **not** add stereo depth.  One camera can only produce visible-surface evidence.  Any hidden side, backside, or 3D volume remains `object.exp` / hypothesis until motion, touch, stereo, depth, or other evidence confirms it.

## Design law

```text
Bounding boxes locate attention.
Pixel masks define object ownership.
Contours/splines summarize shape.
Scene maps remember where extractions belong.
Fossils decide later whether evidence deserves durable memory.
```

## Added organ

`microbrain.neurons.vision_pixel_ownership_neuron.VisionPixelOwnershipNeuron`

Subscribed topic:

```text
vision/object_isolation
```

Output topic:

```text
vision/pixel_ownership
```

Runtime KV written:

```text
vision:pixel_ownership:last
scene:vision:pixel_ownership
vision:pixel_ownership:label_map
vision:pixel_ownership:extracts
```

The label map is RAM-only.  Durable memory should not store the full frame or the full label map by default.

## What gets extracted

For each current vobj, the organ creates an extraction artifact with:

```text
bbox_xywh
centroid_xy_px
mask_rle
rgba_png_bytes      # transparent object crop, RAM artifact
gray_png_bytes      # tiny grayscale fossil candidate, RAM artifact
gray_dhash
dominant_color_hex
contour_spline
source_frame_ref
track_id
```

The extraction stores only object-owned pixels inside the tight crop, not the whole source frame.

## GPU note

v1 is CPU-first and cheap by default.  It exposes:

```text
vision:pixel_ownership:accelerator
```

but reports `effective: cpu`.  A GPU path can be added later behind that key using a small budget so vision does not consume compute needed by other organs.

Suggested budget rule:

```text
pixel ownership may use GPU only as an opportunistic sidecar;
leave VRAM/compute headroom for LLM mouth, dashboard, future stereo/depth,
and body control.
```

## Knobs

```text
vision:pixel_ownership:enabled = True
vision:pixel_ownership:max_hz = 6.0
vision:pixel_ownership:max_extract_px = 96
vision:pixel_ownership:max_objects = 24
vision:pixel_ownership:max_artifacts = 48
vision:pixel_ownership:accelerator = "cpu"
```

## Memory boundary

This patch does not promote vision extractions into durable mem_cells by itself.

```text
Full frame = sensation
Label map = current ownership, RAM only
Extraction = object evidence, RAM first
Fossil = promoted compressed memory
mem_cell = meaning/identity branch
```
