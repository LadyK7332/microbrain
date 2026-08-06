# Vision Object Isolation + Attention Binding v1

## Rule

The visual system should maintain a localized, persistent scene representation rather than rediscovering the same thing every frame.

A coherent group of changing pixels/features is evidence for an independent visual object or object-part. The system should first ask:

> What changed together, and does it correspond to an existing track?

Only genuinely new coherent regions should create new visual IDs.

## Pipeline

```text
frame(t-1) + frame(t)
        ↓
global camera-motion estimate
        ↓
local frame delta
        ↓
coherent motion grouping
        ↓
association against existing tracks
        ↓
existing ID? ─ yes → update in place
        │
        └─ no → candidate region
                  ↓
              persists?
                  ↓
             tracked object
```

A tracked motion-isolated object may carry:

- stable runtime track ID,
- scene-relative bbox,
- direct contour/border,
- motion vector/state,
- isolation confidence,
- RAM-only crop/snippet reference,
- first/last seen timestamps,
- later recognition/identity candidates.

Isolation is not the same as semantic recognition. A border means "MB currently treats these pixels/features as one coherent visual thing." A label is a separate recognition hypothesis.

## Motion attention

Motion onset is salient before identity is known.

```text
motion onset
   ↓
object isolation
   ↓
curiosity boost
   ↓
gaze lock/follow
   ↓
track lost?
   ↓
predict nearby continuation
   ↓
bounded local search
   ↓
reacquire same ID or expire
```

This is intended to support learned object permanence/occlusion behavior later rather than hand-coding mature hunting/search behavior.

## User pointing / selected-object context

Selecting a current visual object in the dashboard is treated like pointing at it.

```text
click visual object
      ↓
temporary ATTENTION_REF
      ↓
next user input
      ↓
"this / that / it" has a strong current visual referent
```

The selection is context only. It must not:

- assign an identity,
- create a factual memory,
- force recognition confidence,
- make a later user label automatically true.

The anchor is consumed after the configured relevant turn count or expires after its TTL.
