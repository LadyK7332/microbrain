# base.object v1 — First-pass object frame

`base.object` is the shared root frame for anything MicroBrain can notice,
compare, remember, classify, or act from.

This first pass is intentionally non-invasive. It adds a schema and a small
object-frame neuron that mirrors cognition-plane events into object frames. It
does **not** migrate existing memory yet.

## Core equivalence

```text
context = current scene
scene = current interaction/event field
interaction = sequence of events
event = sensor/action/internal-state change
base.object = structured handle MB can reference
```

## Object kinds

```text
base.object
scene.object
context.object
event.object
entity.object
state.object
action.object
utterance.object
visual.object
auditory.object
feedback.object
internal_state.object
drive.object
hormone.object
memory.object
```

Grammar roles remain language-facing references:

```text
noun-like      -> entity.object
verb-like      -> action/process.object
adjective-like -> state/property.object
```

## Current runtime keys

The first-pass neuron writes:

```text
object:last
object:recent
object:current_scene
context:current_scene
```

It also emits:

```text
object/base
object/scene
```

Those events are marked `cognitive_visible=False` so they are structural
telemetry, not normal speech or memory text.

## Design laws

```text
Everything MB can notice can become an object.
Events change objects.
Scenes combine objects.
Context is the active scene-object.
Grammar arranges objects into speech.
Classifiers point to other objects.
Hormones modulate how much objects matter.
```
