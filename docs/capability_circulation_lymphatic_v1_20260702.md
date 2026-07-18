# Capability Circulation / Lymphatic Readiness Layer v1

This patch adds a passive glue neuron for MicroBrain/Exis Mechanica: `CapabilityCirculationNeuron`.

The purpose is not to command behavior. The neuron acts more like a lymphatic system for capability state. It circulates what is possible, what is blocked, and which redundant routes can satisfy a requirement.

## Core rule

Organs do work. Thoughts express need. Actions consume resources. Capability circulation tells the rest of MB what is currently possible.

## Why this exists

Thought turn arbitration already creates `thought.obj` entries with required components. The missing piece was a passive status layer that could connect tool/equipment/organ availability to those thoughts without making the thought system own every device rule.

This layer watches:

- `component/status`
- `equipment/status`
- `organ/status`
- `control/capability`
- `power/state`
- `thought/object`
- `thought/action_candidate`
- `clock/tick`

It emits:

- `capability/state`
- `capability/readiness`
- `thought/drawer_recheck`

It does not emit `act/speech`.

## Redundancy

The layer supports fallback groups. For example:

- `textual_available` may be satisfied by `audio_available`
- `lidar_available` may degrade to `depth_available` or `vision_available`
- `motion_available` may degrade to `user_assist_available`
- `safety_clear` may be supported by `guardian_clear` or `hazard_clear`

This gives MB organism-like redundancy without letting every organ fight for control.

## Thought drawer integration

When a capability changes, the neuron emits `thought/drawer_recheck`. The thought turn neuron then rechecks waiting thoughts. If the missing components are now available, a waiting thought can become a `thought/action_candidate`.

This creates the intended flow:

`need pressure -> thought.obj -> missing tools -> drawer -> capability changes -> recheck -> action candidate`

## Storage policy

Capability events are non-speech and non-memory by default. They are body telemetry, not lived semantic memory.

They may update KV state:

- `capability:components`
- `capability:available_components`
- `capability:alias_available`
- `capability:state`
- `capability:last_readiness`
- `capability:readiness:<thought_id>`

## Design warning

This is not a central controller. It should remain a passive circulation system. Other organs may consult it, but it should not select final actions, speak to the user, or override safety/goal gates.

