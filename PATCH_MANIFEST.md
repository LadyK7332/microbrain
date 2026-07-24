# MicroBrain heartbeat body stream patch manifest

Date: 2026-07-24

## Purpose

Move the raw periodic tick toward a body/infrastructure heartbeat model so it
can drive timing organs without polluting cognition or memory.

## Changed files

- docs/heartbeat_body_stream_rule_20260724.md
- microbrain/utils/heartbeat_stream.py
- microbrain/mind.py
- microbrain/neurons/thought_momentum_neuron.py
- microbrain/neurons/thought_turn_arbitration_neuron.py
- microbrain/neurons/capability_circulation_neuron.py
- microbrain/memory/filters.py
- microbrain/ui/textual_bridge.py
- microbrain/ui/dashboard/bridge.py

## Notes

- `body/heartbeat` is now the primary infrastructure stream.
- `clock/tick` is still emitted as a compatibility alias for older organs.
- UI bridges hide raw heartbeat packets from visible event traces by default.
- Memory classification now explicitly rejects heartbeat events.
- Thought momentum preserves decay behavior while keeping the last cognitive
  topic separate from heartbeat decay ticks.
