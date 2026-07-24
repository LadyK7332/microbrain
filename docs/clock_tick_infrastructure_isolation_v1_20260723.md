# Clock Tick Infrastructure Isolation v1

## Rule

`clock/tick` is an infrastructure/metabolism trigger, not semantic input.

It may still wake neurons that explicitly subscribe to it for decay, expiry,
maintenance, scheduling, or time-based state changes. It must not gain Hebbian
weight or become a thought/context merely because it fires frequently.

## Changes

- `mind.py` marks every `clock/tick` as:
  - `event_class = infrastructure`
  - `semantic_input = False`
  - `store_in_memory = False`
  - `reinforcement_eligible = False`
  - `cognitive_visible = False`
- `BaseNeuron` rejects infrastructure/non-semantic inputs from base Hebbian
  reinforcement while continuing to deliver them to explicit subscribers.
- `BaseNeuron` now also honors the existing `reinforcement_eligible = False`
  event contract.
- `ThoughtMomentumNeuron` uses ticks only for passive decay, preserves the last
  semantic input topic, and does not emit a fresh `thought/momentum` event for a
  tick.
- `CapabilityCirculationNeuron` still refreshes and expires state on ticks, but
  only emits capability state when something actually changes. Repeated stable
  ticks no longer flood the bus.
- Capability state/readiness/recheck events are explicitly marked
  reinforcement-ineligible.

## Expected effect

A live trace should no longer show repeating `clock/tick -> thought/momentum` or
unchanged `clock/tick -> capability/state` chains. More importantly,
`clock/tick` can no longer accumulate associative/Hebbian importance.

Real time-derived changes remain visible. For example, if a capability TTL
expires because time passed, the resulting capability change may still emit a
state-change event; the scheduler pulse itself remains non-semantic.

## Existing polluted weights

Base-neuron Hebbian weights are runtime-local in the current architecture. A
normal MicroBrain restart after applying this patch starts them clean, so old
`clock/tick` weights do not need a memory migration.

## Validation

- `python -m compileall -q microbrain tests` passed.
- `python -m pytest -q tests` passed: 34 tests.
