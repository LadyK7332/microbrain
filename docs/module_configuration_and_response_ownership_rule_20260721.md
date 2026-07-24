# Module Configuration and Response Ownership Rule

This document records the configuration law adopted for MicroBrain modules and
applies to neurons, organs, sidecars, pattern tools, and memory components.

## Configuration placement

Each module should separate adjustable behavior from structural law near the top
of the file.

```python
# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Minimum response pressure required before outward release is considered.
# Range: 0.0-1.0. Higher values make the module quieter.
RESPONSE_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

# Schema identifier used by producers and consumers.
# Do not change without a coordinated schema migration.
HYPOTHESIS_SCHEMA = "hypothesis.obj.v1"
```

### Behavioral tuning

Behavioral tuning represents the physiology or normal operating nature of the
body and its organs. It includes adjustable thresholds, cooldowns, timeouts,
gains, caps, batch sizes, query depth, queue limits, sample rates, retry timing,
and reinforcement amounts.

Tunable values should be named, grouped, and documented with their unit,
behavioral effect, and practical or safe range where applicable. Working logic
must reference these names instead of scattered numeric or boolean literals.

### Required static constants

Static constants represent anatomy, protocol, safety, and shared structural law.
They include bus routes, schema names, serialization markers, confidence bounds,
response-ownership rules, quorum rules, contamination gates, and other values
whose casual modification would break compatibility or correctness.

Static values are inspectable, but they are not presented as adjustment knobs.

## DDNA boundary

DDNA defines the durable nature of the mind: curiosity, social pull, restraint,
risk sensitivity, novelty appetite, response selectivity, reward sensitivity,
patience, trainer sensitivity, and uncertainty tolerance.

Module defaults define organ physiology. DDNA may apply a bounded modifier to
mind-facing behavior, while current state may apply a temporary modifier.
Neither DDNA nor current state may rewrite evidence, schemas, routes, or safety
invariants.

```text
module default
+ bounded DDNA temperament modifier
+ temporary current-state modifier
= inspectable effective runtime value
```

The effective value and its contributing modifiers should be published to the KV
state when practical so behavior changes can be observed and tested.

## One-turn response ownership

External participant input has one outward response owner:

```text
participant input
-> interaction pressure observes and publishes need state
-> hypothesis path interprets reply/action/silence choices
-> one committed outward response route
```

`InteractionReleaseVectorNeuron` may still publish
`drive/interaction_request`, but it must not also emit `speech/reason` for the
same external turn. Internal interaction needs remain eligible for the need
speech route.

This prevents duplicate speech, preserves clean outcome attribution, and keeps
memory reinforcement attached to the route that actually won arbitration.

## Patch scope

The 2026-07-21 response-ownership and tuning pass applies the canonical layout
to the pattern/hypothesis/outcome/memory-reinforcement chain, the interaction
release vector, desire release arbitration, and mem-cell lifecycle writer.

Older modules can be normalized in later behavior-preserving passes. Cleanup and
retuning should remain separate so a structural refactor does not silently alter
MB's behavior.
