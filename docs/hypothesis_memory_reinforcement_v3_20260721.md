# Hypothesis Memory Reinforcement v3

Date: 2026-07-21

## Purpose

This patch closes the credit-assignment path between memory recall, hypothesis
selection, actual output construction, observed outcome, and later pruning.

```text
memory queried
  -> exact query trace retained
  -> one-hop associative neighbors warmed lightly
  -> selected hypothesis route receives stronger direct-use credit
  -> exact cells assembled into outward output receive higher credit
  -> later outcome strengthens or weakens the exact stable route
  -> use and success statistics contribute to lifecycle value
```

Raw hypotheses and outcome observations remain ephemeral. Reinforcement lands on
reusable memory cells and stable connection strings rather than saving each
hypothesis as durable truth.

## Asymmetric reinforcement law

```text
Positive credit may diffuse one hop.
Negative credit must be directly attributable.
```

### Query touch: soft `+`

A memory cell returned by the hypothesis query receives a small retrieval count,
activation adjustment, and `last_retrieved_ts` update.

Only explicitly traversed one-hop neighbors may receive a smaller positive
association touch. Neighbor propagation is bounded by:

- maximum one hop
- maximum neighbors per root
- maximum total neighbor activation per hypothesis cycle
- positive activation/accessibility only

A neighbor touch does not increase factual trust, successful recall count, or
promotion.

### Selected route: direct `++`

Cells explicitly attached to the selected hypothesis action receive direct-use
credit. Stable edges are written between:

```text
cell:<id> -> action:<action>
pattern:<pattern> -> action:<action>
```

This indicates that the evidence or pattern participated in the selected route,
not that the route has already been proven correct.

### Final output construction: stronger direct `++`

When `act/speech` carries exact `memory_cell_ids`, those cells receive stronger
activation, usage, and promotion credit than general hypothesis evidence.

`NativeResponderNeuron` now passes exact composer-selected cell IDs when its
reply matches the current answer bundle. `SpeechReasonNeuron` also passes the
selected candidate cell ID.

### Positive observed outcome: `+++`

A sufficiently positive, reliable outcome adds successful-recall, activation,
and promotion credit to directly used evidence/output cells. It also reinforces
stable evidence/outcome, pattern/action, and action/outcome connections.

### Negative observed outcome: direct only

A negative outcome increments failure statistics and weakens only the exact
route implicated in the action:

```text
cell:<id> -> action:<action>
pattern:<pattern> -> action:<action>
action:<action> -> outcome:<status>
```

There is no negative propagation to neighboring cells. Factual trust is reduced
only when the observer explicitly classifies the result as a direct memory
contradiction. A poor conversational choice must not make correct supporting
memory less trustworthy.

## Tuning locations

All hypothesis reinforcement deltas, caps, floors, and per-event limits are
grouped at the top of:

```text
microbrain/neurons/hypothesis_memory_reinforcement_neuron.py
```

Lifecycle/pruning value weights and caps are grouped at the top of:

```text
microbrain/memory/mem_cell_store.py
```

This keeps behavioral numbers visible and prevents scattered magic values.

## Composer-safe additive writes

Memory reinforcement is written as additive operations rather than complete row
snapshots:

```json
{
  "schema": "mem_cell.pending_reinforce.v1",
  "op": "reinforce",
  "update": {
    "cell_id": "cell:example",
    "usage_inc": 1,
    "activation_delta": 0.015,
    "promotion_delta": 0.004,
    "last_used_ts": 1780000000.0
  }
}
```

`MemCellComposer` applies all increments and performs one canonical tier flush.
This prevents two concurrent touches from collapsing into one through snapshot
merging.

## New cell statistics

Canonical search results and memory rows may now include:

```text
retrieval_count
association_touch_count
usage_count
successful_recalls
failed_uses
last_retrieved_ts
last_associated_ts
last_used_ts
```

## Pruning behavior

The lifecycle activity timestamp is now the newest of:

```text
last_seen
last_retrieved_ts
last_associated_ts
last_used_ts
```

Lifecycle value includes bounded bonuses for retrieval, associative touch,
direct use, and successful recall. Direct use and proven success carry more
value than a soft neighbor touch. Natural decay still handles unused material;
negative neighbor scoring is unnecessary.

## Inspection keys

```text
hypothesis:last_memory_reinforcement
hypothesis:memory_reinforcement_history
hypothesis:memory_reinforcement_seen
hypothesis:pending_outcome
hypothesis:last_outcome
```

## Attribution boundary

Final-output cell attribution is exact when a response builder supplies
`memory_cell_ids`. Hypothesis-decision evidence currently uses the engine's
highest-ranked recalled cell trace. That is useful first-pass attribution, but
future response assemblers can improve it by returning exact component/link IDs
for every constructed phrase or thought object.
