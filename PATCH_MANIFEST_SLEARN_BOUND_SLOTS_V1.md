# SLEARN Bound Slots v1 — 2026-07-25

## Purpose

Replace the one-off `say ...` responder shortcut with a reusable learned-language
mechanism.

SLEARN may now capture named values from a USER-speech pattern and reuse those
values in a learned reply template.

Example:

```text
IF USER says "say {payload}" THEN CLASSIFY literal_repeat AND REPLY "{payload}"
```

This teaches the language operation in curriculum instead of teaching
`native_responder_neuron` that the English word `say` has a hard-coded output.

## Runtime shape

```text
USER text
  -> learned syntax-rule lookup
  -> template condition match
  -> bind {slot} values
  -> learned rule earns direct-response ownership
  -> render reply template from bindings
  -> act/speech
```

`native_responder_neuron` only implements the generic binding/rendering tool.
The `.slearn` file supplies the language rule.

## Guardrails

- Reply slots must be captured by the same condition.
- Bare catch-all rules such as `IF USER says "{payload}" ...` are rejected.
- Wrapper quotes around captured values are stripped (`say "haz"` -> `haz`).
- Templated replies do not create trainer-alignment utterances containing a
  literal `{payload}` placeholder.
- Unmatched templates do not contribute classifiers/DDNA/replies.
- Unresolved placeholders are never spoken literally.
- A learned reply rule can request a direct response generically; the responder
  no longer uses `say_request` as a special English-word response gate.
- Body-level sleep quieting still wins over learned response pressure.

## Included curriculum example

`examples/slearn/speech_direction.slearn`

Copy that file into the configured SLEARN directory (normally
`Z:\memory\slearn_dir`) and run/enable SLEARN. Once ingested, `say <payload>` is a
learned reusable rule rather than a code special case.

## Changed files

- `microbrain/neurons/syntax_learning_neuron.py`
  - recognizes `{slot}` placeholders
  - validates condition/reply bindings
  - stores template metadata
  - skips literal trainer-alignment creation for templated replies

- `microbrain/neurons/native_responder_neuron.py`
  - generic SLEARN slot matcher/renderer
  - matched learned replies get generic direct-response ownership
  - removes `say_request` from response shaping
  - reuses the same syntax lookup for shape + response construction

- `microbrain/neurons/input_text.py`
  - `/slearn template` now shows the bound-slot form

- `docs/slearn_v1.md`
  - documents bound speech slots

- `examples/slearn/speech_direction.slearn`
  - ready-to-ingest `say {payload}` curriculum rule

- `tests/test_slearn_bound_slots.py`
  - bound-value rendering
  - quote wrapper handling
  - unbound-slot rejection
  - catch-all rejection
  - unrelated-input rejection
  - direct-response ownership through learned rule

## Verification

Focused SLEARN/language/hypothesis suite:

```text
24 passed
```

Full repository `tests/` suite after this patch on top of Hot Memory v1:

```text
74 passed
```

Manual SLEARN ingest check:

```text
rules accepted: 1
saved cells: 1
input:  say Hazarem Ark
output: Hazarem Ark
```

## Patch order

This patch was built against the 2026-07-24 repo snapshot with Hot Memory v1
overlaid.

It supersedes `microbrain-direct-say-fix.zip`. If that direct-say patch was
already applied, apply this patch afterward; the generic SLEARN implementation
replaces the hard-coded output branch.
