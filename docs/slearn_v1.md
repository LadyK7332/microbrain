# Structured Learning Sheets (`/slearn`) v1

`/slearn` is a control-plane curriculum ingest path for the CAPS teaching grammar.
It lets the operator prepare repeatable day-to-day speech sheets without spending
hours typing every rule through Textual.

## Rule shape

One rule per line:

```text
IF USER says moin THEN CLASSIFY social_greeting, warmth, friendly AND REPLY good morning
IF USER says thanks THEN CLASSIFY gratitude, warmth, friendly AND REPLY you're welcome
IF USER says stop THEN CLASSIFY boundary, user_serious AND NOT REPLY playful
```

Lines beginning with `#` or `//` are ignored. The grammar keywords must be visibly
CAPS so normal prose cannot become behavior rules by accident.

## Commands

```text
/slearn status
/slearn on
/slearn off
/slearn next
/slearn dir <folder>
/slearn weight <1-5>
/slearn template
```

By default sheets live in:

```text
Z:\memory\slearn_dir
```

Completed files move to:

```text
Z:\memory\slearn_dir\ready
```

## Safety split

Normal `/read` stores low-trust sensory text.

`/slearn` does not store the document as conversation. It extracts CAPS rules and
emits `control/slearn` events. `syntax_learning_neuron.py` parses those events and
writes structured learned rules into memory.

This keeps curriculum ingestion separate from normal reading and ordinary user
conversation.

## v1.1: Domain rule support

`/slearn` now accepts both user-speech rules and starter curriculum domain rules.

Accepted examples:

```text
IF USER says moin THEN CLASSIFY social_greeting, warmth, friendly AND REPLY good morning
IF POWER is low THEN CLASSIFY need_power, energy_deficit, homeostasis
IF NEED exists THEN CLASSIFY need_object, internal_pressure, homeostasis
IF OBJECT detected THEN CLASSIFY base_object, noticed_thing
IF scene_delta detected THEN CLASSIFY surprise, investigation_candidate
IF relationship incomplete THEN CLASSIFY supposition_candidate
```

Rules still require CAPS control rails: `IF`, `THEN`, `ELSE`, `CLASSIFY`, `REPLY`, `NOT REPLY`, `AND`, and `SUPPRESS`.
Natural condition words such as `says`, `is`, `exists`, and `detected` may remain lowercase.

Storage behavior:

- `IF USER says ...` rules remain `syntax_rule` cells and may create trainer alignment when a `REPLY` exists.
- drive/internal-state rules become `drive_rule` cells.
- object/scene/state/action rules become `object_rule` cells.
- expectation/supposition/question/reasoning rules become `reasoning_rule` cells.
- fallback curriculum rows become `curriculum_rule` cells.

All `/slearn` cells include `source_mode: slearn`, `source_decay_bias`, and `lived_experience_can_override: true` so starter curriculum can yield to lived experience and reinforcement later.

## Operator Visibility / Audit Trail

`/slearn` now writes an operator audit trail so a curriculum run is visible even when the memory store writes quietly.

Default audit file:

```text
Z:\memory\slearn\slearn_audit.jsonl
```

Runtime status keys:

```text
slearn:active_file
slearn:chunk_index
slearn:last_result
slearn:completed_files
slearn:files_completed_count
slearn:rules_emitted_total
slearn:rules_applied_total
slearn:last_applied_rule
slearn:audit_path
```

The sidecar emits visible `ui/status` updates when a chunk emits rules and when a file completes/moves to `ready/`.

Notes:

- `rules_emitted_total` means the sidecar found valid CAPS rules and sent them to `syntax_learning_neuron`.
- `rules_applied_total` means the syntax-learning neuron accepted/stored the rule into learned memory.
- `.slearn` files move to `ready/` only after all chunks are read.
- Learned cells may be stored inside the mem-cell store rather than as obvious new top-level files. Use the audit file and status counters to verify training activity.
