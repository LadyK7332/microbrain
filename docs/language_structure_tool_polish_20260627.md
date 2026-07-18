# Language Structure Tool Polish — 2026-06-27

This patch polishes the existing semantic/language learning layer so MB stores **word jobs** and **thought templates**, not only raw token anchors and adjacent n-grams.

## Design law

```text
Words are tools.
Tools form relations.
Relations become thought templates.
Thought templates become better learning surfaces.
```

This keeps the old scaffold, but raises the useful memory layer from:

```text
token_anchor
adjacent_bigram
adjacent_trigram
```

toward:

```text
word_role
thought_template
pattern_linker
```

## What changed

### `microbrain/language_scaffold.py`

`TokenAtom` now carries richer parse information:

```text
idx
norm
tag
shape
head_idx
head_text
head_lemma
ent_type
```

The spaCy model is now loaded lazily. If `en_core_web_sm` is not installed, the scaffold falls back to a lightweight heuristic parser instead of killing startup.

### `microbrain/neurons/language_atomizer_neuron.py`

The atomizer now emits `language/thought_templates` and expands `atom_candidates` with:

```text
word_roles
thought_templates
```

Word roles classify how a word is being used as a cognitive tool:

```text
speaker_self_reference
listener_reference
need_or_drive_relation
action_or_process
time_or_urgency_modifier
preference_relation
structure_connector
attribute_modifier
```

Thought templates include forms such as:

```text
need_action
query_need_action
request_action
preference_action
assert_attribute
action_relation
question_about
```

Example:

```text
I need to charge soon
```

becomes:

```text
NEED_ACTION(
    subject = i,
    subject_ref = speaker_self,
    relation = need,
    action = charge,
    need_type = power_recovery,
    urgency = soon
)
```

### `microbrain/memory/mem_cell_store.py`

The memory cell store now creates:

```text
word_role cells
thought_template cells
```

in addition to the older token/pattern/general/linker cells.

This means rough surface fragments still exist as evidence, but durable learning now has better structure to promote.

### `microbrain/neurons/memory_logger_neuron.py`

`memory:last_memcell_ingest` now reports:

```text
word_role_ids
thought_template_ids
```

so the UI/debug layer can see whether structure extraction is working.

### `microbrain/memory/answer_composer.py`

Memory answer assembly now treats `thought_template` as structural material and can render basic thought templates back into readable summaries.

### `microbrain/memory/builder_forge.py`

The builder forge recognizes new template types as action/relation material:

```text
request_action
need_action
query_need_action
preference_action
action_relation
```

## Why this matters

The old memory behavior could preserve things like:

```text
I need
to charge
charge soon
```

That creates phrase recall but weak understanding.

The new behavior preserves the structure:

```text
self/speaker -> need relation -> charge action -> soon urgency
```

That lets later CAPS/internal-state rules ask whether the structure is currently true before it becomes speech.

## Battery-goblin impact

`I need to charge soon` is no longer just a loose semantic phrase. It is stored as a self-status need/action template with:

```text
need_type = power_recovery
urgency = soon
subject_ref = speaker_self
```

That gives the speech gate and CAPS bridge a cleaner target:

```text
self_status phrase remembered
→ check current internal state
→ check scene relevance
→ generate fresh speech only if appropriate
```

## Validation

Validated with:

```text
python -m compileall -q microbrain
PYTHONPATH=. pytest -q tests
```

Targeted tests added:

```text
tests/test_language_structure_memcells.py
```

Full bare `pytest -q` in this archive still tries to collect `microbrain/utils/capture_test.py`, which requires the optional `mss` package. That pre-existing optional dependency is unrelated to this patch.
