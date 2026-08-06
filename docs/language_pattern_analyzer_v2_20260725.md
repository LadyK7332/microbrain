# Language Pattern Analyzer v2 — 2026-07-25

## Design rule

English structure is evidence.

The analyzer should not require every word to be known before it can understand the job the word is probably doing. Closed-class anchors, word order, morphology, phrase shape, and surrounding context may narrow unknown words into ranked grammatical roles.

```text
text
  -> token role candidates[]
  -> phrase chunks
  -> clause candidates[]
  -> best current interpretation + confidence
  -> thought/memory structure
```

A parse is a hypothesis, not truth. Ambiguous interpretations remain candidates for later context or investigation.

## Example: unknown vocabulary

```text
The dax snorp the blen.
```

The words do not need dictionary definitions for English order to suggest:

```text
dax   -> likely entity/noun
snorp -> likely action/verb
blen  -> likely entity/noun

clause:
dax --snorp--> blen
```

Once a coherent clause is found, one bounded feedback pass may reinforce the role candidates that made the clause coherent. This is structure teaching vocabulary roles, not a permanent dictionary assignment.

## Supported structure shapes

The fallback analyzer now recognizes useful forms including:

- declarative subject/action/object
- imperative with implied listener
- WH questions with a missing/query slot
- yes/no and modal questions
- copular attribute/identity forms
- passive voice normalized to agent/action/patient
- existential `there is/are ...`
- prepositional adjuncts
- ambiguous perception forms such as `I saw her duck`

## Candidate arrays

Token roles retain alternatives and scores rather than collapsing immediately:

```text
snorp:
  verb 0.88
  noun 0.30
  adjective 0.10
```

Clause parsing does the same. Ambiguous sentences can keep several interpretations with confidence values.

## Input path

`language_atomizer_neuron` now exposes:

```text
role_candidates
phrase_chunks
clause_frames
best_clause
```

The emitted schema advances to `language.parsed.v3`.

Base utterance objects also keep a compact structure view so scene/context objects can use the same interpretation without carrying the entire analyzer payload.

## Reading path

`MemCellStore.ingest_text()` uses the same structure analyzer for both live text and document reading.

Reading chunks may contain multiple sentences. They are analyzed sentence-by-sentence and may create one `clause_frame` per sentence rather than treating an entire paragraph as one clause.

`clause_frame` cells store:

- the best current structural interpretation
- normalized subject/action/object or equivalent slots
- voice / question target / negation
- parse confidence
- nearby alternate interpretations, explicitly marked as parse candidates

This does **not** make an alternate parse a factual memory.

## Memory behavior

The structure layer improves:

- `word_role` cells for unknown words
- `thought_template` extraction
- `general_pattern` extraction
- searchable structural memory through `clause_frame`
- active/passive normalization (`Haz opened the door` ~= `The door was opened by Haz`)

Reading-generated structural cells retain the existing lower-trust reading caps.

## Dependency behavior

spaCy remains optional. If `en_core_web_sm` is installed, its POS/dependency output contributes strong evidence while MB still keeps candidate alternatives.

If spaCy or the English model is unavailable, the machine-native structure parser remains fully functional.

## Validation

```text
python -m compileall -q microbrain
pytest -q tests

82 passed
```

Focused new analyzer coverage includes unknown vocabulary, imperatives, WH questions, passive normalization, ambiguity retention, multi-sentence reading, mem-cell clause frames, and base-object integration.
