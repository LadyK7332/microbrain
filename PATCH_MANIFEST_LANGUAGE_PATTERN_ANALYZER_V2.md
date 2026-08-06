# Patch Manifest — Language Pattern Analyzer v2

Date: 2026-07-25

## Base lineage

Apply over the current Orch/Neur 12 working lineage:

1. `microbrain-src-20260724-203723.zip`
2. `microbrain-hot-memory-v1-patch.zip`
3. `microbrain-slearn-bound-slots-v1-patch.zip`

The earlier direct-`say` emergency patch remains superseded by the SLEARN bound-slots patch.

## Purpose

Upgrade text/reading analysis from mostly per-word hints into a structure-first English pattern analyzer.

Core rule:

> Sentence structure may narrow unknown word roles, but interpretations remain weighted candidates rather than hard truth.

## Changed files

- `microbrain/language_scaffold.py`
  - optional spaCy import/model path
  - stronger no-model fallback
  - weighted word-role candidates
  - noun/verb/preposition phrase chunks
  - clause candidate frames
  - bounded clause-to-token role feedback
  - imperative/question/copular/passive/existential handling
  - multi-sentence structure bundles

- `microbrain/neurons/language_atomizer_neuron.py`
  - emits structure candidates and `language.parsed.v3`

- `microbrain/objects/base_object.py`
  - compact structure-aware grammar roles in utterance objects
  - avoids duplicate parse inside classifier construction

- `microbrain/memory/mem_cell_store.py`
  - shared analyzer used by live text and reading ingest
  - structure-refined unknown word roles
  - structure-derived thought/general patterns
  - new searchable `clause_frame` cells

- `microbrain/neurons/memory_logger_neuron.py`
  - reports clause-frame IDs in latest mem-cell ingest status

- `microbrain/sidecars/read_sidecar.py`
  - applies reading trust/activation caps to word roles, thought templates, and clause frames

- `microbrain/memory/answer_composer.py`
- `microbrain/memory/cross_modal_answer.py`
- `microbrain/memory/builder_forge.py`
  - recognize clause frames as structural material

- `tests/test_language_pattern_analyzer_v2.py`
  - 8 focused analyzer tests

- `docs/language_pattern_analyzer_v2_20260725.md`
  - architecture/design note

## Validation

```text
python -m compileall -q microbrain
pytest -q tests
```

Result:

```text
82 passed
```

`ruff` was not available in the patch-generation environment, so Ruff lint was not run. Compileall and the full project test directory passed.
