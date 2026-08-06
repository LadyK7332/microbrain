# SLEARN learned-only composer flush v1

Problem observed:

- SLEARN reached `waiting_commit` on `11_wordnet_definitions_deduped_part_11_of_16.slearn`.
- Dashboard showed `pending 51` learned receipts, but the composer phase was stuck at `scan_pending now` for several minutes.
- This means the composer was servicing unrelated `now` tier scanning before flushing the learned receipts that SLEARN was waiting on.

Fix:

- During an active SLEARN bucket job, when learned flush is due because EOF was reached or the outstanding batch threshold was reached, the composer sidecar now returns only `["learned"]` for that cycle.
- Deferred bucket ingestion still excludes learned while the buffer is below threshold.
- Normal non-SLEARN operation remains unchanged.
- Added `mem_cell:composer:learned_flush_due` telemetry so the UI can show when the composer is intentionally servicing learned receipts.

Files changed:

- `microbrain/sidecars/memory_composer_sidecar.py`
- `tests/test_slearn_composer_coalescing.py`

Validation:

- `python -m compileall -q microbrain`
- `python -m pytest -q tests`
- Result: `91 passed`
