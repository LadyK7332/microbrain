# MicroBrain patch — SLEARN receipt ownership / crash recovery v1

Date: 2026-07-26

## Why

A force-close could leave pending memory-composer receipt paths behind. A later
SLEARN attempt could then wait indefinitely on those stale receipts. The captured
state showed an active job started on July 26 while its 42 receipt files had
`queued_at` timestamps from July 24/25. Manually clearing the `receipts` array
allowed learning to resume.

## Changed files

- `microbrain/sidecars/read_sidecar.py`
  - persisted SLEARN state is the crash-recovery source of truth for `active_file`;
  - validates v2 state `job_id` against the exact current source revision;
  - reconciles pending receipt ownership before backpressure/finalization;
  - uses composer envelope `writer_id` to reject foreign-job receipts;
  - uses composer envelope `queued_at` vs job `started_at` to reject older attempts
    of the same stable job;
  - detaches stale receipts without deleting their pending composer files;
  - records `receipt_recovery` state and emits engineering-only `slearn/recovery`.

- `tests/test_slearn_bucket_workbench.py`
  - adds crash-recovery coverage for a foreign-job receipt;
  - adds coverage for an older-attempt receipt with the same stable job ID;
  - verifies detached receipt files remain on disk for the memory composer.

- `docs/slearn_receipt_ownership_recovery_v1_20260726.md`
  - documents the recovery boundary and ownership rules.

## Verification

Focused SLEARN workbench tests:

`7 passed`

Project `tests/` suite:

`86 passed`

Compile check:

`python -m compileall -q microbrain` passed.

A repository-wide bare `pytest` also discovers `microbrain/utils/capture_test.py`,
which requires optional package `mss` that is not installed in this patch runtime.
That collection error is unrelated to this change; the canonical `tests/` suite
passes.
