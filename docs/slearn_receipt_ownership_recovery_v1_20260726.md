# SLEARN receipt ownership / crash recovery v1 — 2026-07-26

## Observed failure

After an abrupt UI/process shutdown, a later SLEARN job could inherit composer
receipt paths from an older ingestion attempt. The dashboard then showed a
finished or active file waiting on a fixed number of memory-composer commits.
Removing the stale entries from `_slearn_state.json -> receipts` allowed SLEARN
to continue.

The important evidence was that the active job's `started_at` was newer than the
`queued_at` timestamps embedded in the pending composer receipt files.

## Rule

A SLEARN job may wait only on composer receipts owned by its current ingestion
attempt.

Receipt ownership is checked with two existing pieces of evidence:

1. `writer_id` in the composer envelope must belong to the active stable job.
2. `queued_at` must not predate the active state's persisted `started_at`.

The second check matters because `stable_job_id()` intentionally stays stable
for the same source revision across process restarts. A writer ID alone therefore
cannot distinguish a clean new attempt from abandoned receipts produced by an
older attempt of the same file.

## Recovery behavior

When stale/foreign receipts are found:

- they are detached from the active SLEARN state's `receipts` list;
- they are **not deleted** from `mem_cell/_pending` or `_processing`;
- the memory composer remains free to commit valid abandoned work;
- SLEARN records a compact `receipt_recovery` summary in its persisted state;
- a `slearn/recovery` engineering-only diagnostic is emitted;
- the recovery action is added to the SLEARN workspace action list.

This keeps crash recovery conservative: the current learning job stops waiting
on somebody else's receipt without destroying potentially valid staged memory.

## State/file coherence

The persisted `_slearn_state.json` active file is now preferred over runtime KV
when recovering a job. A v2 state's `job_id` is also checked against the stable
ID of the exact source revision. If they disagree, SLEARN starts a clean attempt
rather than applying a stale cursor/receipt set to another file revision.

## Architecture boundary

**Receipt ownership is a SLEARN bookkeeping relationship, not ownership of the
memory payload itself. Detaching a receipt never authorizes SLEARN to delete the
payload.**
