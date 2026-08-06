# Composer Receipt-Focused Commit v1

## Purpose

Fix SLEARN `waiting_commit` stalls where the active job has a small exact receipt list, but `mem_cell/_pending/learned` contains a very large stale/unrelated backlog. In that state the composer was committing arbitrary learned files from the tier directory instead of the receipt files the current SLEARN job was waiting on, so the dashboard could show movement while the SLEARN pending count stayed frozen.

## Behavioral rule

When SLEARN is waiting on learned commit receipts, those receipt paths are authoritative. The composer should commit those exact receipt files first and must not walk or drain unrelated learned backlog before acknowledging the active job.

## Changed files

- `microbrain/memory/mem_cell_composer.py`
  - Adds `compose_receipts(...)` focused commit path.
  - Adds exact receipt resolution for pending and already-moved processing files.
  - Allows `_compose_tier(...)` to consume supplied files without scanning the full tier.
  - Avoids full pending-tier scan for `pending_remaining` when receipt-focused.
  - Replaces preflight `Path.glob()` with bounded `os.scandir()`.

- `microbrain/sidecars/memory_composer_sidecar.py`
  - Reads exact active SLEARN receipt paths from runtime KV.
  - Uses `compose_receipts(...)` when SLEARN learned flush is due.
  - Counts only exact receipts in health scans during focused SLEARN commit.
  - Exposes `receipt_focused`, `receipts_observed`, and `target_receipts_count` in composer health.

- `microbrain/sidecars/read_sidecar.py`
  - Publishes `slearn:receipt_paths` alongside `slearn:outstanding_batches`.

- `tests/test_slearn_composer_coalescing.py`
  - Adds regression coverage for exact receipt commit ignoring unrelated learned backlog.
  - Adds regression coverage for health counting exact receipts instead of backlog.

## Validation

- `python -m compileall -q microbrain`
- `python -m pytest -q tests`

Result: `90 passed`
