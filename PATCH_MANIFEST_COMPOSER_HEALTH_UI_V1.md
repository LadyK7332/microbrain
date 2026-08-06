# MicroBrain — Memory Composer Health UI v1

Patch-only archive. Overlay onto the current MicroBrain repo root.

## Changed files
- `microbrain/sidecars/memory_composer_sidecar.py`
- `microbrain/ui/dashboard/bridge.py`
- `microbrain/ui/dashboard/app.py`
- `tests/test_slearn_composer_coalescing.py`

## What this adds
- independent composer health pulse so a long/hung compose cycle remains observable
- worker/task alive state
- current busy age and active tiers
- raw pending vs `_processing` queue counts per tier without mutating recovery state
- composer lock presence + lock age
- cycle index, last cycle duration, last successful cycle time
- actual exception type/text/time and consecutive error count
- SLEARN dashboard rows for Worker / Queue / Cycle / Fault
- `busy_long` warning after 60 seconds; this is intentionally a warning, not proof of a deadlock

## Validation
- `python -m compileall -q microbrain tests` passed
- focused SLEARN/composer tests: 12 passed
- `pytest -q tests`: 88 passed
- repo-wide bare `pytest -q` still collects `microbrain/utils/capture_test.py` and fails because optional `mss` is not installed in this environment; unrelated to this patch
