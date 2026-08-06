# Composer Phase Telemetry v1

Purpose: make the SLEARN/composer window identify the exact composer phase that is stuck, instead of only showing `busy` or `queue scan STALLED`.

## Changed files

- `microbrain/memory/mem_cell_composer.py`
  - adds thread-safe fine-grained composer telemetry
  - phases include lock wait/acquired, recover processing, scan pending, move pending, read pending, operations loaded, apply operations, flush tier, cleanup processing, tier done, idle, and error
  - exposes `telemetry_snapshot()` for the sidecar/UI

- `microbrain/sidecars/memory_composer_sidecar.py`
  - includes `compose_phase` in `mem_cell:composer:health`
  - continues using cached queue scans so UI/startup paths do not touch slow storage

- `microbrain/ui/dashboard/app.py`
  - displays current composer phase, tier, phase age, file, detail, and operation counts in the SLEARN/composer panel
  - prioritizes long composer phase fault text over queue-scan stall text so real commit stalls are easier to identify

- `tests/test_mem_cell_composer.py`
  - adds coverage for composer phase telemetry

- `tests/test_slearn_composer_coalescing.py`
  - adds coverage that sidecar health includes compose phase telemetry

## Validation

- `python -m compileall -q microbrain`
- `python -m pytest tests -q`

Result: `88 passed`
