# Composer Focused Commit v1

Purpose: keep bulk SLEARN learned-memory commits from being blocked by unrelated tier scans and stop the SLEARN UI from flickering between full health telemetry and legacy fallback payloads.

Changed files:
- `microbrain/sidecars/memory_composer_sidecar.py`
  - Treat `waiting_commit` / `waiting_composer` with outstanding SLEARN receipts as an explicit learned-only composer cycle.
  - Publish `target_tiers_reason` and `scan_tiers` so the dashboard can show why the composer is focused.
  - Restrict queue health scans to the active/target tiers during a SLEARN learned flush.
  - Use bounded `os.scandir()` counts instead of glob/stat walks for health counts.
- `microbrain/memory/mem_cell_composer.py`
  - Include phase telemetry and lock ownership/recovery support.
  - Skip global composer lock when no selected tier has actionable pending work.
  - Use bounded `os.scandir()` pending-file selection to avoid expensive network-dir stat sorting.
- `microbrain/ui/dashboard/app.py`
  - Keep the last real composer health snapshot when progress/status events do not carry one.
  - Show phase, scan tiers, and target reason without flickering to "legacy telemetry only".
- `tests/test_slearn_composer_coalescing.py`
  - Adds regression coverage for waiting-commit learned-only targeting and learned-only health scan focus.

Validation run in patch workspace:
- `python -m compileall -q microbrain`
- `python -m pytest -q tests`
- Result: `88 passed`

Operational note:
- If MB is currently stuck inside an old worker thread at `scan_pending now`, restart MB after applying this patch. Python cannot kill a currently-blocked `to_thread` scan safely in-place.
