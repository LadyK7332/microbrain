# Textual retirement candidates — leave in place for v1

The native dashboard now carries the functionality that previously required the Textual face. **Do not delete these yet.** They remain useful rollback/maintenance artifacts while dashboard v1 is calibrated.

## Runtime files that can be retired after dashboard acceptance

1. `microbrain/ui/textual_app.py`
   - Textual widget/layout implementation.
   - Dashboard now owns conversation, raw events, pressure/body state, input, transcripts, and local quit/user-label behavior.

2. `microbrain/ui/textual_bridge.py`
   - Textual-specific orchestrator adapter.
   - Its pressure-state calculation was moved into transport-neutral `microbrain/ui/frontend_common.py`.
   - Dashboard has its own all-event bridge and runtime snapshot path.

## References to remove/update when Textual is actually retired

These are **edits, not file removals**:

- `microbrain/mind.py`
  - remove `textual` from `--ui` choices
  - remove the `run_textual_frontend` branch
  - simplify `--debug-tail` Textual-specific behavior
- `microbrain/config.py`
  - remove `textual` from the UI mode comment/default if dashboard becomes the default

## Historical artifact to keep

- `docs/textual_pressure_band_v1_20260703.md`
  - historical design/implementation record; no runtime reason to delete it

## External memdir artifacts

Old Textual logs may remain under `<memdir>/logs/`:

- `textual_raw.jsonl`
- `textual_conversation.log`

They are historical telemetry and should not be auto-deleted by the dashboard migration.

## Not a Textual retirement candidate

- `microbrain/utils/debug_tail.py`

It is a generic maintenance/log utility and still has value for REPL/headless recovery even after Textual is gone.
