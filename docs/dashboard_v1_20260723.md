# MicroBrain Native Dashboard v1 — 2026-07-23

## Purpose

The native dashboard is an engineering face for MicroBrain. It observes and controls through the orchestrator/bus boundary; it does not become part of cognition.

Window 1 is the **Presence / Perception** view:

- live vision frame display
- object/recognition boxes plus center-to-object line overlays
- rough body/proprioception map with touch/motor/proprio telemetry when those topics exist
- inherited Textual body/pressure strip
- conversation / speaking log
- text input with the same `/...` command path as Textual
- immediate local echo so the face stays responsive even if cognition stalls

Window 2 is the **Engineering / Internal** workbench:

- raw all-event bus firehose (clock ticks suppressed by default)
- correlation-grouped cognitive trace map
- selected-event JSON inspection
- direct evidence references carried by sensory/cognitive events
- double-click opening of referenced local sensory files
- organ/bus queue and error telemetry
- runtime tuning keys that are safe scalar KV controls
- DDNA inspection (read-only in v1)
- top-of-file `Behavioral tuning` and `Required static constants` catalogue

## Configuration law represented in the UI

- **DDNA** = nature / temperament of mind. Read-only in v1 until a DDNA Viability Validator exists.
- **Behavioral tuning** = organ/body physiology. File defaults are inspectable. Existing scalar runtime KV tuning keys can be adjusted and are audit-logged.
- **Required static constants** = anatomy, routing, protocol, safety, shared law. Always read-only.
- **Current state** = circumstance. Observable, not a setting.

## Launch

Install the optional desktop stack:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Launch:

```powershell
python -m microbrain.mind --ui dashboard --debug --memdir Z:\Memory
```

The existing Textual UI remains available during the v1 transition:

```powershell
python -m microbrain.mind --ui textual --debug --memdir Z:\Memory
```

## Two-monitor behavior

When two displays are present, Window 1 is placed on the first display and Window 2 on the second. With one display, the two windows are placed side-by-side. Both are normal native Qt windows and can be moved/maximized independently.

## Evidence rule

The dashboard never searches folders and guesses which sensory file caused a decision. It only exposes references explicitly carried through event payloads (`data_ref`, `image_ref`, `frame_ref`, `audio_ref`, `touch_ref`, `motor_ref`, memory cell IDs, etc.). This preserves traceworthiness from sensory evidence → recognition → hypothesis → output/outcome.

## Runtime tuning audit

Accepted dashboard tuning changes are recorded in:

`<memdir>/logs/dashboard_tuning_changes.jsonl`

and mirrored into:

- `dashboard:last_tuning_change`
- `dashboard:tuning_history`

The dashboard is intentionally unable to edit DDNA in v1.

## Textual feature parity migrated

- conversation log: yes
- raw event log: yes, expanded to all-event observer
- body/pressure band: yes, moved into transport-neutral `frontend_common.py`
- input and slash commands: yes
- local `/quit` / `/exit`: yes
- local `/user` display-label refresh: yes
- memdir assistant/user labels: yes
- best-effort transcript persistence: yes
- UI responsiveness while cognition chews: yes
- status/error/speech filtering: yes
- separate debug visibility: superseded by Window 2 raw/trace/organ panes

## Validation boundary

The build environment used for this patch did not provide installable PySide6/qasync wheels, so the Qt windows could not be rendered in that environment. The Python source was compile-checked, and the transport/common/event-bus/config-catalog logic is covered by non-Qt tests. Runtime Qt validation should be done on the Windows project host after installing `requirements-dashboard.txt`.
