# Capability status strip v1 — 2026-07-24

`capability/state` is persistent body telemetry, not a useful scrolling cognition/process line.

Dashboard rule:

- Window 1: render capability state as a compact lamp strip inside the Process panel.
- Window 2: render the same state as an Engineering status-bar instrument.
- Green filled dot = component available.
- Red filled dot = component unavailable.
- Prefer actual `available_components` / `unavailable_components` over `alias_available`; aliases describe fallback satisfaction and must not make an absent physical sensor look present.
- `capability/state` is filtered from the Window 1 Process text stream and the Window 2 trace/raw-event panes.
- `capability/readiness` remains traceable because it is a specific action/thought readiness result rather than constant body state.
- Periodic dashboard snapshots carry the latest KV capability state so a newly opened dashboard can populate the lamps even when no capability changed after startup.

Runtime rule:

- Heartbeat may refresh capability expiry/KV bookkeeping internally.
- An unchanged heartbeat must not publish another `capability/state` event.
- A meaningful capability change publishes one new state plus the existing drawer recheck signal.

This keeps the workbench readable while preserving live body visibility.
