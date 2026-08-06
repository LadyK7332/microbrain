# Memory Composer Health UI v1 — 2026-07-26

The SLEARN panel previously exposed only a coarse composer state such as `queued 42 batch(es)`. That was not enough to distinguish a healthy long commit from a dead worker, a lock wait, an exception loop, or a batch stuck in `_processing`.

The composer sidecar now publishes `mem_cell:composer:health` once per second from a watchdog task that is independent of the compose call itself. The watchdog does not call recovery-mutating queue functions while the composer is busy; it only counts files in `_pending` and `_processing` directly.

The dashboard exposes:

- worker alive/dead and health-pulse age
- composer state (`idle`, `busy`, `busy_long`, `deferred`, `error`, `worker_dead`)
- current busy duration
- lock presence and age
- pending and processing counts by tier
- cycle index and last cycle duration
- time since last successful compose
- last batch file/row totals
- real exception type/text/time

`busy_long` begins after 60 seconds. It means "inspect this cycle" rather than "deadlock proven" because very large learned-memory commits may legitimately be slow.
