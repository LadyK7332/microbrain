# MicroBrain — Composer Health Nonblocking v1

Patch-only archive. Overlay onto the current MicroBrain repo root.

## Root cause

The composer health UI added a synchronous filesystem health snapshot to
`MemoryComposerSidecar.start()`. That snapshot walked pending/processing folders
and probed `_composer.lock` on `Z:\memory` before startup could log that the
sidecar or dashboard had started. A slow, overloaded, or briefly unavailable
network-backed memory directory could therefore block MB's asyncio/UI thread at
boot. The same synchronous walk was also performed around every composer cycle.

## Changed files

- `microbrain/sidecars/memory_composer_sidecar.py`
- `microbrain/ui/dashboard/app.py`
- `microbrain/mind.py`
- `tests/test_slearn_composer_coalescing.py`

## What changed

- startup publishes a cached/lightweight health snapshot with no filesystem I/O
- composer cycles no longer wait on health directory scans
- queue/lock health is sampled by one detached worker-thread task
- only one queue scan may be in flight, preventing runaway thread creation
- a stuck queue scan becomes visible as `queue scan STALLED` while MB remains alive
- cached pending/processing/lock telemetry remains available to the dashboard
- legacy pending-count telemetry no longer performs a synchronous queue walk
- startup checkpoint logs identify entry into composer and read/SLEARN startup
- actual composer work remains in its existing `asyncio.to_thread` worker path

## Expected startup log sequence

```text
speech_output neuron registered.
Starting memory composer sidecar …
Memory composer sidecar started.
Starting read/SLEARN sidecar …
Read/slearn sidecar started.
Starting native PySide6 dashboard …
```

If the memory share itself is slow, the dashboard may later show:

```text
queue scan STALLED <seconds>
```

without freezing the dashboard or preventing MB startup.

## Validation

- `python -m compileall -q microbrain tests` passed
- composer/SLEARN focused checks: `8 passed`
- full `tests/` suite: `86 passed`
