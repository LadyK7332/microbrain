# SLEARN bucket/workbench v2 — 2026-07-24

## Problem

The legacy SLEARN path loaded text chunks, emitted up to 100 `control/slearn`
events per chunk, and let each rule perform synchronous memory staging while the
orchestrator/Qt loop was also trying to stay responsive. Large lexical sheets
therefore looked frozen even though the process eventually caught up.

The old text chunker also re-read/split the complete text file for every chunk,
which becomes progressively wasteful on 50,000-line curriculum files.

## v2 rule

SLEARN is a background ingestion organ. It performs its internal rule work
privately and sends compact job results/status to the main bus.

### Preflight

Before parsing, SLEARN streams the file once to obtain:

- byte size
- physical line count
- average bytes/line

Mode selection is deterministic:

- `NORMAL` when both measurements are below the bulk thresholds
- `BUCKET` when file bytes **or** line count reaches its bulk threshold

Defaults:

- `slearn:bucket_min_bytes = 2 MiB`
- `slearn:bucket_min_lines = 5,000`

### Streaming cursor

Jobs persist a byte offset plus physical line number. Bucket processing seeks to
that byte cursor and reads only the next bucket; it does not reload the full
file for every step.

Defaults:

- normal batch: 80 physical lines
- bucket batch: 1,000 physical lines
- maximum outstanding composer batch receipts: 8

These are behavioral tuning values, not DDNA.

### Memory staging

Pure classifier rules are converted directly into deterministic learned memory
cells and staged as one composer file per batch. They do not become one
`control/slearn` bus event per rule.

Rule IDs are semantic/restart-stable: timestamps, source line numbers and job
nonces do not mint a new cell ID. This makes resume/replay safe against duplicate
cell spraying.

Reply-bearing rules still use the richer trainer-alignment route, but their
memory flush is deferred to the end of the batch.

### Backpressure and durable completion

SLEARN retains receipt paths for its composer batches. If too many are pending,
it pauses bucket production and reports `waiting_composer` rather than growing
an unbounded queue.

A file is not declared complete until all of its composer receipts have left
both pending and processing queues. Only then does SLEARN run post-cleanup, move
the source file to `ready`, and emit completion.

### Workspace hygiene

SLEARN owns only `memdir/slearn/workspace`.

Before a job it:

- snapshots its scratch floor
- removes stale SLEARN-owned temp files
- quarantines unknown prior job workspaces instead of deleting them
- creates a job marker

After durable commit it:

- removes the current job workspace
- clears SLEARN temp files
- verifies the active floor returned to/below its previous baseline
- releases the bulk MemCellStore/index and per-job duplicate set from process
  memory

It never prunes semantic/episodic memory or unrelated sidecar state.

### Signals

Engineering-only events:

- `slearn/preflight`
- `slearn/progress`
- `slearn/status`
- `slearn/completed`
- `slearn/blocked`
- `slearn/failed`

These feed the dedicated Window 2 SLEARN panel and are not cognitive input.

After verified completion/failure/block, one compact result event is also sent:

- `learning/completed`
- `learning/failed`
- `learning/blocked`

This is the meaningful result boundary: cognition may notice the learning job
finished, but never has to watch the ingestion machinery chew each row.

## Core architecture law

**The learning organ does the work privately; cognition receives the outcome.**
