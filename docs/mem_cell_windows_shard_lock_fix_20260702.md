# Mem-Cell Windows Shard Lock Fix — 2026-07-02

## Problem

The read sidecar was repeatedly failing while ingesting into `mem_cell/now/now.jsonl` on Windows:

`PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`

The failing call was the canonical shard rewrite path:

`now.jsonl.tmp -> now.jsonl`

## Cause

Older mem-cell writes used one fixed temporary filename per tier:

`now.jsonl.tmp`

That is brittle on Windows, especially when MB has the face, read sidecar, memory organs, file watchers, antivirus/indexing, or multiple `MemCellStore` instances touching the same shard. If two writers race, or if another process has the target open during `os.replace`, Windows refuses the replacement.

## Fix

`MemCellStore` now uses:

- per-tier in-process `RLock`
- per-shard process-wide `RLock`
- best-effort cross-process `.lock` files
- unique temporary filenames per process/thread/timestamp
- short retry/backoff around `Path.replace`
- cleanup of temporary and lock files after successful or failed writes

## Design note

This keeps canonical shard rewrites, but makes them less brittle under sidecar pressure. It does not turn the read sidecar into a central memory owner. It simply gives the shard writer a proper body rule: only one writer should replace a living shard at a time.
