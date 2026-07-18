# /read sidecar queue/status repair — 2026-06-07

## Problem found

`/read` was technically wired, but it was easy for it to look dead:

- `/read on` enabled reading but waited for the idle timer before doing the first pass.
- The read sidecar defaulted to `<memdir>/read_dir`, while the explicit reading writer uses `<memdir>/reading/queue`.
- Normal `/read` emitted less UI/status feedback than `/slearn`, so successful/empty/completed passes were hard to see.

## Fix

- `/read on` now forces one immediate read pass.
- Canonical intake folder is now `<memdir>/reading/queue`.
- Legacy `<memdir>/read_dir` is still scanned for backwards compatibility.
- Completed files from the queue move to `<memdir>/reading/ready`.
- `/read` now emits `read/status` and `ui/status` for:
  - no files found
  - chunk ingested
  - file completed/moved
- Read folders are created at sidecar startup so `/read status` has visible paths immediately.

## Expected use

Drop files into:

```text
Z:\memory\reading\queue\
```

Then run:

```text
/read on
```

or:

```text
/read next
```

Use:

```text
/read status
```

to inspect current file, chunk, folder, and last result.
