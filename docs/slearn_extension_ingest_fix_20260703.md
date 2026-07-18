# /slearn .slearn extension ingest fix - 2026-07-03

## Symptom

`/slearn on` listed `.slearn` files as candidates, then immediately marked them completed and moved them to `ready`, while `emitted_total=0` and `applied_total=0` stayed flat.

## Cause

`ReadSidecar._list_slearn_candidates()` accepted `.slearn` files, but `ReadSidecar._chunk_for()` only chunked `.txt`, `.md`, and `.pdf`. That meant `.slearn` candidates could be selected, but produced no chunk and were treated as already complete.

## Fix

Treat `.slearn` files as text chunks, same as `.txt` and `.md`, so CAPS rules can be extracted and emitted as `control/slearn` events.

## Expected result

After dropping `.slearn` files into the configured `slearn_dir`, `/slearn on` or `/slearn next` should show lines like:

```text
/slearn: lesson.slearn chunk 0 emitted 42 rule(s); emitted_total=42 applied_total=...
```

A completed file should move to `ready` only after its chunks have been emitted.
