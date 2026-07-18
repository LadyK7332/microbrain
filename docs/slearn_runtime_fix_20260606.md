# /slearn Runtime Visibility Fix

This patch starts the `ReadSidecar` from `mind.py`. Before this patch, `/slearn on` and `/slearn next` could update KV state, but the background sidecar that scans `Z:\memory\slearn_dir` was not actually started during normal runtime.

`/slearn status` now reports:

- whether the sidecar started
- active file and chunk
- completed file count
- emitted/applied rule totals
- audit path
- last result

Default sheet folder remains:

```text
Z:\memory\slearn_dir\
```

Audit path:

```text
Z:\memory\slearn\slearn_audit.jsonl
```
