# SLEARN composer coalescing v2.1 — 2026-07-24

Bulk `.slearn` ingestion is disk-backed and must not force the canonical learned
memory shard to be rewritten every few thousand rules.

For BUCKET mode:

- rule parsing/staging stays in the SLEARN sidecar worker;
- pending learned batches are allowed to accumulate on disk;
- the memory composer defers only the `learned` tier while a bucket job is still
  staging below the flush threshold;
- `now`, `short`, and `long` memory tiers continue composing normally;
- learned memory is released for composition when the job reaches EOF or the
  configured bulk flush threshold is reached;
- default inflight/flush threshold is 64 batches. At the current 1,000-line
  bucket size, a 50k WordNet chunk normally reaches EOF before its first learned
  shard rewrite, avoiding repeated full-shard write amplification.

The dashboard SLEARN panel reports buffering/queued/committing state and
suppresses repeated identical waiting messages.

This is a coalescing fix, not a segmented learned-memory storage redesign. The
canonical learned shard is still rewritten when a bulk commit occurs.
