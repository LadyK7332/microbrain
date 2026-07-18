# Mem-cell Memory Composer v1 — 2026-07-02

## Problem

Windows can deny `Path.replace()` when another process has the target shard open. The read sidecar, face, memory organs, and other MB processes can all touch `mem_cell/<tier>/<tier>.jsonl`, which makes direct shard rewrites brittle.

## Design

Normal organs no longer rewrite canonical mem-cell shards.

Writers stage pending updates under:

```text
mem_cell/_pending/<tier>/*.jsonl
```

Each staged file is unique to the writer/call and contains pending `upsert` envelopes. The memory composer is the only component that drains pending files, merges rows, and rewrites canonical shards:

```text
mem_cell/<tier>/<tier>.jsonl
```

This turns memory writes into a desk-job goblin: many organs can submit paperwork, but only one composer files the official memory.

## Components

`microbrain.memory.mem_cell_store.MemCellStore`

Default writer mode stages rows for the composer. Direct canonical writes remain available with:

```python
MemCellStore(memdir, composer_enabled=False)
```

`microbrain.memory.mem_cell_composer.MemCellComposer`

Single-writer composer that drains `_pending`, applies rows to direct-mode `MemCellStore`, then writes one canonical shard per touched tier.

`microbrain.sidecars.memory_composer_sidecar.MemoryComposerSidecar`

Background sidecar started before the read sidecar. It periodically runs the composer and publishes compact status to KV.

## Rule

Writers stage. Composer composes. Canonical memory gets one desk goblin.
