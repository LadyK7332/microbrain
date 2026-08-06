# MicroBrain SLEARN composer coalescing v2.1

Changed files:
- `microbrain/sidecars/slearn_workbench.py`
- `microbrain/sidecars/read_sidecar.py`
- `microbrain/sidecars/memory_composer_sidecar.py`
- `microbrain/memory/mem_cell_composer.py`
- `microbrain/ui/dashboard/bridge.py`
- `microbrain/ui/dashboard/app.py`
- `tests/test_slearn_composer_coalescing.py`
- `docs/slearn_composer_coalescing_v2_1_20260724.md`

Purpose:
- allow a 50k-line bucket file to stage to disk before learned-memory composition;
- coalesce bulk learned commits and avoid repeated full learned-shard rewrites;
- keep now/short/long composer work running while learned is intentionally buffered;
- expose composer buffering/commit state in Window 2;
- suppress repeated identical SLEARN waiting lines.
