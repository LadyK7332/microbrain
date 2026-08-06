# Heartbeat Isolation v2 patch manifest

Baseline:
`microbrain-src-20260724-175048-dashboard-compact-slearn-capability-slearn-workbench-v2_1-vision-current-overlay-ram-v1.zip`

Purpose:
- formalize the canonical 20 TPS / 50 ms body heartbeat
- physically isolate raw cadence from the meaningful/cognitive bus
- retire dual-emission `clock/tick` behavior while keeping a routing alias
- derive target-specific organ service cadence
- implement synthetic-adrenaline service reallocation without changing 20 TPS
- migrate legacy tick consumers to organ service targets
- keep heartbeat out of memory, reinforcement, semantic activation, and UI trace
- make cognition-service housekeeping silent unless meaningful state changes
- expose heartbeat/arousal health as a Window 2 instrument

See `docs/body_heartbeat_isolation_v2_20260724.md` for the full contract.
