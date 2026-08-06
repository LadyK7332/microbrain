# Composer Lock Recovery v1

- Avoids acquiring the global mem-cell composer lock when selected tiers have no pending/processing work.
- Adds composer lock owner metadata (`pid`, `thread`, `owner_id`, `ts`).
- Recovers stale/dead-owner `_composer.lock` files before processing real work.
- Reports lock timeouts as real composer errors so the SLEARN dashboard shows the culprit instead of only queue-scan noise.
- Adds regression checks for no-work lock skipping and dead-owner lock recovery.
