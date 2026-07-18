# /r reinforcement visible-turn snapshot fix — 2026-06-26

## Problem

`/r a N` and `/r u N` built their snapshot from HRM recent indices only.
That made the reinforcement menu drift away from the visible conversation lane
when HRM skipped an utterance or had only stale nodes available.

Observed symptom:

```text
Demi> I need to charge soon.
you> /r a 5
status> Reinforcement snapshot
1) hrm_idx=1 good evening
```

The operator wanted to reinforce the recent visible assistant output, but `/r`
was targeting an old HRM node.

## Fix

`TextInputNeuron._handle_r_command()` now builds `/r` snapshots from the rolling
`conversation:scene.turns` list first. That list tracks visible user/assistant
turns and is a better source for operator reinforcement.

HRM remains a fallback only when no conversation-scene turns are available.

Snapshot display now labels items as either:

```text
turn_idx=<n>
```

for conversation-scene items, or:

```text
hrm_idx=<n>
```

for HRM fallback items.

## Expected behavior

```text
Demi> I need to charge soon.
you> /r a 5
status> Reinforcement snapshot [assistant]
1) turn_idx=...  I need to charge soon.
```

`/r clear` and `/r +/- index` behavior is unchanged.
