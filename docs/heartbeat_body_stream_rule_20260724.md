# Heartbeat Body Stream Rule (2026-07-24)

## Rule

`clock/tick` is not cognition.

The raw periodic pulse belongs to body / infrastructure timing and should be
used like a pacemaker / heartbeat source for:

- cooldowns
- decay
- expiry
- periodic maintenance
- scheduler wakeups
- capability refresh
- chronoception updates

The raw pulse must **not** be treated as a thought, semantic input, salience
object, long-term memory candidate, or hypothesis evidence simply because it
exists on the bus.

## Architecture

Primary topic:

- `body/heartbeat`

Compatibility alias during migration:

- `clock/tick`

## Brain boundary

The body may publish raw heartbeat timing.

The brain should only consume **interpreted state** when relevant, for example:

- `time since last interaction`
- `deadline reached`
- `sleep window active`
- `activity lasted 18m`
- `capability expired`

## Implementation notes

- `body/heartbeat` is now the primary stream.
- `clock/tick` remains as a compatibility alias for older organs.
- UI bridges hide raw heartbeat packets from the visible firehose by default.
- Memory filters explicitly reject heartbeat events.
- Thought momentum keeps decay timing, but heartbeat updates no longer replace
  the last cognitive topic.
