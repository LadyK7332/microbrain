# Single-instance memdir lock

MicroBrain now claims a process-level lock at boot:

`<memdir>/runtime/microbrain.instance.lock`

This prevents two full `python -m microbrain.mind` bodies from using the same memory directory at the same time.

Why this exists:

- The memory composer makes mem-cell shard writes single-writer inside one MB body.
- It cannot protect against two whole MB runtimes launched against the same `Z:\memory`.
- On Windows, two bodies can still fight over memory files and produce `WinError 32` locks.

Behavior:

- First MB process creates the lock and continues.
- A second MB process using the same memdir exits before starting sidecars/UI.
- If the previous process was killed, the next boot checks the recorded PID and clears stale locks automatically.

This is a body-level guard, not an organ-level controller.
