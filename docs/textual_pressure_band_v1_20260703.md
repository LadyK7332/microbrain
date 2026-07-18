# Textual Pressure Band v1 — 2026-07-03

Adds a two-line pressure band between the raw bus/event trace and the rolling interaction log.

## Purpose

The pressure band is teaching instrumentation, not a new speech path.

It gives the operator a compact view of whether MB's body/organ state and fast reward/novelty signals are actually responding while training, correcting, or interacting.

## Lines

### `body>` slow condition line

Stable or slower-changing variables:

- power mode
- charging/sleep state
- maintenance state
- memory composer state and pending count
- read sidecar state
- capability circulation availability count
- thought drawer ready/waiting counts

### `pulse>` fast pressure line

Fast-moving variables intended to visibly twitch with interaction:

- `sal` salience
- `dop` reinforcement/reward pulse, derived from recent `/r` or `/acc` style reinforcement
- `bored` boredom level
- `cur` curiosity/novelty pressure
- `expr` expression/social pressure
- `train` recent trainer-correction pulse
- `think` thought momentum pressure
- dominant thought intent/status

Small arrows show whether values are rising, falling, or stable.

## Architecture

`textual_bridge.py` samples existing orchestrator KV state about four times per second and sends `ui/pressure_state` directly to the Textual UI queue.

`textual_app.py` consumes `ui/pressure_state` internally and updates the middle band. These state packets are not written into the raw event pane or the conversation pane.

This keeps the face responsive while avoiding another noisy log stream.

## Design rule

Trace shows what fired.
Pressure band shows what matters now.
Interaction shows what was said.
