# Reward / Novelty / Boredom Pulse v1 — 2026-07-03

## Purpose

The Textual pressure band exposed a useful symptom: salience, curiosity,
expression, trainer alignment, and thought pressure were moving, but dopamine
and boredom were too static during interaction/training.

This patch adds a fast affect/metabolism coupling layer so teaching has visible
and usable internal reward/novelty signals.

## Design rule

Teaching needs symptoms.

If MB receives interaction, novelty, trainer correction, or positive
reinforcement, the pressure band should twitch and the internal state should move:

- dopamine / reward rises for positive feedback and successful correction
- novelty rises for new interaction or new output attempts
- boredom is relieved by novel activity and reward
- repeated output can still increase stale pressure

## New neuron

`microbrain/neurons/reward_novelty_pulse_neuron.py`

Inputs:

- `percept/text`
- `percept/vision`
- `act/speech`
- `control/reinforce`
- `control/trainer_correction`
- `clock/tick`

KV outputs:

- `affect:reward_state`
- `affect:novelty_state`
- `drive:boredom_relief`

Event output:

- `affect/reward`

This neuron does not speak and does not plan. It circulates quick reward/novelty
pressure for other organs and the UI to read.

## Boredom coupling

`boredom_drive_neuron.py` now listens to:

- `control/reinforce`
- `control/trainer_correction`
- `affect/reward`

Positive reinforcement and trainer correction now relieve boredom instead of
leaving boredom pinned high. Novel interaction also has a stronger relief effect.

## UI pressure band

`textual_bridge.py` now reads `affect:reward_state` first for dopamine, falling
back to `/r` reinforcement if needed. Curiosity can also see novelty pulse as a
small contributor.

## Expected visible behavior

After positive `/acc` or successful trainer correction:

```text
pulse> sal ... | dop ↑ | bored ↓ | cur ↑ | expr ... | train ↑ | think ...
```

If reward or novelty does not move, the pressure band now exposes where the
teaching circuit is flat.
