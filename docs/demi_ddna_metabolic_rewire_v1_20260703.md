# Demi DDNA metabolic rewire v1 — 2026-07-03

Purpose: let the active `pdna_profile.json` act like a genome/metabolism mutator instead of a static vibe sheet.

## Rule

DDNA does not write responses. DDNA mutates the pressures that cause responses.

## What changed

- `PDNAProfile` now preserves unknown v2 profile sections during load/save. This prevents `ddna_mutators`, `affect_model`, `reinforcement_model`, `drive_thresholds`, and `wans` from being erased the first time PDNA saves after interaction.
- Boot publishes profile sections into KV as `pdna:sections`, `pdna:affect_model`, `pdna:reinforcement_model`, `pdna:drive_thresholds`, `pdna:ddna_mutators`, and `pdna:wans`.
- `derive_ddna_modulators()` now presses v2 profile mutators into runtime gains such as reward, salience, boredom growth/relief, curiosity, expression, thought momentum, drawer persistence, safety strictness, and human-uplift bias.
- Reward/novelty pulse now reads active DDNA and affect-model decay values. `/acc 1-10` produces real variance instead of flattening high scores into the same pulse.
- Boredom now uses DDNA growth/relief/novelty multipliers, so a profile can become more novelty-hungry or calmer without rewriting the drive.
- Attention salience now uses DDNA salience/novelty gain and salience decay resistance.
- Thought momentum and thought drawer arbitration now use DDNA persistence/thought-completion/profile route hints.
- Textual pressure band now considers `affect:salience_state`, so reward/training salience can visibly move even when the attention controller is not the source.

## Main live keys

- `drive:ddna_modulators`
- `affect:reward_state`
- `affect:novelty_state`
- `affect:salience_state`
- `drive:boredom`
- `thought:momentum`
- `thought:turn:last_state`
- `pdna:wans`

## Expected visible behavior

A positive teaching pulse such as `/acc 5` or `/acc 10` should move:

- dopamine/reward upward quickly
- salience upward more slowly/firmly
- boredom downward if the interaction is useful or novel
- thought/trainer pressure when the correction/reinforcement binds

Repeated praise without new result is damped to avoid pure dopamine chasing.
