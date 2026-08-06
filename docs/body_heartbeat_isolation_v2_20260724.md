# MicroBrain Body Heartbeat Isolation v2

Date: 2026-07-24

## Purpose

The body heartbeat is MicroBrain's scheduling baseline, not a thought, memory,
semantic observation, or reinforcement signal.  The canonical body pacemaker is
20 ticks per second (20 TPS), one nominal opportunity every 50 ms.

The design rule is:

> The heartbeat stays in the chest.  Organs may use it; cognition only receives
> meaningful results derived from organ work.

## Timing contract

- Canonical topic: `body/heartbeat`
- Nominal cadence: 20 Hz
- Nominal interval: 0.05 seconds
- Tick number is a scheduling coordinate only.
- Monotonic timestamps / measured delta are elapsed-time truth.
- Host stalls do not cause replay/catch-up storms.
- Missed opportunities are summarized as telemetry and discarded.
- Historical `clock/tick` input/subscriptions are normalized to the canonical
  topic.  They do not cause a second emitted pulse.

## Stream separation

The orchestrator owns two buses:

1. `bus`: meaningful perceptual/cognitive/action events. Attention and policy
   operate here.
2. `body_bus`: raw heartbeat and derived body-service opportunities. It bypasses
   attention, policy, memory telemetry taps, and cognition-wide wildcard taps.

A body-side handler may derive a meaningful event.  That result is routed onto
`bus`; raw infrastructure remains on `body_bus`.

## Infrastructure invariants

Heartbeat and body-service events carry infrastructure classification and are:

- non-semantic
- not memory eligible
- not reinforcement eligible
- hidden from normal UI event history
- excluded from normal neuron activation history
- excluded from Hebbian reinforcement/decay caused merely by cadence
- independent of semantic neuron cooldowns
- forbidden from propagating heartbeat correlation IDs into meaningful events

Memory filters reject all body infrastructure from long-term, trace, HRM, and
pattern storage.

## Organ service cadence

Only the body adrenaline/cadence scheduler subscribes to raw `body/heartbeat`.
It derives target-specific infrastructure opportunities:

`body/service/<target>`

Organs subscribe to the service target appropriate to their physiology rather
than to the raw pacemaker.  The scheduler emits only targets that currently have
live subscribers.

Default divisors (number of 20-TPS heartbeat ticks per service opportunity):

| Target | Normal | Alert | Emergency |
| --- | ---: | ---: | ---: |
| cognition | 1 | 1 | 1 |
| affect | 10 | 4 | 2 |
| curiosity | 10 | 20 | 100 |
| vision | 4 | 2 | 1 |
| gaze | 2 | 1 | 1 |
| touch | 2 | 1 | 1 |
| proprioception | 2 | 1 | 1 |
| motor_watch | 2 | 1 | 1 |
| hazard | 4 | 2 | 1 |
| capability | 4 | 2 | 1 |
| outcome | 10 | 4 | 2 |
| ipc | 4 | 4 | 2 |
| evidence | 4 | 2 | 1 |
| power | 20 | 10 | 4 |
| memory | 20 | 40 | 100 |
| maintenance | 20 | 40 | 100 |

These are engineering defaults, not biological claims.  The heartbeat remains
20 TPS in every arousal mode.  Alert/emergency reallocates service opportunity;
it does not accelerate the pacemaker.

An organ may still run a faster or slower private hardware/control loop.  These
body-service divisors govern when the organ gets a MicroBrain coordination/report
opportunity, not the hardware sample rate.

## Synthetic adrenaline

Meaningful danger evidence can switch body arousal:

`normal -> alert -> emergency`

Relevant organs receive denser service opportunities while background work is
reduced.  Emergency uses a hold interval and decays through alert before normal
to prevent rapid oscillation.

Arousal transitions emit `body/arousal_state` as meaningful state changes.
Repeated hazard evidence that merely extends the same mode remains silent.

## Cognition at full body tick

Cognition receives a `body/service/cognition` opportunity every body tick.  This
is still infrastructure, not a thought.  Current cognition-service consumers use
it for private housekeeping such as thought momentum decay, attention timing,
and drawer expiry/recheck.

The service pulse itself must not emit a thought merely because it happened.
For example:

- thought momentum decays in KV/RAM and emits no `thought/momentum` pulse
- thought-turn drawer housekeeping does not overwrite `thought:turn:last_state`
- capability initialization/refresh is private until a real capability change
  occurs

## Dashboard

Window 2 receives heartbeat health as an instrument rather than scrolling trace:

- actual TPS EMA
- jitter
- missed opportunity count
- stale/alive state
- current arousal mode

The dashboard event tap observes only the meaningful bus.  Raw heartbeat and
service traffic is physically absent from its normal event firehose.

## Compatibility / migration rule

`clock/tick` remains accepted only as a compatibility alias.  Runtime routing
normalizes it to `body/heartbeat` before dispatch.  New code must subscribe to
`body/service/<target>` or, for the body cadence scheduler only,
`body/heartbeat`.

A repository audit for this version should show exactly one neuron subscribed to
raw `body/heartbeat`: `body_adrenaline_scheduler_neuron`.
