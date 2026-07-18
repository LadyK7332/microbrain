# Internal Status Speech Gate - 2026-06-26

This patch separates internal body/drive pressure from conversational speech.

## Problem

Power pressure could emit `speech/reason`, which then became `act/speech`, so a normal reply could be hijacked by lines such as:

```text
I need to charge soon.
```

That made the power drive behave like conversation instead of body telemetry.

## Patch shape

Power pressure now updates visible status by default:

```text
status> power: low at 63% | charge soon | mode=active
```

Unsolicited power speech is gated unless:

- `drive:power:allow_unsolicited_speech` is true, or
- urgency crosses `drive:power:critical_speech_threshold`, or
- the operator disables the gate with `drive:power:speech_gate_enabled = false`.

The normal responder can now answer internal-state questions from existing KV state. Examples:

```text
you> How are you?
Demi> I'm stable. power is 63% with elevated charge pressure. Stress is low, boredom is low, and curiosity is moderate.

you> What are your internal scores?
Demi> Internal status: power is 63% with elevated charge pressure; power urgency 0.61; stress 0.05; boredom 0.12; curiosity 0.44; externalize 0.30; mode active; maintenance stable.
```

## New KV defaults

```text
drive:power:speech_gate_enabled = true
drive:power:allow_unsolicited_speech = false
drive:power:critical_speech_threshold = 0.90
drive:power:status_cooldown_s = 120.0
drive:power:last_status_ts = 0.0
drive:power:last_status = {}
```

## Design rule

Internal pressure is not speech.
Internal pressure becomes speech only through permission, relevance, or emergency.
