# Pattern, Hypothesis, and Outcome Observer v2

## Purpose

Pattern recognition describes structure. The hypothesis engine proposes meaning
and predicts the likely utility of response, action, or silence. The outcome
observer compares that commitment with later evidence and returns a bounded
learning signal.

```text
participant/scene input
  -> rolling conversation + selective memory
  -> pattern analysis
  -> interpretations and action candidates
  -> action committed
  -> action executed or deliberate silence
  -> later observation
  -> outcome score + prediction error
  -> bounded action bias for similar future contexts
```

## Outcome evidence order

Strong evidence:
- explicit `/r` reinforcement
- signed `/acc` feedback metadata
- trainer correction

Moderate evidence:
- direct agreement, rejection, or correction in the next participant turn
- repeated question after an attempted answer
- continued same-thread follow-up

Weak evidence:
- topic continuation without explicit appraisal
- clean topic advance after deliberate closure silence

No evidence:
- no later participant response before expiry

No evidence never becomes automatic success or failure.

## Learned action buckets

The observer keeps small KV buckets keyed by:

```text
statement_kind|action
*|action
```

Each bucket tracks weighted observations, average score, confidence, positive /
negative / neutral counts, and the most recent reason. The hypothesis engine
uses exact context buckets more heavily than global action buckets and caps the
resulting candidate movement at +/-0.18.

This means learned history influences judgment without overpowering the current
scene, memory evidence, risk, uncertainty, or conversation context.
