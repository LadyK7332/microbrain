# Pattern + Hypothesis Gears v1

Date: 2026-07-21

## Purpose

This patch inserts two foundational reasoning gears between `context/built` and outward release:

1. `PatternToolkit` detects reusable structure without deciding truth or speaking.
2. `HypothesisEngineNeuron` interprets the current statement, consults the rolling conversation scene and memory, predicts candidate outcomes, and treats silence as a valid action.

The new path is:

```text
percept/text
  -> conversation.scene update
  -> context/request
  -> context/built
  -> pattern analysis
  -> hypothesis/ready
  -> desire release-or-silence decision
  -> action selection
  -> reason/request
```

## Pattern toolkit

File:

```text
microbrain/patterns/pattern_toolkit.py
```

The toolkit is deliberately pure and non-speaking. It analyzes:

- statement kind
- local conversation continuity
- novelty
- contradiction candidates
- co-occurrence
- sequence/thread continuation
- memory recurrence
- consequence/risk salience
- response expectation
- uncertainty

It receives the current context frame plus optional memory evidence and returns a serializable `pattern.analysis.v1` packet.

## Hypothesis engine

File:

```text
microbrain/neurons/hypothesis_engine_neuron.py
```

The engine wakes for every meaningful `context/built` event, not only explicit questions.

It uses three working-memory depths:

```text
near field    = last six exact conversation turns
rolling field = conversation summary, active threads, claims, unresolved points
deep field    = expanded episodic, semantic, and mem-cell recall when justified
```

The current user turn is removed from the near field before comparison so it cannot falsely report perfect continuity with itself.

The engine emits:

```text
pattern/analysis
hypothesis/ready
```

The hypothesis object includes:

- possible interpretations
- candidate actions
- predicted outcome for each action
- recommended action
- explicit silence score
- response-demand estimate
- working/deep memory evidence summary
- expiration and ephemeral state

Hypotheses are not durable truth. They remain temporary until later observations and feedback can test them.

## Deep-pass gates

A deeper memory check is requested for:

- questions
- corrections or disagreement
- uncertain claims
- high uncertainty
- contradiction candidates
- high novelty
- consequence/risk terms
- unusually complex input

Ordinary status updates and conversational continuations normally stay on the cheaper working-memory pass.

## Release changes

`DesireTriggerNeuron` now subscribes to:

```text
hypothesis/ready
```

instead of directly subscribing to `context/built`.

Release pressure now includes:

- response demand
- predicted usefulness
- conversation continuity
- consequence risk
- recommended action
- memory support
- uncertainty/clarification need
- existing boredom, babble, and crisis pressure

Silence subtracts pressure and can veto release deliberately.

This distinguishes:

```text
old silence: no response route woke up
new silence: a response was considered and silence won
```

## Action selection

`ActionSelectorNeuron` now carries the selected hypothesis action into `reason/request`, including:

- `hypothesis_id`
- `selected_action`
- statement kind
- response demand
- uncertainty
- continuity
- memory mode

It also writes `hypothesis:pending_action`, which gives a later prediction-error/feedback neuron a stable hook for comparing expected and observed outcomes.

## Native response integration

`NativeResponderNeuron` receives the hypothesis packet and selected action. A statement judged response-worthy can now cross the native expression gate even without a question mark.

No canned acknowledgement lines were added. The hypothesis system decides whether and how a response is useful; learned syntax, memory, internal status, or another reasoning source must still construct the actual words.

## State keys

```text
pattern:last_analysis
hypothesis:last
hypothesis:last_correlation_id
hypothesis:history
hypothesis:last_release_decision
hypothesis:pending_action
desire:last_trigger
```

## Tests

Added:

```text
tests/test_pattern_hypothesis_gears.py
```

Coverage includes:

- rolling-window continuity without self-matching
- plain statement hypothesis and response recommendation
- deliberate silence for conversational closure
- correction-triggered deep memory lookup
- full hypothesis -> release -> action-selection flow

Validation on the uploaded 2026-07-21 source:

```text
python -m compileall -q microbrain
python -m pytest -q tests

24 passed
```
