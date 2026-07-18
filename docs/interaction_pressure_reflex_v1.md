# Interaction Pressure Reflex v1

Input should create short-lived interaction pressure.

A normal external `percept/text` now seeds `drive:interaction:last_input_stimulus` before the slower initiative layer has to catch up. The interaction release vector can use that stimulus immediately to generate a response motive.

Control-plane traffic, command confirmations, command errors, thought events, and non-cognitive status traffic do not create interaction pressure.

Flow:

```text
user percept/text
-> input stimulus
-> interaction pressure
-> thought/internal pressure note
-> drive/interaction_request
-> speech/reason
-> act/speech
-> event/relief/interaction
```

Behavior law:

```text
Input creates interaction pressure.
Response resolves interaction pressure.
Repeated identical input still obeys cooldown/novelty gates.
New input bypasses the old-signal cooldown.
```
