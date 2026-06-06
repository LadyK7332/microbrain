# Thought Momentum v1

Thought momentum is a short-lived internal state organ. It keeps active intent vectors alive across turns so MB does not collapse every event into a one-shot reflex.

Core rule:

```text
event -> vector pressure -> decay over time -> resolve / vent / override
```

Momentum is not normal speech, not command output, and not memory by default. It is emitted on `thought/momentum` with non-cognitive metadata and stored in KV under:

```text
thought:momentum
thought:momentum:active_vectors
thought:momentum:last_update_ts
```

Current vector examples:

- `understand_user` from questions or requests
- `curiosity` from questions or curiosity boost
- `social_continuity` from greetings
- `resolve_thread` from continuity/problem references
- `await_result` after MB speaks
- `seek_novelty` from boredom
- `seek_social_contact` from social drive
- `social_experiment` from combined boredom/social pressure

Context builder includes `thought_momentum` in built context. Action selection and native response shaping use momentum as a small bias, not a command.

Tiny law:

**A thought that ends instantly is a reflex. Momentum lets it become a thread.**
