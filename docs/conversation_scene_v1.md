# conversation.scene v1

Conversation is treated as a verbal scene, not a separate memory system.

`conversation.scene` is a rolling RAM/KV drawer that tracks current topic, active threads, recent claims, unresolved questions, and recent user/assistant turns. It is ephemeral by default and should not be dumped to durable memory unless another organ promotes a specific lesson.

Core rule:

```text
Conversation is a scene made of language.
Context is scene continuity.
```

Runtime keys:

```text
conversation:scene
conversation:current_scene
conversation:summary
```

Events:

```text
conversation/scene
```

The scene listens to:

```text
percept/text
act/speech
thought/internal
question/unresolved
```

Purpose:

- keep dialogue from drifting into unrelated fallback replies
- preserve active verbal context for learning
- let conversation follow the same scene/expectation architecture as vision/audio/touch
- keep the active drawer in RAM/KV until something is important enough to promote

Memory rule:

```text
Conversation updates the active scene first.
Only promoted lessons become durable memory.
```
