# scene.expectation.v1

Core rule:

```text
scene.obj + time = scene.exp
```

`scene.exp` is an ephemeral expectation object. It is built from the previous scene plus the current time/place context and is used only to compare expected state against observed state.

It should not be saved as durable memory by default.

Flow:

```text
scene.obj at T0
leave / time passes
scene.exp at T1
observed scene.obj at T1
expected vs observed delta
```

If the delta is small, expectation is confirmed.

If the delta is meaningful, MB emits a `thought/internal` question such as:

```text
thought> Expected scene changed here; Why did X appear in this scene?
```

If the delta is important enough, MB parks a tiny unresolved `question.object` for the day under:

```text
memdir/questions/unresolved_questions.jsonl
```

Parked questions are not full memory dumps. They are small unresolved change records that can later be resolved by observation or user explanation.

Tiny law:

```text
Memory stores scenes.
Time creates expectations.
Observation corrects them.
Delta creates questions.
```
