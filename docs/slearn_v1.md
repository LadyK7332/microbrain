# Structured Learning Sheets (`/slearn`) v1

`/slearn` is a control-plane curriculum ingest path for the CAPS teaching grammar.
It lets the operator prepare repeatable day-to-day speech sheets without spending
hours typing every rule through Textual.

## Rule shape

One rule per line:

```text
IF USER says moin THEN CLASSIFY social_greeting, warmth, friendly AND REPLY good morning
IF USER says thanks THEN CLASSIFY gratitude, warmth, friendly AND REPLY you're welcome
IF USER says stop THEN CLASSIFY boundary, user_serious AND NOT REPLY playful
```

Lines beginning with `#` or `//` are ignored. The grammar keywords must be visibly
CAPS so normal prose cannot become behavior rules by accident.

## Commands

```text
/slearn status
/slearn on
/slearn off
/slearn next
/slearn dir <folder>
/slearn weight <1-5>
/slearn template
```

By default sheets live in:

```text
Z:\memory\slearn_dir
```

Completed files move to:

```text
Z:\memory\slearn_dir\ready
```

## Safety split

Normal `/read` stores low-trust sensory text.

`/slearn` does not store the document as conversation. It extracts CAPS rules and
emits `control/slearn` events. `syntax_learning_neuron.py` parses those events and
writes structured learned rules into memory.

This keeps curriculum ingestion separate from normal reading and ordinary user
conversation.
