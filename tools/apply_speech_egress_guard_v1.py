from __future__ import annotations

"""Apply Speech Egress Guard v1 to the local MicroBrain repo.

Run from repo root:
    python tools\apply_speech_egress_guard_v1.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "microbrain" / "orchestrator" / "orchestrator.py"

IMPORT_LINE = "from microbrain.speech_egress_guard import guard_speech_event, observe_speech_context\n"
IMPORT_ANCHOR = "from microbrain.utils.heartbeat_stream import (\n"
QUEUE_ANCHOR = """    def _queue_event(self, event: Event) -> None:\n        \"\"\"Route events to body or cognitive queue from their event class.\"\"\"\n        if not isinstance(event, Event):\n            return\n        canonicalize_event_in_place(event)\n"""
QUEUE_REPLACEMENT = """    def _queue_event(self, event: Event) -> None:\n        \"\"\"Route events to body or cognitive queue from their event class.\n\n        Final speech passes through the speech egress guard here because this is\n        the last shared mouth boundary before any ``act/speech`` event reaches\n        subscribers/UI.  Per-neuron guards are useful, but one-word fragments can\n        leak from other routes unless the final queue boundary repairs them.\n        \"\"\"\n        if not isinstance(event, Event):\n            return\n        canonicalize_event_in_place(event)\n        observe_speech_context(event, self.kv_store)\n        if event.topic == \"act/speech\":\n            guarded = guard_speech_event(event, self.kv_store)\n            if guarded is None:\n                return\n            event = guarded\n"""


def main() -> None:
    if not ORCH.exists():
        raise SystemExit(f"Missing orchestrator file: {ORCH}")
    text = ORCH.read_text(encoding="utf-8")
    original = text

    if IMPORT_LINE not in text:
        if IMPORT_ANCHOR not in text:
            raise SystemExit("Could not find heartbeat import anchor in orchestrator.py")
        text = text.replace(IMPORT_ANCHOR, IMPORT_LINE + IMPORT_ANCHOR, 1)

    if "guard_speech_event(event, self.kv_store)" not in text:
        if QUEUE_ANCHOR not in text:
            raise SystemExit("Could not find _queue_event anchor in orchestrator.py")
        text = text.replace(QUEUE_ANCHOR, QUEUE_REPLACEMENT, 1)

    if text != original:
        ORCH.write_text(text, encoding="utf-8")
        print("Applied Speech Egress Guard v1 to microbrain/orchestrator/orchestrator.py")
    else:
        print("Speech Egress Guard v1 already applied")


if __name__ == "__main__":
    main()
