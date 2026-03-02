from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class MouthSelfCheckNeuron(BaseNeuron):
    """
    Compares what the Mouth sidecar reports speaking vs what MB last intended to speak.

    Ingests:
      - topic: act/spoken
        payload: {id, text, ok, error, duration_s, expected_sha1}

    Writes KV:
      - mouth:last_spoken
      - mouth:last_mismatch (only if mismatch)

    Optional behavior:
      - If KV 'mouth:self_check_speak' is True, emits a small warning via act/speech.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "act/spoken":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        spoken_text = str(payload.get("text", "") or "").strip()
        ok = bool(payload.get("ok", True))
        expected_sha1 = str(payload.get("expected_sha1", "") or "").strip()

        last = await ctx.get_kv("mouth:last_intended", {}) or {}
        if not isinstance(last, dict):
            last = {}

        mismatch = False
        last_sha = str(last.get("expected_sha1", "") or "").strip()
        if expected_sha1 and last_sha:
            mismatch = expected_sha1 != last_sha
        else:
            intended_text = str(last.get("text", "") or "").strip()
            mismatch = bool(intended_text and spoken_text and intended_text != spoken_text)

        await ctx.set_kv(
            "mouth:last_spoken",
            {
                "id": payload.get("id"),
                "text": spoken_text,
                "ok": ok,
                "mismatch": mismatch,
                "duration_s": payload.get("duration_s"),
            },
        )

        if mismatch:
            await ctx.set_kv("mouth:last_mismatch", {"intended": last, "spoken": payload})

            speak_on_mismatch = bool(await ctx.get_kv("mouth:self_check_speak", False))
            if speak_on_mismatch:
                return [
                    Event(
                        topic="act/speech",
                        payload={
                            "text": "[Self-check] Mouth output mismatch detected. Logging details.",
                            "channel": "repl",
                            "style": "system",
                        },
                        source=NEURON_NAME,
                        correlation_id=event.correlation_id,
                        meta={"kind": "mouth_self_check"},
                    )
                ]

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["act/spoken"],
        output_topics=["act/speech"],
        priority=2,
        cooldown_sec=0.0,
    )
    yield MouthSelfCheckNeuron(cfg)
