from __future__ import annotations

from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class ReflectiveReasonerNeuron(BaseNeuron):
    """
    Reflective reasoning neuron.

    Listens on:
        - "introspect/report_text"  (status report from StatusIntrospectNeuron)

    Emits:
        - "reason/request"          (prompt to LLMReasonerNeuron)

    This lets MicroBrain take a structured self-status and ask its own
    reasoning core to reflect on it in natural language.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        payload = event.payload
        if not isinstance(payload, dict) or "status_text" not in payload:
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload for introspect/report_text",
                payload_type=str(type(payload)),
            )
            return []

        status_text: str = str(payload.get("status_text", "")).strip()
        if not status_text:
            await ctx.log_debug(
                f"[{self.name}] Empty status_text, ignoring",
                topic=event.topic,
            )
            return []

        source = str(payload.get("source", "user"))
        channel = str(payload.get("channel", "default"))
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        command = str(payload.get("command", ""))

        # ------------------------------
        # Build reflection prompt
        # ------------------------------
        prompt_lines: List[str] = []

        prompt_lines.append(
            "You are MicroBrain reflecting on your own internal state."
        )
        prompt_lines.append("")
        prompt_lines.append("Here is a status report about your current brain:")
        prompt_lines.append("")
        prompt_lines.append(status_text)
        prompt_lines.append("")
        prompt_lines.append(
            "Based on this report, briefly explain the following in first person:"
        )
        prompt_lines.append("- What components (neurons) I currently have.")
        prompt_lines.append("- What roles they play in my thinking and behavior.")
        prompt_lines.append("- What I seem to be capable of right now.")
        prompt_lines.append("")
        prompt_lines.append(
            "Keep the answer concise (3–6 sentences), speak as 'I', "
            "and avoid restating the status report verbatim."
        )

        prompt = "\n".join(prompt_lines)

        await ctx.log_debug(
            f"[{self.name}] Built reflection prompt",
            channel=channel,
            command=command,
        )

        # ------------------------------
        # Emit reason/request for LLMReasonerNeuron
        # ------------------------------
        reason_payload = {
            "text": prompt,
            "source": "system",
            "channel": channel,
            "raw_meta": {
                "mode": "reflection",
                "original_source": source,
                "command": command,
                "raw_meta": raw_meta,
            },
        }

        reason_event = Event(
            topic="reason/request",
            payload=reason_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"kind": "reflection_request"},
        )

        return [reason_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="reflective_reasoner",
        subscribed_topics=["introspect/report_text"],
        output_topics=["reason/request"],
        priority=8,  # after status_introspect (10), before generic stuff
    )
    yield ReflectiveReasonerNeuron(cfg)