from __future__ import annotations

from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class WhyExplainNeuron(BaseNeuron):
    """
    Causal explanation neuron.

    Listens on:
        - "introspect/why"

    Emits:
        - "reason/request"  (for LLMReasonerNeuron)

    It takes a small trace (last user input + last assistant reply) and asks
    the reasoning core to explain, in first person, why it responded that way.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        payload = event.payload

        # ---- safety check: no dict, no work ----
        if not isinstance(payload, dict):
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload for introspect/why",
                payload_type=str(type(payload)),
            )
            return []

        channel = str(payload.get("channel", "default"))
        source = str(payload.get("source", "user"))
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}

        last_user: str = str(payload.get("last_user", "") or "").strip()
        last_reply: str = str(payload.get("last_reply", "") or "").strip()

        # ---- nothing to explain = bail ----
        if not last_user and not last_reply:
            await ctx.log_debug(
                f"[{self.name}] No last_user / last_reply in payload; nothing to explain",
                channel=channel,
            )
            return []

        # ------------------------------
        # Build causal explanation prompt
        # ------------------------------
        prompt_lines: List[str] = []

        prompt_lines.append(
            "You are MicroBrain explaining your recent behavior in first person."
        )
        prompt_lines.append("")

        if last_user:
            prompt_lines.append("Most recent user message was:")
            prompt_lines.append(f"USER: {last_user}")
        else:
            prompt_lines.append("Most recent user message is unknown.")

        prompt_lines.append("")

        if last_reply:
            prompt_lines.append("Your most recent reply was:")
            prompt_lines.append(f"ASSISTANT: {last_reply}")
        else:
            prompt_lines.append("Your most recent reply is unknown.")

        prompt_lines.append("")
        prompt_lines.append(
            "Explain briefly, as 'I', why you responded that way. "
            "Refer to your internal components (neurons) and routing if helpful, "
            "for example mentioning input processing, routing, reflection, or reasoning."
        )
        prompt_lines.append(
            "Keep it concise, 3–6 sentences, and describe the reasoning chain "
            "in natural language."
        )

        prompt = "\n".join(prompt_lines)

        await ctx.log_debug(
            f"[{self.name}] Built causal explanation prompt",
            channel=channel,
            has_last_user=bool(last_user),
            has_last_reply=bool(last_reply),
        )

        # ------------------------------
        # Emit reason/request for LLMReasonerNeuron
        # ------------------------------
        reason_payload: Dict[str, Any] = {
            "text": prompt,
            "source": "system",
            "channel": channel,
            "raw_meta": {
                "mode": "why_explain",
                "original_source": source,
                "trace": {
                    "last_user": last_user,
                    "last_reply": last_reply,
                },
                "raw_meta": raw_meta,
            },
        }

        reason_event = Event(
            topic="reason/request",
            payload=reason_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"kind": "why_explain_request"},
        )

        # only outbound event: hand off to LLMReasonerNeuron
        return [reason_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="why_explain",
        subscribed_topics=["introspect/why"],
        output_topics=["reason/request"],
        priority=9,  # after status_introspect (10), before general responders
    )
    yield WhyExplainNeuron(cfg)
