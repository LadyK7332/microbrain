from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


class StatusIntrospectNeuron(BaseNeuron):
    
    """
    Introspection / status neuron.

    Listens on:
        - "introspect/status"

    Emits:
        - "ui/status" with structured/system status.

    It uses a reference to the Orchestrator (injected at build time) to
    inspect which neurons are loaded and which topics are covered.
    """

    def __init__(self, cfg: NeuronConfig, orchestrator: Optional[Orchestrator] = None):
        super().__init__(cfg)
        self._orch: Optional[Orchestrator] = orchestrator

    def set_orchestrator(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        channel = "cli"
        payload = event.payload

        if isinstance(payload, dict):
            channel = str(payload.get("channel", channel))

        # If we don't have an orchestrator reference, report a structured status.
        if self._orch is None:
            return [self._status(
                {"kind": "introspect_unavailable", "channel": channel},
                channel,
                event,
            )]

        # ------------------------------
        # 1) Gather neuron + topic info
        # ------------------------------
        neuron_objs = list(self._orch.neurons.values())
        neuron_count = len(neuron_objs)
        neuron_names: List[str] = sorted(n.name for n in neuron_objs)

        subscribed_topics: Set[str] = set()
        output_topics: Set[str] = set()

        for n in neuron_objs:
            subs = getattr(n, "subscribed_topics", []) or []
            outs = getattr(n, "output_topics", []) or []
            for t in subs:
                subscribed_topics.add(str(t))
            for t in outs:
                output_topics.add(str(t))

        # ------------------------------
        # 2) Build a human-readable summary
        # ------------------------------
        parts: List[str] = []
        parts.append("=== MicroBrain Status ===")
        parts.append(f"Neurons active: {neuron_count}")
        if neuron_names:
            parts.append("Neuron list:")
            for name in neuron_names:
                parts.append(f"  - {name}")

        if subscribed_topics:
            parts.append("")
            parts.append("Subscribed topics:")
            for t in sorted(subscribed_topics):
                parts.append(f"  - {t}")

        if output_topics:
            parts.append("")
            parts.append("Output topics:")
            for t in sorted(output_topics):
                parts.append(f"  - {t}")

        text = "\n".join(parts)

        await ctx.log_info(
            f"[{self.name}] Reported status",
            neuron_count=neuron_count,
            subscribed_topics=sorted(subscribed_topics),
            output_topics=sorted(output_topics),
        )

        events: List[Event] = []
        events.append(self._status(text, channel, event))

        # If this was a reflective/introspective command, also emit a
        # machine-readable report for the ReflectiveReasonerNeuron.
        cmd = ""
        if isinstance(payload, dict):
            cmd = str(payload.get("command", "")).lower()

        if cmd in ("reflect", "introspect"):
            report_payload: Dict[str, Any] = {
                "status_text": text,
                "channel": channel,
                "source": payload.get("source", "user") if isinstance(payload, dict) else "user",
                "command": cmd,
                "raw_meta": payload.get("raw_meta", {}) if isinstance(payload, dict) else {},
            }
            events.append(Event(
                topic="introspect/report_text",
                payload=report_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "introspect_report"},
            ))

        return events


    # ------------------------------
    # Helper to build status events
    # ------------------------------

    def _status(self, payload: Any, channel: str, event: Event) -> Event:
        if isinstance(payload, dict):
            data = dict(payload)
            data.setdefault("channel", channel)
        else:
            data = {"text": str(payload or ""), "channel": channel, "style": "system"}
        return Event(
            topic="ui/status",
            payload=data,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"control": True, "kind": "introspect_status"},
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name="status_introspect",
        subscribed_topics=["introspect/status"],
        output_topics=["ui/status", "introspect/report_text"],
        priority=10,  # runs before most general responders
    )
    neuron = StatusIntrospectNeuron(cfg, orchestrator=orchestrator)
    yield neuron