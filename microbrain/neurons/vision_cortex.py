from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


NEURON_NAME = Path(__file__).stem


class VisionCortexNeuron(BaseNeuron):
    """
    Vision cortex adapter.

    Listens on:
        - "percept/vision"

    Behavior:
        - Takes structured vision events (description + objects).
        - Builds a compact natural-language summary suitable for HRM.
        - Writes a new HRM node with role="vision" so visual experience
          becomes part of episodic memory and can be recollected later.
        - Optionally emits a small text echo for debugging / tracing.
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

        if event.topic != "percept/vision":
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            return []

        description = str(payload.get("description", "") or "").strip()
        objects = payload.get("objects", []) or []
        if not isinstance(objects, list):
            objects = [str(objects)]

        # If we literally have nothing, don't create a ghost memory
        if not description and not objects:
            await ctx.log_debug(
                f"[{self.name}] Empty vision payload; nothing to store",
                topic=event.topic,
            )
            return []

        # Compose a compact text for HRM
        text_parts: List[str] = []
        if description:
            text_parts.append(description)
        if objects:
            obj_list = ", ".join(str(o) for o in objects[:8])
            text_parts.append(f"(I recognized: {obj_list})")

        vision_text = " ".join(text_parts).strip()

        if not vision_text:
            return []

        # Try to get HRM core
        hrm = await ctx.get_kv("hrm:core", None)
        if hrm is None:
            await ctx.log_warn(
                f"[{self.name}] HRM core not available; cannot store visual memory"
            )
            return []

        try:
            node = hrm.observe(vision_text, role="vision", meta={"modality": "vision"})
        except Exception as exc:
            await ctx.log_error(
                f"[{self.name}] Error writing visual memory to HRM",
                exception=str(exc),
            )
            return []

        await ctx.log_debug(
            f"[{self.name}] Stored visual memory in HRM",
            node_idx=getattr(node, "idx", None),
            text_preview=vision_text[:80],
        )

        await ctx.set_kv(
            "vision:last_summary",
            {
                "ts": time.time(),
                "text": vision_text,
                "objects": [str(o) for o in objects[:16]],
                "source": event.source,
            },
        )

        emit_internal = bool(await ctx.get_kv("vision:emit_internal_notes", True))
        speak_observations = bool(await ctx.get_kv("vision:speak_observations", False))

        out: List[Event] = []
        if emit_internal:
            out.append(
                Event(
                    topic="reason/output",
                    payload={"text": f"visual note: {vision_text}"},
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"channel": "thought", "kind": "vision_note", "lobe": "vision"},
                )
            )
        if speak_observations:
            out.append(
                Event(
                    topic="act/speech",
                    payload={
                        "text": f"(Noted a visual scene: {vision_text})",
                        "channel": "cli",
                        "style": "system",
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/vision"],
        output_topics=["reason/output", "act/speech"],
        priority=4,  # after raw percepts, before higher-level reasoning
    )
    yield VisionCortexNeuron(cfg)
