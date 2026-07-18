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
        - "vision/percept_commit"

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

        if event.topic not in ("percept/vision", "vision/percept_commit", "vision/object_delta"):
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            return []

        # Raw vision frames/feature packets are now parser input, not durable memory.
        # The delta neuron emits vision/object_delta when a compact change exists.
        if event.topic in ("percept/vision", "vision/percept_commit"):
            await self._remember_last_raw_summary(event, ctx, payload)
            raw_hrm_enabled = bool(await ctx.get_kv("vision:raw_hrm_enabled", False))
            if not raw_hrm_enabled:
                return []

        vision_text, objects, memory_candidate = self._vision_text_from_event(event, payload)
        if not vision_text:
            return []

        await ctx.set_kv(
            "vision:last_summary",
            {
                "ts": time.time(),
                "text": vision_text,
                "objects": [str(o) for o in objects[:16]],
                "source": event.source,
                "source_topic": event.topic,
                "memory_candidate": memory_candidate,
            },
        )

        if event.topic == "vision/object_delta" and not memory_candidate:
            return []

        # Try to get HRM core
        hrm = await ctx.get_kv("hrm:core", None)
        if hrm is None:
            await ctx.log_warn(
                f"[{self.name}] HRM core not available; cannot store visual memory"
            )
            return []

        try:
            node = hrm.observe(
                vision_text,
                role="vision",
                meta={
                    "modality": "vision",
                    "source_topic": event.topic,
                    "memory_shape": "object_delta" if event.topic == "vision/object_delta" else "raw_summary",
                    "memory_candidate": memory_candidate,
                },
            )
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

    async def _remember_last_raw_summary(self, event: Event, ctx, payload: Dict[str, Any]) -> None:
        description = str(payload.get("description", "") or payload.get("text", "") or "").strip()
        objects = payload.get("objects", []) or []
        if not isinstance(objects, list):
            objects = [str(objects)]
        await ctx.set_kv(
            "vision:last_raw",
            {
                "ts": time.time(),
                "source_topic": event.topic,
                "description": description,
                "objects": [str(o) for o in objects[:16]],
                "source": event.source,
                "policy": "raw_vision_is_temporary_unless_vision_raw_hrm_enabled",
            },
        )

    def _vision_text_from_event(self, event: Event, payload: Dict[str, Any]) -> tuple[str, List[str], bool]:
        if event.topic == "vision/object_delta":
            memory_candidate = bool(payload.get("memory_candidate", False) or (event.meta or {}).get("memory_candidate", False))
            text = str(payload.get("text", "") or "").strip()
            objects: List[str] = []
            for delta in list(payload.get("deltas", []) or [])[:8]:
                if isinstance(delta, dict):
                    label = str(delta.get("label", "") or "").strip()
                    if label:
                        objects.append(label)
            if not text:
                descriptions = [
                    str(d.get("description", "") or "").strip()
                    for d in list(payload.get("deltas", []) or [])[:4]
                    if isinstance(d, dict)
                ]
                text = " ".join(d for d in descriptions if d).strip()
            return text, objects, memory_candidate

        if event.topic == "vision/percept_commit":
            description = str(payload.get("text", "") or "").strip()
            resolved_label = str(payload.get("resolved_label", "") or "").strip()
            objects = [resolved_label] if resolved_label else [str(payload.get("fallback_ref", "that thing") or "that thing")]
            return description, objects, True

        description = str(payload.get("description", "") or "").strip()
        raw_objects = payload.get("objects", []) or []
        if not isinstance(raw_objects, list):
            raw_objects = [str(raw_objects)]
        objects = [str(o) for o in raw_objects[:8]]
        text_parts: List[str] = []
        if description:
            text_parts.append(description)
        if objects:
            text_parts.append(f"(I recognized: {', '.join(objects)})")
        return " ".join(text_parts).strip(), objects, False


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/vision", "vision/percept_commit", "vision/object_delta"],
        output_topics=["reason/output", "act/speech"],
        priority=4,  # after raw percepts, before higher-level reasoning
    )
    yield VisionCortexNeuron(cfg)
