from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class CuriosityDriveNeuron(BaseNeuron):
    """
    Curiosity drive built on top of boredom + HRM + PDNA.

    v1 behavior:
      - Reads drive:boredom from KV.
      - When boredom is high enough and enough "ticks" have passed,
        emits an internal reason/request event (an "internal thought").
      - Uses optional HRM/PDNA context to shape the curiosity prompt.
      - Designed to be safe when vision/voice are not yet online.
        (It checks sensor flags but treats missing keys as False.)

    This neuron does NOT listen for user prompts; it only piggybacks on
    normal bus activity (percepts / speech) to keep time.
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

        # Only treat certain topics as "time ticks" for curiosity
        if event.topic not in ("percept/text", "percept/vision", "act/speech"):
            return []

        # Read boredom drive state
        boredom = await ctx.get_kv("drive:boredom", None)
        if not isinstance(boredom, dict):
            # No boredom drive yet, nothing to do
            return []

        boredom_level = float(boredom.get("level", 0.0) or 0.0)
        boredom_high = bool(boredom.get("high", False))

        # If boredom is low, curiosity stays mostly idle
        if boredom_level < 0.4:
            # We'll still increment our internal tick counter below,
            # but don't fire any internal thoughts.
            pass

        # Optional context: PDNA, HRM, sensors (vision/voice)
        pdna_profile = await ctx.get_kv("pdna:profile", None)
        hrm = await ctx.get_kv("hrm:core", None)
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)

        # Sensor flags (to be set later by vision/voice wiring).
        # For now, they default to False and never crash.
        vision_online = bool(await ctx.get_kv("sensors:vision_online", False))
        voice_online = bool(await ctx.get_kv("sensors:voice_online", False))

        # Load curiosity internal state
        state = await self.load_state(
            ctx,
            "curiosity_state",
            default={
                "ticks_since_last": 0,
                "internal_thought_id": 0,
            },
        )
        ticks_since_last = int(state.get("ticks_since_last", 0) or 0)
        internal_thought_id = int(state.get("internal_thought_id", 0) or 0)

        # Advance our simple "time"
        ticks_since_last += 1

        # Map boredom to a firing interval:
        # - low boredom => very rare curiosity
        # - high boredom => more frequent internal thoughts
        #
        # Interval in "ticks" (events). We clamp around [3, 20].
        if boredom_level <= 0.4:
            target_interval = 20
        elif boredom_level >= 0.9:
            target_interval = 3
        else:
            # linear interpolation between 20 and 3
            target_interval = int(
                20 - (boredom_level - 0.4) * (17 / 0.5)
            )
            if target_interval < 3:
                target_interval = 3
            elif target_interval > 20:
                target_interval = 20

        # If we haven't waited long enough, just persist state and exit
        if ticks_since_last < target_interval:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []

        # We've waited long enough AND boredom is at least moderate.
        # If boredom isn't actually high, we can decide to do nothing.
        if boredom_level < 0.4:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []

        # Reset tick counter and increment internal thought id
        ticks_since_last = 0
        internal_thought_id += 1

        # Construct an internal curiosity prompt.
        # We adapt slightly based on context:
        #  - If we have HRM + last_idx, we can phrase it as "thinking more about that".
        #  - If vision is online, we can phrase it as visual wondering.
        #  - Otherwise, use a generic introspective curiosity.
        curiosity_text = None

        # Attempt HRM-based curiosity first
        if hrm is not None and isinstance(hrm_last_idx, int):
            try:
                node = hrm.get_node(hrm_last_idx)
            except Exception:
                node = None

            node_text = ""
            if node is not None:
                node_text = getattr(node, "text", "") or ""
                node_text = str(node_text).strip()

            if node_text:
                curiosity_text = (
                    f"I find myself thinking more about this: \"{node_text}\". "
                    "What else can I notice or understand about it?"
                )

        # If we have vision online, we can phrase curiosity visually
        if curiosity_text is None and vision_online:
            curiosity_text = (
                "I'm looking around and wondering about the things I see. "
                "Is there anything here I should focus on or try to understand better?"
            )

        # Generic introspective curiosity fallback
        if curiosity_text is None:
            curiosity_text = (
                "My mind is wandering because things feel repetitive. "
                "Is there anything interesting, useful, or meaningful I can explore right now?"
            )

        # We treat this as an "internal" reason request.
        # This flows into LLMReasonerNeuron like any external prompt,
        # but source/channel mark it as self-initiated.
        meta = {
            "kind": "curiosity_probe",
            "boredom_level": boredom_level,
            "boredom_high": boredom_high,
            "internal_thought_id": internal_thought_id,
            "vision_online": vision_online,
            "voice_online": voice_online,
        }

        internal_event = Event(
            topic="reason/request",
            payload={
                "text": curiosity_text,
                "source": "internal",
                "channel": "thought",
                "raw_meta": meta,
            },
            source=self.name,
            correlation_id=event.correlation_id,
        )

        # Persist curiosity state
        state.update(
            {
                "ticks_since_last": ticks_since_last,
                "internal_thought_id": internal_thought_id,
            }
        )
        await self.save_state(ctx, "curiosity_state", state)

        self.debug(
            "curiosity_fired",
            boredom_level=boredom_level,
            target_interval=target_interval,
            internal_thought_id=internal_thought_id,
            vision_online=vision_online,
            voice_online=voice_online,
        )

        # Emit the internal thought as a reason/request
        return [internal_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "percept/vision",
            "act/speech",
        ],
        output_topics=["reason/request"],
        priority=-9,  # after boredom, before more "outward" behaviors if needed
    )
    yield CuriosityDriveNeuron(cfg)
