from __future__ import annotations

from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

# Neuron name = this file's basename without .py
NEURON_NAME = Path(__file__).stem


class BoredomDriveNeuron(BaseNeuron):
    """
    Tracks a simple "boredom" drive based on repetition and low novelty.

    Heuristic v1:
      - If the HRM keeps reporting the same last_idx repeatedly, boredom rises.
      - Novel experiences (change in last_idx) reduce boredom.
      - PDNA traits (energy, playfulness) modulate how quickly boredom rises.
      - The current boredom state is exposed via KV as "drive:boredom".
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

        # We trigger on percepts and speech ticks, but ignore other topics
        if event.topic not in ("percept/text", "percept/vision", "act/speech"):
            return []

        # Fetch HRM's last node index (set by hrm_observer_neuron)
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)

        # Fetch PDNA profile (for energy / playfulness / introspection)
        pdna_profile = await ctx.get_kv("pdna:profile", None)

        # Load previous boredom state
        state = await self.load_state(
            ctx,
            "boredom_state",
            default={"prev_idx": None, "repetitions": 0, "level": 0.0},
        )

        prev_idx = state.get("prev_idx", None)
        repetitions = int(state.get("repetitions", 0) or 0)
        level = float(state.get("level", 0.0) or 0.0)

        # Natural boredom decay each tick (prevents runaway)
        decay = 0.98
        level *= decay

        # If HRM has not yet produced any nodes, we can't measure novelty yet.
        if hrm_last_idx is None:
            # Persist decay-only state and publish drive
            state.update(
                {
                    "prev_idx": prev_idx,
                    "repetitions": repetitions,
                    "level": max(0.0, min(1.0, level)),
                }
            )
            await self.save_state(ctx, "boredom_state", state)

            boredom_payload = {
                "active": level >= 0.6,
                "high": level >= 0.85,
                "level": level,
                "repetitions": repetitions,
            }
            await ctx.set_kv("drive:boredom", boredom_payload)

            self.debug("boredom_updated_no_hrm", **boredom_payload)
            return []

        # Compute novelty vs repetition
        if prev_idx is None:
            # First time seeing a valid HRM index
            prev_idx = hrm_last_idx
        else:
            if hrm_last_idx == prev_idx:
                # Same conceptual region again -> increase boredom
                repetitions += 1

                # Base increment
                base_inc = 0.02

                # PDNA modulation: energetic + playful minds get bored faster
                energy = (
                    getattr(pdna_profile, "energy", 0.5)
                    if pdna_profile is not None
                    else 0.5
                )
                playfulness = (
                    getattr(pdna_profile, "playfulness", 0.5)
                    if pdna_profile is not None
                    else 0.5
                )

                pdna_factor = 0.5 + energy * 0.3 + playfulness * 0.3
                repetition_factor = 1.0 + min(repetitions, 20) / 20.0

                level += base_inc * pdna_factor * repetition_factor
            else:
                # New conceptual node -> boredom drops more aggressively
                repetitions = max(0, repetitions - 1)
                prev_idx = hrm_last_idx

                # Novelty relieves boredom
                level *= 0.8

        # Clamp boredom level to [0.0, 1.0]
        if level < 0.0:
            level = 0.0
        elif level > 1.0:
            level = 1.0

        # Persist state
        state.update(
            {
                "prev_idx": prev_idx,
                "repetitions": repetitions,
                "level": level,
            }
        )
        await self.save_state(ctx, "boredom_state", state)

        # Publish boredom drive to KV so other neurons can use it
        boredom_payload = {
            "active": level >= 0.6,
            "high": level >= 0.85,
            "level": level,
            "repetitions": repetitions,
        }
        await ctx.set_kv("drive:boredom", boredom_payload)

        self.debug("boredom_updated", **boredom_payload)

        # This neuron only updates internal drive state; no outbound events for now.
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "percept/vision",
            "act/speech",
        ],
        output_topics=[],
        # Run after PDNA/HRM observers (which often use small negative priorities)
        priority=-10,
    )
    yield BoredomDriveNeuron(cfg)
