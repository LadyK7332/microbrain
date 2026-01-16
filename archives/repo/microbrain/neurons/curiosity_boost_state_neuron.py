from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class CuriosityBoostStateNeuron(BaseNeuron):
    """
    Persists and decays curiosity:boost.

    Inputs:
      - curiosity/adjust payload {"boost": float, "pause_s": float, "reason": str, ...}
      - clock/tick payload {"ts": ...}

    KV keys written:
      - curiosity:boost (float 0..1)
      - curiosity:cooldown_until (timestamp float)
      - curiosity:last_feedback_ts (timestamp float)
      - curiosity:last_feedback_score (float)  # negative magnitude
      - curiosity:last_feedback_reason (str)
      - curiosity:boost_last_update_ts (timestamp float)
    """

    BOOST_MAX = 1.0
    # Linear decay per second. 0.01 => 1.0 drains to 0 in ~100 seconds.
    BOOST_DECAY_PER_S = 0.01

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic == "curiosity/adjust":
            payload = event.payload or {}
            if not isinstance(payload, dict):
                return []

            now = time.time()

            boost_add = float(payload.get("boost", 0.0) or 0.0)
            pause_s = float(payload.get("pause_s", 0.0) or 0.0)
            reason = str(payload.get("reason", "") or "")

            current_boost = float(await ctx.get_kv("curiosity:boost", 0.0) or 0.0)
            new_boost = max(0.0, min(self.BOOST_MAX, current_boost + boost_add))

            current_cd = float(await ctx.get_kv("curiosity:cooldown_until", 0.0) or 0.0)
            new_cd = max(current_cd, now + pause_s) if pause_s > 0.0 else current_cd

            await ctx.set_kv("curiosity:boost", new_boost)
            await ctx.set_kv("curiosity:cooldown_until", new_cd)
            await ctx.set_kv("curiosity:last_feedback_ts", now)
            await ctx.set_kv("curiosity:last_feedback_score", -abs(boost_add))
            await ctx.set_kv("curiosity:last_feedback_reason", reason)
            await ctx.set_kv("curiosity:boost_last_update_ts", now)

            self.debug(
                "curiosity_boost_set",
                boost_add=boost_add,
                boost=new_boost,
                pause_s=pause_s,
                cooldown_until=new_cd,
                reason=reason,
            )
            return []

        if event.topic == "clock/tick":
            now = time.time()

            boost = float(await ctx.get_kv("curiosity:boost", 0.0) or 0.0)
            last_ts = float(await ctx.get_kv("curiosity:boost_last_update_ts", now) or now)

            dt = now - last_ts
            if dt < 0.0:
                dt = 0.0

            if boost > 0.0 and dt > 0.0:
                decayed = boost - (self.BOOST_DECAY_PER_S * dt)
                if decayed < 0.0:
                    decayed = 0.0

                # Only write when it changes, keeps KV quieter
                if decayed != boost:
                    await ctx.set_kv("curiosity:boost", decayed)
                    self.debug(
                        "curiosity_boost_decayed",
                        before=boost,
                        after=decayed,
                        dt_s=dt,
                        rate_per_s=self.BOOST_DECAY_PER_S,
                    )

            await ctx.set_kv("curiosity:boost_last_update_ts", now)
            return []

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "curiosity/adjust",
            "clock/tick",
        ],
        output_topics=[],
        priority=5,  # early-ish (after feedback, before most drives is fine)
    )
    yield CuriosityBoostStateNeuron(cfg)
