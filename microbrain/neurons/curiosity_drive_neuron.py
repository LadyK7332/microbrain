from __future__ import annotations

import time
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

    This neuron does NOT listen for user prompts. It keeps time primarily
    via clock/tick, and will only emit internal thoughts when attention gates allow it.
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
        if event.topic not in ("clock/tick", "percept/text", "percept/vision", "act/speech"):
            return []

        # Load curiosity internal state (persists across runs)
        state = await self.load_state(
            ctx,
            "curiosity_state",
            default={
                "ticks_since_last": 0,
                "internal_thought_id": 0,
                "last_fire_ts": 0.0,
            },
        )
        ticks_since_last = int(state.get("ticks_since_last", 0) or 0)
        internal_thought_id = int(state.get("internal_thought_id", 0) or 0)
        last_fire_ts = float(state.get("last_fire_ts", 0.0) or 0.0)

        # Advance our simple "time"
        ticks_since_last += 1

        now = time.time()
        boost = float(await ctx.get_kv("curiosity:boost", 0.0) or 0.0)
        feedback_cooldown_until = float(
            await ctx.get_kv("curiosity:cooldown_until", 0.0) or 0.0
        )
        in_feedback_cooldown = now < feedback_cooldown_until

        # Read boredom drive state (optional; defaults to "not bored")
        boredom = await ctx.get_kv("drive:boredom", None)
        boredom_level = 0.0
        boredom_high = False
        boredom_active = False
        if isinstance(boredom, dict):
            boredom_level = float(boredom.get("level", 0.0) or 0.0)
            boredom_high = bool(boredom.get("high", False))
            boredom_active = bool(boredom.get("active", False))

        # Gate internal curiosity/babble:
        #  - when boredom is ACTIVE, OR when feedback boost is present
        #  - and the attention controller allows internal speech (no recent external stimulus)
        #  - and we're not in a feedback pause window
        allow_babble = bool(await ctx.get_kv("attention:allow_babble", True))
        gate_open = allow_babble and (not in_feedback_cooldown) and (boredom_active or boost > 0.0)
        if not gate_open:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                "last_fire_ts": last_fire_ts,
                "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            self.debug(
                "curiosity_gate_closed",
                boredom_active=boredom_active,
                allow_babble=allow_babble,
                in_feedback_cooldown=in_feedback_cooldown,
                boost=boost,
                boredom_level=boredom_level,
                ticks_since_last=ticks_since_last,
            )
            return []

        # Optional context: PDNA, HRM, sensors (vision/voice)
        pdna_profile = await ctx.get_kv("pdna:profile", None)
        hrm = await ctx.get_kv("hrm:core", None)
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)

        # Sensor flags (to be set later by vision/voice wiring).
        # For now, they default to False and never crash.
        vision_online = bool(await ctx.get_kv("sensors:vision_online", False))
        voice_online = bool(await ctx.get_kv("sensors:voice_online", False))

        # Map boredom to a firing interval:
        # - low boredom => very rare curiosity
        # - high boredom => more frequent internal thoughts
        #
        # Interval in "ticks" (events). We clamp around [3, 20].
        # Interval in "ticks" (events). Base it on boredom, then optionally
        # accelerate it when we have a feedback-driven curiosity boost.
        if boredom_level <= 0.4:
            base_interval = 20
        elif boredom_level >= 0.9:
            base_interval = 3
        else:
            # linear interpolation between 20 and 3
            base_interval = int(20 - (boredom_level - 0.4) * (17 / 0.5))
            if base_interval < 3:
                base_interval = 3
            elif base_interval > 20:
                base_interval = 20

        target_interval = base_interval
        if boost > 0.0:
            # After a correction, we want a quicker, smaller probe.
            # Don’t let "low boredom" force a long wait.
            if target_interval > 8:
                target_interval = 8
            cut = int(round(min(boost, 1.0) * 6))  # 0..6
            target_interval = max(2, target_interval - cut)

        # If we haven't waited long enough, just persist state and exit
        if ticks_since_last < target_interval:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []

        # We've waited long enough AND boredom is at least moderate.
        # If boredom isn't actually high, we can decide to do nothing.
        if boredom_level < 0.4 and boost <= 0.0:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []


        # NEW: only allow internal babble if boredom is active and attention gate allows it
        boredom_active = bool(boredom.get("active", False))
        allow_babble = bool(await ctx.get_kv("attention:allow_babble", False))
        if not boredom_active or not allow_babble:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []
        # Hard throttle: even if ticks line up, don't spam internal thoughts too fast.
        min_fire_gap_s = 3.0
        if (now - last_fire_ts) < min_fire_gap_s:
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)
            return []

        
        # Reset tick counter and increment internal thought id
        ticks_since_last = 0
        internal_thought_id += 1
        last_fire_ts = now

        # Construct an internal curiosity prompt.
        # We adapt slightly based on context:
        #  - If we have HRM + last_idx, we can phrase it as "thinking more about that".
        #  - If vision is online, we can phrase it as visual wondering.
        #  - Otherwise, use a generic introspective curiosity.
        curiosity_text = None

        # Feedback-driven micro-probe: after a user correction, ask a smaller question.
        micro_probe = (boost > 0.0) and (event.topic in ("clock/tick", "percept/text"))
        if micro_probe:
            node_text = ""
            if hrm is not None and isinstance(hrm_last_idx, int):
                try:
                    node = hrm.get_node(hrm_last_idx)
                except Exception:
                    node = None

                if node is not None:
                    node_text = getattr(node, "text", "") or ""
                    node_text = str(node_text).strip()

            if node_text:
                clipped = " ".join(node_text.split())[:120]
                curiosity_text = (
                    f"Noted. Quick check about \"{clipped}\": what should I do differently? "
                    "Reply in 1 short sentence."
                )
            else:
                curiosity_text = "Noted. What should I do differently? Reply in 1 short sentence."

            # Spend some boost so we don't keep probing forever
            new_boost = 0.0
            await ctx.set_kv("curiosity:boost", new_boost)
            self.debug("curiosity_boost_spent", before=boost, after=new_boost)

            llm_enabled = bool(await ctx.get_kv("llm:enabled", False))

            if not llm_enabled:
                # Speak directly to the text UI when LLM is disabled
                return [
                    Event(
                        topic="act/speech",
                        payload={"text": curiosity_text, "channel": "repl", "style": "assistant"},
                        source=self.name,
                        meta={"kind": "curiosity_probe"},
                    )
                ]

        # Attempt HRM-based curiosity first
        if curiosity_text is None and hrm is not None and isinstance(hrm_last_idx, int):
            try:
                node = hrm.get_node(hrm_last_idx)
            except Exception:
                node = None

            # Walk backwards to the most recent USER node so we don't quote our own assistant output.
            if node is not None and getattr(node, "role", "") != "user":
                for back in range(1, 25):
                    try:
                        cand = hrm.get_node(hrm_last_idx - back)
                    except Exception:
                        cand = None
                    if cand is not None and getattr(cand, "role", "") == "user":
                        node = cand
                        break

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
            "last_fire_ts": last_fire_ts,
            "vision_online": vision_online,
            "voice_online": voice_online,
        }

        llm_enabled = bool(await ctx.get_kv("llm:enabled", False))
        if not llm_enabled:
            # LLM is disabled: generate a short babble line and speak it directly to the UI.
            try:
                from microbrain.babble_backend import babble_generate
            except Exception as e:
                self.debug("babble_backend_missing", err=str(e))
                return []

            prompt = (
                "Generate one short sentence of curious babble. "
                "Keep it under 12 words."
            )
            meta2 = {
                "boredom_active": True,
                "allow_babble": True,
                "source": "curiosity_drive",
                "boost": boost,
                "mimic": {
                    "unigrams": await ctx.get_kv("mimic:unigrams", {}) or {},
                    "bigrams": await ctx.get_kv("mimic:bigrams", {}) or {},
                    "recent_phrases": await ctx.get_kv("mimic:recent_phrases", []) or [],
                    "last_user_text": await ctx.get_kv("mimic:last_user_text", "") or "",
                },
            }

            babble = (await babble_generate(prompt, meta2)).strip()
            if not babble:
                return []

            # Persist curiosity state before speaking
            state.update(
                {
                    "ticks_since_last": ticks_since_last,
                    "internal_thought_id": internal_thought_id,
                    "last_fire_ts": last_fire_ts,
                }
            )
            await self.save_state(ctx, "curiosity_state", state)

            self.debug(
                "curiosity_babble_said",
                text=babble,
                boredom_level=boredom_level,
                target_interval=target_interval,
                internal_thought_id=internal_thought_id,
            )

            return [
                Event(
                    topic="act/speech",
                    payload={"text": babble, "channel": "repl", "style": "assistant"},
                    source=self.name,
                    meta={"kind": "curiosity_babble"},
                )
            ]


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
                "last_fire_ts": last_fire_ts,
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
            "clock/tick",
            "percept/text",
            "percept/vision",
            "act/speech",
        ],
        output_topics=["reason/request", "act/speech"],
        priority=-9,  # after boredom, before more "outward" behaviors if needed
    )
    yield CuriosityDriveNeuron(cfg)
