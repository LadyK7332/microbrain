from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class AttentionControllerNeuron(BaseNeuron):
    """
    Global attention / salience controller.

    High-level behavior:

      - Listens to perceptual and speech events:
          * "percept/text"
          * "percept/vision" (future)
          * "act/speech"

      - Maintains a simple global salience scalar in [0.0, 1.0]
        that represents "how much am I paying attention right now?".

      - Detects novelty based on a coarse "focus signature" derived
        from the current event's topic + text (first N characters).
          * If the signature changes, we treat it as a new focus / novelty.
          * If it repeats, we treat it as sustained exposure and allow
            salience to decay over time.

      - Reads boredom drive (if present) from KV:
          * "drive:boredom" -> {"level": float, "high": bool, ...}

        and uses it to modulate:
          * how large the novelty boost is,
          * how quickly salience decays when nothing changes.

      - Writes the current salience to KV:
          * "affect:global_salience" -> float

      - Emits "affect/salience" events so other neurons can react.
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

        # ------------------------------
        # Curiosity refractory pause (negative feedback -> quiet gap)
        # ------------------------------
        if event.topic == "curiosity/adjust":
            payload = event.payload or {}
            if not isinstance(payload, dict):
                return []

            pause_s = float(payload.get("pause_s", 0.0) or 0.0)
            if pause_s <= 0.0:
                return []

            now = time.time()

            state: Dict[str, Any] = await self.load_state(
                ctx,
                "attention_state",
                default={
                    "salience": 0.3,
                    "focus_signature": None,
                    "focus_age": 0,
                    "external_hold_ms": 4000,
                    "last_external_ts": time.time(),
                    "allow_babble": False,
                    "prev_allow_babble": None,
                    "cooldown_until": 0.0,
                },
            )

            current_cd = float(state.get("cooldown_until", 0.0) or 0.0)
            new_cd = max(current_cd, now + pause_s)

            state["cooldown_until"] = new_cd
            state["last_external_ts"] = now
            state["allow_babble"] = False

            await ctx.set_kv("attention:allow_babble", False)
            await ctx.set_kv("attention:focus_target", "external")
            await self.save_state(ctx, "attention_state", state)

            self.debug(
                "attention_cooldown_set",
                pause_s=pause_s,
                cooldown_until=new_cd,
                reason=str(payload.get("reason", "")),
            )
            return []

        # ------------------------------
        # Step 4.1: Heartbeat-driven attention gate
        # ------------------------------
        if event.topic == "clock/tick":
            state: Dict[str, Any] = await self.load_state(
                ctx,
                "attention_state",
                default={
                    "salience": 0.3,
                    "focus_signature": None,
                    "focus_age": 0,
                    "external_hold_ms": 4000,
                    "last_external_ts": time.time(),
                    "allow_babble": False,
                    "prev_allow_babble": None,
                    "cooldown_until": 0.0,
                },
            )

            now = time.time()

            cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
            if now < cooldown_until:
                state["allow_babble"] = False
                await ctx.set_kv("attention:allow_babble", False)
                await ctx.set_kv("attention:focus_target", "external")
                await self.save_state(ctx, "attention_state", state)
                return []

            hold_ms = int(state.get("external_hold_ms", 4000))
            last_ext = float(state.get("last_external_ts", now))
            elapsed_ms = (now - last_ext) * 1000.0

            allow_babble = elapsed_ms >= float(hold_ms)
            state["allow_babble"] = allow_babble

            await ctx.set_kv("attention:allow_babble", allow_babble)
            await ctx.set_kv("attention:focus_target", "internal" if allow_babble else "external")

            prev = state.get("prev_allow_babble", None)
            if prev is None or bool(prev) != bool(allow_babble):
                state["prev_allow_babble"] = allow_babble
                self.debug(
                    "attention_gate_flip",
                    allow_babble=allow_babble,
                    focus_target=("internal" if allow_babble else "external"),
                    elapsed_ms=int(elapsed_ms),
                    hold_ms=hold_ms,
                )

            await self.save_state(ctx, "attention_state", state)
            return []

        # Only treat certain topics as attentional updates
        if event.topic not in ("percept/text", "percept/vision", "act/speech"):
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):closing
            return []

        # Extract a rough "content" string to fingerprint
        text = str(payload.get("text", "") or "").strip()
        # For vision, we might have a description field later
        if not text and event.topic == "percept/vision":
            text = str(payload.get("description", "") or "").strip()

        # If there's truly nothing to work with, just decay on a generic tick
        content_for_signature = text if text else event.topic

        # Build a simple focus signature: topic + first 80 chars lowercased
        content_snippet = content_for_signature[:80].lower()
        focus_signature = f"{event.topic}|{content_snippet}"

        # ------------------------------
        # Load boredom drive (if present)
        # ------------------------------
        boredom = await ctx.get_kv("drive:boredom", None)
        boredom_level = 0.0
        boredom_high = False
        if isinstance(boredom, dict):
            try:
                boredom_level = float(boredom.get("level", 0.0) or 0.0)
            except Exception:
                boredom_level = 0.0
            boredom_high = bool(boredom.get("high", False))

        # Clamp boredom to [0,1] for safety
        boredom_level = max(0.0, min(1.0, boredom_level))

        # ------------------------------
        # Load prior attention state
        # ------------------------------
        state: Dict[str, Any] = await self.load_state(
            ctx,
            "attention_state",
            default={
                "salience": 0.3,
                "focus_signature": None,
                "focus_age": 0,
                "external_hold_ms": 4000,
                "last_external_ts": time.time(),
                "allow_babble": False,
                "prev_allow_babble": None,
                "cooldown_until": 0.0,
            },
        )

        # ------------------------------
        # Attention gating (percept/speech updates)
        # ------------------------------
        now = time.time()

        # Best-effort source extraction
        src = ""
        if event.source:
            src = str(event.source)
        elif event.meta and event.meta.get("source"):
            src = str(event.meta.get("source"))
        elif isinstance(event.payload, dict) and event.payload.get("source"):
            src = str(event.payload.get("source"))

        # External stimulus updates last_external_ts
        if src in {"cli", "mic"}:
            state["last_external_ts"] = now

        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        hold_ms = int(state.get("external_hold_ms", 4000))
        last_ext = float(state.get("last_external_ts", now))
        elapsed_ms = (now - last_ext) * 1000.0

        allow_babble = False if now < cooldown_until else (elapsed_ms >= float(hold_ms))
        state["allow_babble"] = allow_babble

        await ctx.set_kv("attention:allow_babble", allow_babble)
        await ctx.set_kv("attention:focus_target", "internal" if allow_babble else "external")

        prev = state.get("prev_allow_babble", None)
        if prev is None or bool(prev) != bool(allow_babble):
            state["prev_allow_babble"] = allow_babble
            self.debug(
                "attention_gate_flip",
                allow_babble=allow_babble,
                focus_target=("internal" if allow_babble else "external"),
                elapsed_ms=int(elapsed_ms),
                hold_ms=hold_ms,
                src=src,
            )

        salience = float(state.get("salience", 0.3) or 0.3)
        prev_sig = state.get("focus_signature", None)
        focus_age = int(state.get("focus_age", 0) or 0)

        # ------------------------------
        # Novelty detection
        # ------------------------------
        is_novel = False
        if prev_sig is None:
            is_novel = True
        elif focus_signature != prev_sig:
            is_novel = True

        if is_novel:
            focus_age = 0
        else:
            focus_age += 1

        # ------------------------------
        # Update salience based on novelty + boredom
        # ------------------------------
        salience = max(0.0, min(1.0, salience))

        if is_novel:
            novelty_boost = 0.20 + 0.40 * boredom_level  # in [0.2, 0.6]
            salience = salience + novelty_boost
        else:
            base_decay = 0.04
            decay = base_decay + 0.10 * (1.0 - boredom_level)  # ~[0.04, 0.14]
            salience = salience - decay

        salience = max(0.0, min(1.0, salience))

        # Persist updated state
        state.update(
            {
                "salience": salience,
                "focus_signature": focus_signature,
                "focus_age": focus_age,
            }
        )
        await self.save_state(ctx, "attention_state", state)

        await ctx.set_kv("affect:global_salience", salience)

        self.debug(
            "updated_attention",
            topic=event.topic,
            salience=salience,
            is_novel=is_novel,
            boredom_level=boredom_level,
            boredom_high=boredom_high,
            focus_age=focus_age,
        )

        salience_event = Event(
            topic="affect/salience",
            payload={
                "salience": salience,
                "is_novel": is_novel,
                "boredom_level": boredom_level,
                "boredom_high": boredom_high,
                "focus_age": focus_age,
                "source_topic": event.topic,
            },
            source=self.name,
            correlation_id=event.correlation_id,
        )

        return [salience_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "clock/tick",
            "curiosity/adjust",
            "percept/text",
            "percept/vision",  # safe even if not yet used
            "act/speech",
        ],
        output_topics=["affect/salience"],
        # Priority: moderate; it should run after raw percepts are in,
        # but before higher-level drives that might read salience.
        priority=3,
    )
    yield AttentionControllerNeuron(cfg)
