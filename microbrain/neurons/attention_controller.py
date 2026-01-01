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

                    # Step 4.1: external-vs-internal speech gate
                    "external_hold_ms": 4000,
                    "last_external_ts": time.time(),
                    "allow_babble": False,
                    "prev_allow_babble": None,
                },
            )

            now = time.time()
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

        # Only treat certain topics as attentional "ticks"
        if event.topic not in ("percept/text", "percept/vision", "act/speech"):
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
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
        if boredom_level < 0.0:
            boredom_level = 0.0
        elif boredom_level > 1.0:
            boredom_level = 1.0

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

                # Step 4.1: external-vs-internal speech gate
                "external_hold_ms": 4000,
                "last_external_ts": time.time(),
                "allow_babble": False,
                "prev_allow_babble": None,
            },
        )

        # Attention Gating
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

        hold_ms = int(state.get("external_hold_ms", 4000))
        last_ext = float(state.get("last_external_ts", now))
        elapsed_ms = (now - last_ext) * 1000.0

        allow_babble = elapsed_ms >= float(hold_ms)
        state["allow_babble"] = allow_babble

        # Publish to global KV so other neurons can consult it
        await ctx.set_kv("attention:allow_babble", allow_babble)
        await ctx.set_kv("attention:focus_target", "internal" if allow_babble else "external")

        # Optional: only log when the gate flips
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
            # First-ever focus
            is_novel = True
        elif focus_signature != prev_sig:
            is_novel = True

        if is_novel:
            # New focus: reset age
            focus_age = 0
        else:
            # Same focus: age increases
            focus_age += 1

        # ------------------------------
        # Update salience based on novelty + boredom
        # ------------------------------
        # Baseline salience clamped to [0, 1]
        if salience < 0.0:
            salience = 0.0
        elif salience > 1.0:
            salience = 1.0

        if is_novel:
            # Novel stimuli: boost salience.
            # Stronger boost if boredom is high (bored mind reacts strongly to novelty).
            novelty_boost = 0.20 + 0.40 * boredom_level  # in [0.2, 0.6]
            salience = salience + novelty_boost
        else:
            # No novelty: allow salience to decay.
            # If boredom is already high, we decay more slowly
            # (mind is "clinging" to rare stimuli).
            base_decay = 0.04
            # When boredom is low, decay faster; when high, decay slower.
            decay = base_decay + 0.10 * (1.0 - boredom_level)  # ~[0.04, 0.14]
            salience = salience - decay

        # Clamp salience again
        if salience < 0.0:
            salience = 0.0
        elif salience > 1.0:
            salience = 1.0

        # ------------------------------
        # Persist updated state
        # ------------------------------
        state.update(
            {
                "salience": salience,
                "focus_signature": focus_signature,
                "focus_age": focus_age,
            }
        )
        await self.save_state(ctx, "attention_state", state)

        # Also publish a KV so others can quickly read it
        await ctx.set_kv("affect:global_salience", salience)

        # Log for debug visibility
        self.debug(
            "updated_attention",
            topic=event.topic,
            salience=salience,
            is_novel=is_novel,
            boredom_level=boredom_level,
            boredom_high=boredom_high,
            focus_age=focus_age,
        )

        # ------------------------------
        # Emit an affect/salience event for interested neurons
        # ------------------------------
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
