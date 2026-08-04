from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.hormone import (
    HormoneState,
    compute_base_needs,
    derive_ddna_modulators,
    derive_want_vector,
    merge_need_maps,
    safe_float,
    update_hormone_state,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("affect")


class HormoneStateNeuron(BaseNeuron):
    """
    Shared endocrine / want-field engine.

    Purpose:
      - maintain slow-moving hormone state in one place
      - derive DDNA-weighted modulation from the current PDNA profile
      - expose base needs + hormone weather + want vector to the rest of MB

    This neuron intentionally does not choose behavior directly. It only keeps
    the internal chemistry coherent and inspectable.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in (
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "act/speech",
            "affect/state",
            "affect/salience",
        ):
            return []

        now = time.time()
        state = await self.load_state(
            ctx,
            "hormone_state_runtime",
            default={
                "last_update_ts": now,
                "last_external_ts": now,
                "last_speech_ts": 0.0,
            },
        )

        if event.topic in ("percept/text", "percept/vision"):
            state["last_external_ts"] = now
        elif event.topic == "act/speech":
            payload = event.payload if isinstance(event.payload, dict) else {}
            channel = str(payload.get("channel", "repl") or "repl")
            state["last_speech_ts"] = now
            if channel != "thought":
                state["last_external_ts"] = now

        last_update_ts = safe_float(state.get("last_update_ts", now), now)
        dt_s = max(0.0, min(10.0, now - last_update_ts)) if last_update_ts > 0 else 1.0
        state["last_update_ts"] = now

        boredom = await ctx.get_kv("drive:boredom", {}) or {}
        stress = await ctx.get_kv("drive:stress", {}) or {}
        affect_state = await ctx.get_kv("affect:state", {}) or {}
        global_salience = await ctx.get_kv("affect:global_salience", None)
        pdna = await ctx.get_kv("pdna:profile", None)
        interaction = await ctx.get_kv("interaction:last_input", {}) or {}
        initiative_last = await ctx.get_kv("initiative:last", {}) or {}
        initiative_need_signal = await ctx.get_kv("drive:need_signal:initiative", {}) or {}
        prev_hormones = await ctx.get_kv("drive:hormones", {}) or {}
        power_sleep = bool(await ctx.get_kv("power:sleep", False))
        power_charging = bool(await ctx.get_kv("power:charging", False))

        boredom_level = safe_float((boredom or {}).get("level", 0.0), 0.0)
        stress_level = safe_float((stress or {}).get("level", 0.0), 0.0)

        salience = 0.0
        if isinstance(global_salience, (float, int)):
            salience = float(global_salience)
        elif isinstance(affect_state, dict):
            salience = safe_float(affect_state.get("salience", 0.0), 0.0)

        last_user_ts = safe_float(interaction.get("ts", now), now)
        unresolved_pending = bool(initiative_last.get("pending_text"))
        pending_age_s = safe_float(initiative_last.get("pending_age_s", 0.0), 0.0)
        coherence_hint = safe_float(initiative_last.get("talk_pressure", 0.0), 0.0)
        direct_address = 1.0 if unresolved_pending else 0.0
        blocked = 1.0 if bool(initiative_last.get("clarify_ready", False)) and unresolved_pending else 0.0
        resolution = 0.0
        if not unresolved_pending and safe_float(initiative_last.get("tier", 0.0), 0.0) <= 1.0:
            resolution = 0.35

        base_needs = compute_base_needs(
            boredom_level=boredom_level,
            stress_level=stress_level,
            salience=salience,
            now=now,
            last_user_ts=last_user_ts,
            last_external_ts=safe_float(state.get("last_external_ts", now), now),
            sleeping=power_sleep,
            charging=power_charging,
            unresolved_pending=unresolved_pending,
            pending_age_s=pending_age_s,
            coherence_hint=coherence_hint,
        )
        merged_needs = merge_need_maps(base_needs, initiative_need_signal)

        ddna_mods = derive_ddna_modulators(pdna)
        hormones = update_hormone_state(
            prev_hormones if prev_hormones else HormoneState().to_dict(),
            needs=merged_needs,
            ddna=ddna_mods,
            dt_s=dt_s,
            context={
                "blocked": blocked,
                "resolution": resolution,
                "interruption_cost": 0.0,
                "direct_address": direct_address,
            },
        )
        wants = derive_want_vector(hormones, needs=merged_needs, ddna=ddna_mods)

        await ctx.set_kv("drive:endocrine_authority", self.name)
        await ctx.set_kv("drive:needs_base", base_needs)
        await ctx.set_kv("drive:needs_stack", merged_needs)
        await ctx.set_kv("drive:ddna_modulators", ddna_mods)
        await ctx.set_kv("drive:hormones", hormones)
        await ctx.set_kv("drive:want_vector", wants)
        await ctx.set_kv(
            "drive:hormone_last",
            {
                "ts": now,
                "dt_s": round(dt_s, 4),
                "base_needs": base_needs,
                "needs": merged_needs,
                "need_signals": {
                    "initiative": initiative_need_signal,
                },
                "hormones": hormones,
                "wants": wants,
            },
        )

        await self.save_state(ctx, "hormone_state_runtime", state)
        self.debug(
            "hormone_state",
            arousal=hormones.get("arousal"),
            inquiry=hormones.get("inquiry"),
            caution=hormones.get("caution"),
            externalize=wants.get("externalize"),
        )
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            SERVICE_TOPIC,
            "percept/text",
            "percept/vision",
            "act/speech",
            "affect/state",
            "affect/salience",
        ],
        output_topics=[],
        priority=-7,
        cooldown_sec=0.0,
    )
    yield HormoneStateNeuron(cfg)
