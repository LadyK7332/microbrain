from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.patterns.lexicon_store import simple_tokenize
from microbrain.patterns.pattern_edge_log import PatternEdgeLog

NEURON_NAME = Path(__file__).stem


def _now() -> float:
    return time.time()


def _to_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        t = payload.get("text") or payload.get("message") or ""
        return str(t).strip()
    return str(payload).strip()


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _sha16(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _token_sig(text: str) -> str:
    toks = [t for t in simple_tokenize(text) if t]
    return " ".join(toks[:24])


class CauseEffectNeuron(BaseNeuron):
    """
    Cause/effect + multimodal priming.

    1) Conversation inertia:
       - remembers last assistant speech
       - on new reason/request from user, stores a bundle:
         {prev_assistant, user_text, correlation_id, ts, modality}

    2) Multimodal trigger rules:
       - if audio-text contains keywords (e.g. "alarm", "fire") -> PRIME VISION
       - writes cross-modal pattern edges:
         sense:audio:<label> <-> concept:<concept>

    This does not force speech by default (optional announce flag).
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # ---- knobs ----
        prime_window_s = float(await ctx.get_kv("ce:prime_window_s", 10.0) or 10.0)
        prime_fps = float(await ctx.get_kv("ce:prime_fps", 6.0) or 6.0)
        trigger_cooldown_s = float(await ctx.get_kv("ce:trigger_cooldown_s", 4.0) or 4.0)
        announce = bool(await ctx.get_kv("ce:announce_triggers", False))

        # Default triggers (you can override by setting ce:audio_triggers in KV)
        audio_triggers = await ctx.get_kv("ce:audio_triggers", None)
        if not isinstance(audio_triggers, list):
            audio_triggers = [
                {"concept": "fire", "keywords": ["fire", "smoke", "alarm", "burning"]},
            ]

        # Pattern edges handle (optional; PatternBinderNeuron exposes it)
        edges = await ctx.get_kv("patterns:edges", None)
        if not isinstance(edges, PatternEdgeLog):
            edges = None

        # ------------------------------------------------------------
        # Track the last assistant speech (for "preceding output" logic)
        # ------------------------------------------------------------
        if event.topic == "act/speech":
            assistant_text = _to_text(event.payload)
            if assistant_text:
                await ctx.set_kv("ce:last_assistant_text", assistant_text)
                await ctx.set_kv("ce:last_assistant_ts", _now())
            return []

        # ------------------------------------------------------------
        # Track last vision frame (so we can bind triggers to "what was on screen")
        # ------------------------------------------------------------
        if event.topic == "percept/vision":
            payload = event.payload if isinstance(event.payload, dict) else {}
            # VisionWindowCapture emits data_ref + window.title
            data_ref = str(payload.get("data_ref", "") or "")
            window = payload.get("window", {}) if isinstance(payload.get("window", {}), dict) else {}
            title = str(window.get("title", "") or "")
            await ctx.set_kv(
                "ce:last_vision",
                {"ts": _now(), "data_ref": data_ref, "window_title": title, "frame_id": payload.get("frame_id")},
            )
            return []

        # ------------------------------------------------------------
        # Track audio energy (optional future hook for alarm/beep detectors)
        # ------------------------------------------------------------
        if event.topic == "affect/audio_energy":
            payload = event.payload if isinstance(event.payload, dict) else {}
            await ctx.set_kv(
                "ce:last_audio_energy",
                {"ts": _now(), "rms": payload.get("rms"), "peak": payload.get("peak"), "clipped": payload.get("clipped")},
            )
            return []

        # ------------------------------------------------------------
        # Main: on user "reason/request", build bundle + audio->vision priming
        # ------------------------------------------------------------
        if event.topic != "reason/request":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        user_text = _to_text(payload.get("text"))
        if not user_text:
            return []

        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        source = str(payload.get("source", "user") or "user")
        modality = str(raw_meta.get("input_modality", "") or "")
        channel = str(payload.get("channel", "default") or "default")

        # Only treat actual user-origin turns as "cause/effect"
        if source not in ("user", "mic"):
            return []

        prev_asst = str(await ctx.get_kv("ce:last_assistant_text", "") or "")
        prev_ts = float(await ctx.get_kv("ce:last_assistant_ts", 0.0) or 0.0)

        bundle: Dict[str, Any] = {
            "kind": "cause_effect_bundle",
            "ts": _now(),
            "channel": channel,
            "correlation_id": event.correlation_id,
            "prev_assistant": prev_asst,
            "prev_assistant_age_sec": max(0.0, _now() - prev_ts) if prev_ts else None,
            "user_text": user_text,
            "input_source": source,
            "input_modality": modality or ("audio" if source == "mic" else "text"),
        }

        await ctx.set_kv("cause_effect:last_bundle", bundle)

        out: List[Event] = []

        # ---- AUDIO TEXT TRIGGERS -> PRIME VISION + BIND CONCEPT ----
        # This fires when:
        # - the input came from mic (AudioCortex -> percept/text -> router_text -> reason/request)
        # - OR you typed the trigger words manually (still useful for testing)
        tokens = set(simple_tokenize(user_text))
        triggered: List[str] = []

        for rule in audio_triggers:
            concept = str(rule.get("concept", "") or "").strip().lower()
            kws = rule.get("keywords", []) or []
            kws = [str(k).strip().lower() for k in kws if str(k).strip()]
            if not concept or not kws:
                continue

            if any(k in tokens for k in kws):
                triggered.append(concept)

        if triggered:
            # cooldown per concept
            last_fired = await ctx.get_kv("ce:last_trigger_ts", {}) or {}
            if not isinstance(last_fired, dict):
                last_fired = {}

            fired_now: List[str] = []
            for concept in triggered:
                t_last = float(last_fired.get(concept, 0.0) or 0.0)
                if (_now() - t_last) >= trigger_cooldown_s:
                    last_fired[concept] = _now()
                    fired_now.append(concept)

            await ctx.set_kv("ce:last_trigger_ts", last_fired)

            if fired_now:
                # Prime vision capture: ensure enabled + temporarily raise FPS
                vision_enabled = bool(await ctx.get_kv("vision:enabled", False))
                if not vision_enabled:
                    out.append(Event(topic="control/vision", payload={"action": "on"}, source=NEURON_NAME))

                prev_fps = await ctx.get_kv("vision:fps", 2.0)
                await ctx.set_kv("ce:vision_fps_prev", prev_fps)
                await ctx.set_kv("vision:fps", prime_fps)
                await ctx.set_kv("ce:vision_prime_until", _now() + prime_window_s)

                # Bind sense(audio)->concept edges if the edge logger is available
                if edges is not None:
                    ts = _now()
                    for concept in fired_now:
                        sense_id = f"sense:audio:kw:{concept}"
                        concept_id = f"concept:{concept}"
                        edges.add("sense_concept", sense_id, concept_id, 0.06, role="system", channel=channel, ts=ts)
                        edges.add("concept_sense", concept_id, sense_id, 0.06, role="system", channel=channel, ts=ts)

                # Optional speech/notice
                if announce:
                    out.append(
                        Event(
                            topic="act/speech",
                            payload={"text": f"(Audio cue: {', '.join(fired_now)} — priming vision scan)", "style": "system", "channel": channel},
                            source=NEURON_NAME,
                            correlation_id=event.correlation_id,
                        )
                    )

        # ---- decay prime window (bring FPS back down when prime window expires) ----
        prime_until = float(await ctx.get_kv("ce:vision_prime_until", 0.0) or 0.0)
        if prime_until and _now() > prime_until:
            prev_fps = await ctx.get_kv("ce:vision_fps_prev", None)
            if prev_fps is not None:
                await ctx.set_kv("vision:fps", prev_fps)
            await ctx.set_kv("ce:vision_prime_until", 0.0)

        # Emit a context event other neurons can consume (optional, harmless)
        out.append(Event(topic="memory/cause_effect_context", payload=bundle, source=NEURON_NAME))
        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "reason/request",
            "act/speech",
            "percept/vision",
            "affect/audio_energy",
        ],
        output_topics=[
            "control/vision",
            "memory/cause_effect_context",
            "act/speech",
        ],
        priority=6,
    )
    yield CauseEffectNeuron(cfg)