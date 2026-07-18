from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

_COMMON_FRAGMENTS = {
    "ok", "okay", "k", "kk", "yes", "no", "nah", "yep", "nope",
    "lol", "lmao", "rofl", "ty", "thanks", "thank you", "hi", "hey", "hello",
    "moin", "yo", "nice", "good", "cool", "same", "true", "right",
}

_UNCERTAINTY_MARKERS = (
    "?", "o.o", "o_o", "???", "huh", "what", "why", "how", "maybe",
    "possibly", "not sure", "unsure", "unclear", "unknown", "missing",
    "doesn't make sense", "doesnt make sense", "confusing", "stuck",
)

_GOAL_WORDS = (
    "build", "make", "patch", "wire", "add", "update", "fix", "change",
    "implement", "connect", "route", "use", "load", "read", "ingest",
)

_PRONOUN_ANCHORS = (" it ", " that ", " this ", " those ", " these ")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _fingerprint(text: str) -> str:
    norm = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm[:140]


class CuriosityRelationshipExplorationNeuron(BaseNeuron):
    """
    Curiosity-side gap detector for clarification learning.

    Shape:
        unknown + current active sense + scene relevance + low inference confidence
        -> curiosity gap pressure
        -> one narrow clarification question
        -> answer attaches back to the gap object.

    Important boundary:
        This neuron only raises clarification from the currently ACTIVE sense.
        A stale visual/audio gap should not interrupt a live text conversation unless
        that sense becomes active again.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in (
            "percept/text",
            "percept/audio",
            "percept/vision",
            "vision/proto_object",
            "vision/percept_commit",
            "thought/object",
        ):
            return []

        now = time.time()

        if event.topic == "thought/object":
            await self._mark_gap_asked(event, ctx, now)
            return []

        active_sense = self._sense_from_event(event)
        if not active_sense:
            return []

        await self._remember_active_sense(event, ctx, active_sense, now)

        # If MB asked a clarification and the user replies, bind the answer to the
        # open gap instead of treating it as unrelated text soup.
        if event.topic == "percept/text":
            resolved = await self._maybe_resolve_gap_with_text(event, ctx, now)
            if resolved:
                return []

        gap = await self._detect_gap(event, ctx, active_sense, now)
        if not gap:
            return []

        pressure = float(gap.get("pressure", 0.0) or 0.0)
        ask_threshold = float(await ctx.get_kv("curiosity:relation_gap_ask_threshold", 0.62) or 0.62)
        if pressure < ask_threshold:
            await ctx.set_kv("curiosity:active_gap", gap)
            return [
                Event(
                    topic="curiosity/gap",
                    payload=gap,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "curiosity_gap", "active_sense": active_sense, "pressure": round(pressure, 4)},
                )
            ]

        if not await self._can_ask(ctx, now, gap):
            await ctx.set_kv("curiosity:active_gap", gap)
            return [
                Event(
                    topic="curiosity/gap",
                    payload=gap,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={"kind": "curiosity_gap", "active_sense": active_sense, "pressure": round(pressure, 4)},
                )
            ]

        question = str(gap.get("question", "") or "").strip()
        if not question:
            return []

        gap["status"] = "asking"
        gap["asked_ts"] = now
        gap["question"] = question
        await ctx.set_kv("curiosity:active_gap", gap)
        await ctx.set_kv("curiosity:relation_gap_last_ask_ts", now)
        await ctx.set_kv("curiosity:relation_gap_last_anchor", str(gap.get("anchor", "") or ""))

        channel = str(gap.get("channel", "repl") or "repl")
        return [
            Event(
                topic="curiosity/gap",
                payload=gap,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "curiosity_gap", "active_sense": active_sense, "pressure": round(pressure, 4)},
            ),
            Event(
                topic="curiosity/adjust",
                payload={
                    "boost": min(0.16, 0.05 + pressure * 0.12),
                    "pause_s": 6.0,
                    "reason": "active-sense clarification gap",
                    "active_sense": active_sense,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "curiosity_gap_pressure", "active_sense": active_sense},
            ),
            Event(
                topic="thought/object",
                payload={
                    "kind": "curiosity_gap_clarify",
                    "content": question,
                    "scene_ref": str(gap.get("anchor", "") or "scene:unknown"),
                    "need_ref": "curiosity",
                    "status": "drawer_waiting",
                    "gap": gap,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "channel": "thought",
                    "kind": "curiosity_gap_clarify",
                    "active_sense": active_sense,
                    "gap_id": gap.get("id", ""),
                },
            ),
        ]

    def _sense_from_event(self, event: Event) -> str:
        if event.topic == "percept/text":
            return "text"
        if event.topic == "percept/audio":
            return "audio"
        if event.topic in ("percept/vision", "vision/proto_object", "vision/percept_commit"):
            return "vision"
        return ""

    async def _remember_active_sense(self, event: Event, ctx, active_sense: str, now: float) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {"value": event.payload}
        summary = self._summary_for_event(event, payload)
        entry = {
            "sense": active_sense,
            "topic": event.topic,
            "source": event.source,
            "summary": summary,
            "ts": now,
            "correlation_id": event.correlation_id,
        }
        await ctx.set_kv("curiosity:active_sense", active_sense)
        await ctx.set_kv("curiosity:active_sense:last", entry)

    async def _detect_gap(self, event: Event, ctx, active_sense: str, now: float) -> Optional[Dict[str, Any]]:
        payload = event.payload if isinstance(event.payload, dict) else {"value": event.payload}
        if active_sense == "text":
            return await self._detect_text_gap(event, ctx, payload, now)
        if active_sense == "vision":
            return await self._detect_vision_gap(event, ctx, payload, now)
        if active_sense == "audio":
            return await self._detect_audio_gap(event, ctx, payload, now)
        return None

    async def _detect_text_gap(self, event: Event, ctx, payload: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        src = str(raw_meta.get("source", payload.get("source", "user")) or "user")
        if src in ("assistant", "system", "mb"):
            return None

        text = _clean_text(payload.get("text", ""))
        if not text or text.startswith("/"):
            return None

        lowered = f" {text.lower()} "
        norm = _fingerprint(text)
        words = [w for w in norm.split() if w]
        if norm in _COMMON_FRAGMENTS:
            return None

        marker_hits = sum(1 for marker in _UNCERTAINTY_MARKERS if marker in lowered)
        goal_hits = sum(1 for word in _GOAL_WORDS if f" {word} " in lowered)
        pronoun_hits = sum(1 for anchor in _PRONOUN_ANCHORS if anchor in lowered)
        short_fragment = 0 < len(words) <= 3
        explicit_question = "?" in text

        gap_type = ""
        unknown_importance = 0.0
        inference_confidence = 0.70
        question = ""
        anchor = text[:120]

        if short_fragment and (marker_hits > 0 or any(ch in text for ch in ("_", ".", ":", "=", "?"))):
            gap_type = "unknown_fragment"
            unknown_importance = 0.78
            inference_confidence = 0.22
            question = f"gap:unknown_fragment:{anchor}"
        elif explicit_question and pronoun_hits > 0 and len(words) <= 12:
            gap_type = "missing_referent"
            unknown_importance = 0.72
            inference_confidence = 0.34
            question = "gap:missing_referent"
        elif marker_hits >= 2:
            gap_type = "uncertainty_cluster"
            unknown_importance = 0.66
            inference_confidence = 0.38
            question = "gap:uncertainty_cluster"
        elif goal_hits > 0 and pronoun_hits > 0:
            gap_type = "target_missing"
            unknown_importance = 0.62
            inference_confidence = 0.42
            question = "gap:target_missing"
        else:
            return None

        pressure = self._gap_pressure(
            unknown_importance=unknown_importance,
            active_sense_weight=1.0,
            scene_relevance=0.95,
            inference_confidence=inference_confidence,
        )
        channel = str(raw_meta.get("channel", payload.get("channel", "repl")) or "repl")
        return self._gap_object(
            active_sense="text",
            gap_type=gap_type,
            anchor=anchor,
            summary=text,
            question=question,
            pressure=pressure,
            channel=channel,
            event=event,
            now=now,
        )

    async def _detect_vision_gap(self, event: Event, ctx, payload: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        if event.topic not in ("vision/proto_object", "percept/vision"):
            return None

        status = str(payload.get("status", "") or "").lower()
        resolved = str(payload.get("resolved_label", "") or "").strip()
        should_ask = bool(payload.get("should_ask", False))
        curiosity = float(payload.get("curiosity", 0.0) or 0.0)
        stability = float(payload.get("stability", 0.0) or 0.0)
        fallback = str(payload.get("fallback_ref", "that thing") or "that thing")

        if event.topic == "percept/vision":
            objects = payload.get("objects", []) or []
            description = _clean_text(payload.get("description", ""))
            if not description and not objects:
                return None
            return None  # raw vision frames should not ask until proto tracker stabilizes.

        if resolved or status == "labeled":
            return None
        if not should_ask and not (curiosity >= 0.86 and stability >= 0.55):
            return None

        pressure = self._gap_pressure(
            unknown_importance=max(0.55, curiosity),
            active_sense_weight=1.0,
            scene_relevance=max(0.55, stability),
            inference_confidence=0.18 if not resolved else 0.8,
        )
        question = f"gap:unknown_visual_object:{fallback}"
        return self._gap_object(
            active_sense="vision",
            gap_type="unknown_visual_object",
            anchor=fallback,
            summary=f"{fallback} | stability={stability:.2f} curiosity={curiosity:.2f}",
            question=question,
            pressure=pressure,
            channel="repl",
            event=event,
            now=now,
        )

    async def _detect_audio_gap(self, event: Event, ctx, payload: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        text = _clean_text(payload.get("text", payload.get("transcript", "")))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        if not text or confidence >= 0.45:
            return None
        pressure = self._gap_pressure(
            unknown_importance=0.60,
            active_sense_weight=1.0,
            scene_relevance=0.80,
            inference_confidence=max(0.05, confidence),
        )
        return self._gap_object(
            active_sense="audio",
            gap_type="low_confidence_audio",
            anchor=text[:120] or "unclear audio",
            summary=text or "unclear audio",
            question="gap:low_confidence_audio",
            pressure=pressure,
            channel="repl",
            event=event,
            now=now,
        )

    def _gap_pressure(
        self,
        *,
        unknown_importance: float,
        active_sense_weight: float,
        scene_relevance: float,
        inference_confidence: float,
    ) -> float:
        return _clamp01(
            float(unknown_importance)
            * float(active_sense_weight)
            * float(scene_relevance)
            * (1.0 - float(inference_confidence))
        )

    def _gap_object(
        self,
        *,
        active_sense: str,
        gap_type: str,
        anchor: str,
        summary: str,
        question: str,
        pressure: float,
        channel: str,
        event: Event,
        now: float,
    ) -> Dict[str, Any]:
        gap_id = f"gap_{active_sense}_{abs(hash((gap_type, _fingerprint(anchor)))) % 1000000:06d}"
        return {
            "kind": "curiosity.relationship_gap.v1",
            "id": gap_id,
            "status": "candidate",
            "active_sense": active_sense,
            "gap_type": gap_type,
            "anchor": anchor[:160],
            "summary": summary[:260],
            "question": question,
            "pressure": round(float(pressure), 4),
            "channel": channel,
            "source_topic": event.topic,
            "source": event.source,
            "correlation_id": event.correlation_id,
            "ts": now,
        }

    async def _can_ask(self, ctx, now: float, gap: Dict[str, Any]) -> bool:
        power_state = await ctx.get_kv("power:state", {}) or {}
        if isinstance(power_state, dict) and bool(power_state.get("sleep", False)):
            return False
        if bool(await ctx.get_kv("control:r_pending", False)):
            return False
        if bool(await ctx.get_kv("curiosity:relation_gap_enabled", True)) is False:
            return False

        min_interval = float(await ctx.get_kv("curiosity:relation_gap_min_interval_s", 25.0) or 25.0)
        last_ts = float(await ctx.get_kv("curiosity:relation_gap_last_ask_ts", 0.0) or 0.0)
        if (now - last_ts) < min_interval:
            return False

        active = await ctx.get_kv("curiosity:active_gap", {}) or {}
        if isinstance(active, dict):
            status = str(active.get("status", "") or "")
            if status in ("asking", "asked", "waiting_for_answer"):
                active_ts = float(active.get("asked_ts", active.get("ts", 0.0)) or 0.0)
                wait_ttl = float(await ctx.get_kv("curiosity:relation_gap_wait_ttl_s", 180.0) or 180.0)
                if (now - active_ts) < wait_ttl:
                    return False

        last_anchor = str(await ctx.get_kv("curiosity:relation_gap_last_anchor", "") or "")
        if last_anchor and _fingerprint(last_anchor) == _fingerprint(str(gap.get("anchor", "") or "")):
            repeat_gap = float(await ctx.get_kv("curiosity:relation_gap_repeat_gap_s", 120.0) or 120.0)
            if (now - last_ts) < repeat_gap:
                return False

        return True

    async def _mark_gap_asked(self, event: Event, ctx, now: float) -> None:
        kind = str((event.meta or {}).get("kind", "") or "")
        if kind != "curiosity_gap_clarify":
            return
        gap = await ctx.get_kv("curiosity:active_gap", {}) or {}
        if not isinstance(gap, dict):
            return
        gap["status"] = "asked"
        gap["asked_ts"] = now
        await ctx.set_kv("curiosity:active_gap", gap)

    async def _maybe_resolve_gap_with_text(self, event: Event, ctx, now: float) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        src = str(raw_meta.get("source", payload.get("source", "user")) or "user")
        if src in ("assistant", "system", "mb"):
            return False

        gap = await ctx.get_kv("curiosity:active_gap", {}) or {}
        if not isinstance(gap, dict):
            return False
        if str(gap.get("status", "") or "") not in ("asking", "asked", "waiting_for_answer"):
            return False

        asked_ts = float(gap.get("asked_ts", gap.get("ts", 0.0)) or 0.0)
        ttl = float(await ctx.get_kv("curiosity:relation_gap_wait_ttl_s", 180.0) or 180.0)
        if (now - asked_ts) > ttl:
            gap["status"] = "expired"
            await ctx.set_kv("curiosity:active_gap", gap)
            return False

        answer = _clean_text(payload.get("text", ""))
        if not answer or answer.startswith("/"):
            return False

        resolution = {
            "kind": "curiosity.relationship_gap_resolution.v1",
            "gap": gap,
            "answer": answer[:500],
            "answer_topic": event.topic,
            "answer_source": event.source,
            "ts": now,
            "correlation_id": event.correlation_id,
        }
        await ctx.set_kv("curiosity:last_gap_resolution", resolution)
        await ctx.set_kv("curiosity:active_gap", {})
        await ctx.set_kv("curiosity:relation_gap_last_resolution_ts", now)
        await ctx.emit(
            Event(
                topic="curiosity/gap_resolved",
                payload=resolution,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "curiosity_gap_resolved", "active_sense": gap.get("active_sense", "")},
            )
        )
        return True

    def _summary_for_event(self, event: Event, payload: Dict[str, Any]) -> str:
        if event.topic in ("percept/text", "percept/audio"):
            return _clean_text(payload.get("text", payload.get("transcript", "")))[:240]
        if event.topic == "vision/proto_object":
            return str(payload.get("fallback_ref", payload.get("summary", "that thing")) or "that thing")[:240]
        if event.topic == "vision/percept_commit":
            return _clean_text(payload.get("text", payload.get("fallback_ref", "")))[:240]
        if event.topic == "percept/vision":
            desc = _clean_text(payload.get("description", ""))
            objects = payload.get("objects", []) or []
            if isinstance(objects, list) and objects:
                return f"{desc} | objects: {', '.join(str(x) for x in objects[:8])}"[:240]
            return desc[:240]
        return str(payload)[:240]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "percept/audio",
            "percept/vision",
            "vision/proto_object",
            "vision/percept_commit",
            "thought/object",
        ],
        output_topics=["curiosity/gap", "curiosity/gap_resolved", "curiosity/adjust", "thought/object"],
        priority=18,
        cooldown_sec=0.0,
    )
    yield CuriosityRelationshipExplorationNeuron(cfg)
