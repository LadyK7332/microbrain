from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from microbrain.hormone import derive_ddna_modulators
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm_intent(text: str) -> str:
    return "_".join(str(text or "").strip().lower().split())[:80]


def _is_control_event(event: Event) -> bool:
    meta = event.meta if isinstance(event.meta, Mapping) else {}
    if event.topic.startswith(("ui/", "control/", "debug/")):
        return True
    if meta.get("cognitive_visible") is False:
        return True
    if meta.get("store_in_memory") is False and meta.get("memory_source") == "system_telemetry":
        return True
    payload = event.payload
    text = ""
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, Mapping):
        text = str(payload.get("text", "") or "")
    return bool(text.lstrip().startswith("/"))


class ThoughtMomentumNeuron(BaseNeuron):
    """
    Maintains short-lived thought/intent momentum.

    This is not speech and not memory. It is a small state organ that keeps
    active vectors alive long enough to bias later context/action selection.

    Core rule:
      event -> vector pressure -> decay over time -> resolve/vent/override

    KV keys:
      - thought:momentum
      - thought:momentum:active_vectors
      - thought:momentum:last_update_ts
    """

    MAX_VECTORS = 8
    MAX_STRENGTH = 1.0
    MIN_KEEP = 0.035
    DEFAULT_DECAY_PER_S = 0.018

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if _is_control_event(event):
            return []

        now = time.time()
        pdna = await ctx.get_kv("pdna:profile", None)
        ddna_mods = await ctx.get_kv("drive:ddna_modulators", None)
        if not isinstance(ddna_mods, dict) or not ddna_mods:
            ddna_mods = derive_ddna_modulators(pdna)
            await ctx.set_kv("drive:ddna_modulators", ddna_mods)
        thought_gain = max(0.25, min(2.0, _safe_float((ddna_mods or {}).get("thought_momentum_gain"), 1.0)))
        persistence_gain = max(0.25, min(2.0, _safe_float((ddna_mods or {}).get("drawer_persistence_gain"), 1.0)))

        state = await self._load_state(ctx, now)
        vectors = list(state.get("active_vectors", []) or [])
        vectors = self._decay_vectors(vectors, now, decay_resistance=persistence_gain)

        changed = False
        additions: List[Dict[str, Any]] = []

        if event.topic == "clock/tick":
            changed = True

        elif event.topic == "percept/text":
            additions.extend(self._vectors_from_text(event, now))

        elif event.topic == "thought/internal":
            additions.extend(self._vectors_from_internal_thought(event, now))

        elif event.topic == "act/speech":
            additions.extend(self._vectors_from_speech(event, now))

        elif event.topic in {"drive:boredom", "drive:social_interaction", "drive:social_experimentation"}:
            additions.extend(self._vectors_from_drive(event, now))

        elif event.topic == "curiosity/adjust":
            additions.extend(self._vectors_from_curiosity(event, now))

        elif event.topic in {"reinforcement/feedback", "control/reinforce"}:
            vectors = self._apply_feedback_vent(vectors, event)
            changed = True

        if additions:
            for vec in additions:
                vec = dict(vec)
                vec["strength"] = round(_clamp(_safe_float(vec.get("strength", 0.0), 0.0) * thought_gain), 4)
                vec["decay_per_s"] = max(0.001, _safe_float(vec.get("decay_per_s", self.DEFAULT_DECAY_PER_S), self.DEFAULT_DECAY_PER_S) / persistence_gain)
                vec["ddna_thought_gain"] = round(thought_gain, 4)
                vec["ddna_persistence_gain"] = round(persistence_gain, 4)
                vectors = self._upsert_vector(vectors, vec)
            changed = True

        vectors = self._rank_vectors(vectors)[: self.MAX_VECTORS]
        summary = self._summarize(vectors, now)
        summary["last_event_topic"] = event.topic
        summary["updated_at"] = now

        if changed:
            await ctx.set_kv("thought:momentum", summary)
            await ctx.set_kv("thought:momentum:active_vectors", vectors)
            await ctx.set_kv("thought:momentum:last_update_ts", now)
            self.debug(
                "momentum_updated",
                topic=event.topic,
                count=len(vectors),
                dominant=summary.get("dominant_intent", ""),
                pressure=round(float(summary.get("pressure", 0.0) or 0.0), 3),
            )
            return [
                Event(
                    topic="thought/momentum",
                    payload=summary,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "thought_momentum_state",
                        "channel": "thought",
                        "quiet": True,
                        "store_in_memory": False,
                        "reinforcement_eligible": False,
                        "self_output_track": False,
                        "cognitive_visible": False,
                    },
                )
            ]

        return []

    async def _load_state(self, ctx, now: float) -> Dict[str, Any]:
        active = await ctx.get_kv("thought:momentum:active_vectors", [])
        if not isinstance(active, list):
            active = []
        return {"active_vectors": active, "loaded_at": now}

    def _decay_vectors(self, vectors: List[Dict[str, Any]], now: float, *, decay_resistance: float = 1.0) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for raw in vectors:
            if not isinstance(raw, Mapping):
                continue
            vec = dict(raw)
            strength = _safe_float(vec.get("strength", 0.0), 0.0)
            decay = _safe_float(vec.get("decay_per_s", self.DEFAULT_DECAY_PER_S), self.DEFAULT_DECAY_PER_S) / max(0.25, decay_resistance)
            last_ts = _safe_float(vec.get("last_update_ts", vec.get("created_at", now)), now)
            dt = max(0.0, now - last_ts)
            strength = max(0.0, strength - (decay * dt))
            if strength < self.MIN_KEEP:
                continue
            vec["strength"] = round(_clamp(strength, 0.0, self.MAX_STRENGTH), 4)
            vec["last_update_ts"] = now
            out.append(vec)
        return out

    def _upsert_vector(self, vectors: List[Dict[str, Any]], incoming: Dict[str, Any]) -> List[Dict[str, Any]]:
        intent = _norm_intent(str(incoming.get("intent", "") or ""))
        if not intent:
            return vectors
        incoming = dict(incoming)
        incoming["intent"] = intent
        for vec in vectors:
            if _norm_intent(str(vec.get("intent", "") or "")) == intent:
                vec["strength"] = round(_clamp(_safe_float(vec.get("strength", 0.0), 0.0) + _safe_float(incoming.get("strength", 0.0), 0.0)), 4)
                vec["decay_per_s"] = min(_safe_float(vec.get("decay_per_s", self.DEFAULT_DECAY_PER_S), self.DEFAULT_DECAY_PER_S), _safe_float(incoming.get("decay_per_s", self.DEFAULT_DECAY_PER_S), self.DEFAULT_DECAY_PER_S))
                vec["last_update_ts"] = incoming.get("last_update_ts", time.time())
                vec["source_topic"] = incoming.get("source_topic", vec.get("source_topic", ""))
                vec["reason"] = incoming.get("reason", vec.get("reason", ""))
                tags = set(vec.get("tags", []) or []) | set(incoming.get("tags", []) or [])
                vec["tags"] = sorted(str(t) for t in tags if str(t))[:10]
                return vectors
        vectors.append(incoming)
        return vectors

    def _rank_vectors(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(vectors, key=lambda v: _safe_float(v.get("strength", 0.0), 0.0), reverse=True)

    def _summarize(self, vectors: List[Dict[str, Any]], now: float) -> Dict[str, Any]:
        ranked = self._rank_vectors(vectors)
        dominant = ranked[0] if ranked else {}
        pressure = sum(_safe_float(v.get("strength", 0.0), 0.0) for v in ranked)
        pressure = _clamp(pressure / max(1.0, min(4.0, len(ranked) or 1.0)))
        return {
            "schema": "thought_momentum.v1",
            "ts": now,
            "active": bool(ranked),
            "pressure": round(pressure, 4),
            "count": len(ranked),
            "dominant_intent": str(dominant.get("intent", "") or ""),
            "dominant_strength": round(_safe_float(dominant.get("strength", 0.0), 0.0), 4) if dominant else 0.0,
            "active_vectors": ranked,
        }

    def _payload_text(self, event: Event) -> str:
        if isinstance(event.payload, str):
            return event.payload.strip()
        if isinstance(event.payload, Mapping):
            return str(event.payload.get("text", "") or "").strip()
        return ""

    def _vectors_from_text(self, event: Event, now: float) -> List[Dict[str, Any]]:
        text = self._payload_text(event)
        if not text:
            return []
        lowered = text.lower()
        vectors: List[Dict[str, Any]] = []

        is_question = "?" in text or lowered.startswith(("what ", "why ", "how ", "when ", "where ", "who ", "can ", "could ", "would "))
        if is_question:
            vectors.append(self._new_vector("understand_user", 0.24, 0.012, now, event, "question/request continues until answered", ["understanding", "continuity"]))
            vectors.append(self._new_vector("curiosity", 0.12, 0.018, now, event, "question raised curiosity", ["curiosity"]))

        if any(word in lowered.split() for word in ("moin", "hi", "hello", "hey")) or "good morning" in lowered:
            vectors.append(self._new_vector("social_continuity", 0.16, 0.020, now, event, "social opening should carry briefly", ["social"]))

        if any(phrase in lowered for phrase in ("remember", "why did", "what happened", "same thing", "loop", "again")):
            vectors.append(self._new_vector("resolve_thread", 0.20, 0.014, now, event, "user referenced continuity or unresolved thread", ["continuity", "memory"]))

        return vectors

    def _vectors_from_speech(self, event: Event, now: float) -> List[Dict[str, Any]]:
        text = self._payload_text(event)
        if not text:
            return []
        return [self._new_vector("await_result", 0.10, 0.030, now, event, "MB output awaits result/change", ["attempt", "result"])]

    def _vectors_from_internal_thought(self, event: Event, now: float) -> List[Dict[str, Any]]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        kind = str(payload.get("kind", "") or "")
        source_need = str(payload.get("source_need", payload.get("need", "")) or "")
        urgency = _safe_float(payload.get("urgency", 0.0), 0.0)
        vectors: List[Dict[str, Any]] = []
        if kind in {"need_state", "need_expression_blocked"} or source_need:
            strength = 0.16 + min(0.30, max(0.0, urgency) * 0.30)
            vectors.append(self._new_vector("need_resolution", strength, 0.014, now, event, "need-state thought should carry until resolved", ["need", source_need or "unknown"]))
            if source_need == "power":
                vectors.append(self._new_vector("seek_charge", min(0.34, strength), 0.016, now, event, "power need is seeking a charge route", ["power", "charge"]))
        return vectors

    def _vectors_from_drive(self, event: Event, now: float) -> List[Dict[str, Any]]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        vectors: List[Dict[str, Any]] = []
        if event.topic == "drive:boredom":
            level = _safe_float(payload.get("level", payload.get("boredom", 0.0)), 0.0)
            if level >= 0.45 or bool(payload.get("active", False)):
                vectors.append(self._new_vector("seek_novelty", min(0.22, level * 0.22), 0.018, now, event, "boredom creates novelty momentum", ["boredom", "novelty"]))
        elif event.topic == "drive:social_interaction":
            level = _safe_float(payload.get("level", 0.0), 0.0)
            if level >= 0.35 or bool(payload.get("active", False)):
                vectors.append(self._new_vector("seek_social_contact", min(0.20, level * 0.20), 0.020, now, event, "social pressure carries across turns", ["social"]))
        elif event.topic == "drive:social_experimentation":
            pressure = _safe_float(payload.get("pressure", 0.0), 0.0)
            if pressure >= 0.40:
                vectors.append(self._new_vector("social_experiment", min(0.18, pressure * 0.18), 0.022, now, event, "social experimentation pressure remains available", ["social", "novelty"]))
        return vectors

    def _vectors_from_curiosity(self, event: Event, now: float) -> List[Dict[str, Any]]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        boost = _safe_float(payload.get("boost", 0.0), 0.0)
        if boost <= 0.0:
            return []
        return [self._new_vector("curiosity", min(0.22, boost * 0.22), 0.016, now, event, "curiosity boost adds thought momentum", ["curiosity"])]

    def _apply_feedback_vent(self, vectors: List[Dict[str, Any]], event: Event) -> List[Dict[str, Any]]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        delta = _safe_float(payload.get("delta", payload.get("score", 0.0)), 0.0)
        # Any explicit reinforcement is a partial resolution signal. Positive vents
        # more; negative leaves some resolve_thread pressure alive for correction.
        factor = 0.45 if delta > 0 else 0.72
        out = []
        for vec in vectors:
            vec = dict(vec)
            vec["strength"] = round(_safe_float(vec.get("strength", 0.0), 0.0) * factor, 4)
            if vec["strength"] >= self.MIN_KEEP:
                out.append(vec)
        return out

    def _new_vector(self, intent: str, strength: float, decay: float, now: float, event: Event, reason: str, tags: List[str]) -> Dict[str, Any]:
        return {
            "intent": _norm_intent(intent),
            "strength": round(_clamp(strength, 0.0, self.MAX_STRENGTH), 4),
            "decay_per_s": max(0.001, float(decay)),
            "created_at": now,
            "last_update_ts": now,
            "source_topic": event.topic,
            "source": event.source,
            "reason": reason,
            "tags": tags,
        }


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
            "drive:boredom",
            "drive:social_interaction",
            "drive:social_experimentation",
            "curiosity/adjust",
            "reinforcement/feedback",
            "control/reinforce",
            "clock/tick",
            "thought/internal",
        ],
        output_topics=["thought/momentum"],
        priority=4,
        cooldown_sec=0.0,
    )
    yield ThoughtMomentumNeuron(cfg)
