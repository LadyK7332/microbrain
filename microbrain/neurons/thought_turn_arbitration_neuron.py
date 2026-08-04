from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.hormone import derive_ddna_modulators
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import service_tick_is_for, service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("cognition")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm_token(value: Any, fallback: str = "") -> str:
    text = str(value or fallback or "").strip().lower()
    text = "_".join(part for part in text.replace("/", "_").replace(":", "_").split() if part)
    return text[:80] or fallback


class ThoughtTurnArbitrationNeuron(BaseNeuron):
    """
    Scene-bound thought object + turn arbitration layer.

    Core rule:
      need/scene pressure emits a thought.obj; if required components are
      available, the thought can become an action candidate. If not, it goes
      into the thought drawer with demand-based expiration and periodic/event
      rechecks. Priority is a default ladder scaled by learned circumstance.

    This neuron is intentionally conservative: it does not execute actions.
    It publishes `thought/action_candidate` and `thought/turn_state` for later
    action selectors, and keeps unready thoughts out of the hot loop.
    """

    DEFAULT_BASE_PRIORITIES = {
        "power": 1.00,
        "safety": 0.96,
        "maintenance": 0.90,
        "human_uplift": 0.74,
        "task": 0.68,
        "self_need": 0.60,
        "social": 0.56,
        "novelty": 0.44,
        "expression": 0.34,
        "idle_thought": 0.18,
    }

    DEFAULT_TTLS = {
        "power": 900.0,
        "safety": 600.0,
        "maintenance": 1200.0,
        "human_uplift": 900.0,
        "task": 900.0,
        "self_need": 600.0,
        "social": 420.0,
        "novelty": 240.0,
        "expression": 180.0,
        "idle_thought": 120.0,
    }

    MAX_DRAWER = 24

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        enabled = bool(await ctx.get_kv("thought:turn:enabled", True))
        if not enabled:
            return []

        now = time.time()
        drawer = await self._load_drawer(ctx, now)
        outputs: list[Event] = []

        if service_tick_is_for(event, "cognition"):
            # Full-rate cognition service is housekeeping only. It may advance
            # drawer expiry/recheck state in RAM, but it must never emit a
            # thought/turn_state event or replace the last meaningful turn state.
            due = self._due_drawer_thoughts(drawer, now)
            available = await self._available_components(ctx) if due else {}
            for thought in due:
                self._recheck_thought(thought, available, now)
            drawer = self._prune_and_rank(drawer, now)
            await self._save_drawer_housekeeping(ctx, drawer, now)
            return []

        if event.topic == "thought/drawer_recheck":
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            target_ids = {str(x) for x in payload.get("thought_ids", []) or []}
            target_needs = {str(x) for x in payload.get("needs", []) or []}
            force_all = bool(payload.get("force_all", False)) or not target_ids and not target_needs
            available = await self._available_components(ctx)
            became_ready: list[dict[str, Any]] = []
            for thought in drawer:
                if thought.get("status") not in {"drawer_waiting", "active", "ready"}:
                    continue
                if not force_all:
                    thought_id = str(thought.get("id", ""))
                    need = str(thought.get("need", ""))
                    family = str(thought.get("family", ""))
                    if thought_id not in target_ids and need not in target_needs and family not in target_needs:
                        continue
                previous_status = str(thought.get("status", ""))
                self._recheck_thought(thought, available, now)
                if previous_status != "ready" and thought.get("status") == "ready":
                    became_ready.append(thought)
            drawer = self._prune_and_rank(drawer, now)
            state = self._turn_state(drawer, now, reason=str(payload.get("reason") or "drawer_recheck"))
            await self._save_drawer(ctx, drawer, state)
            for thought in became_ready[:4]:
                outputs.append(self._action_candidate_event(event, thought))
            outputs.append(self._state_event(event, state))
            return outputs

        if event.topic in ("event/relief/power", "reinforcement/feedback", "control/reinforce"):
            await self._learn_from_feedback(ctx, event, now)
            if event.topic == "event/relief/power":
                drawer = self._resolve_need(drawer, "power", now, reason="relief")
                state = self._turn_state(drawer, now, reason="feedback")
                await self._save_drawer(ctx, drawer, state)
                outputs.append(self._state_event(event, state))
            return outputs

        available = await self._available_components(ctx)
        thought = await self._thought_from_event(event, ctx, available, now)
        if thought is None:
            return []

        self._recheck_thought(thought, available, now)
        drawer = self._upsert_drawer(drawer, thought, now)
        drawer = self._prune_and_rank(drawer, now)
        state = self._turn_state(drawer, now, reason="new_thought")
        await self._save_drawer(ctx, drawer, state)

        outputs.append(self._object_event(event, thought))
        if thought.get("status") == "ready":
            outputs.append(self._action_candidate_event(event, thought))
        outputs.append(self._state_event(event, state))
        return outputs

    async def _load_drawer(self, ctx, now: float) -> list[dict[str, Any]]:
        raw = await ctx.get_kv("thought:drawer", [])
        if not isinstance(raw, list):
            return []
        drawer: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            thought = dict(item)
            if self._safe_expired(thought, now):
                thought["status"] = "expired"
                thought["expired_at"] = now
            drawer.append(thought)
        return drawer

    async def _save_drawer(self, ctx, drawer: list[dict[str, Any]], state: dict[str, Any]) -> None:
        await ctx.set_kv("thought:drawer", drawer)
        await ctx.set_kv("thought:turn:last_state", state)
        ready = [t for t in drawer if t.get("status") == "ready"]
        waiting = [t for t in drawer if t.get("status") == "drawer_waiting"]
        await ctx.set_kv("thought:drawer:ready", ready[:8])
        await ctx.set_kv("thought:drawer:waiting", waiting[: self.MAX_DRAWER])

    async def _save_drawer_housekeeping(self, ctx, drawer: list[dict[str, Any]], now: float) -> None:
        await ctx.set_kv("thought:drawer", drawer)
        ready = [t for t in drawer if t.get("status") == "ready"]
        waiting = [t for t in drawer if t.get("status") == "drawer_waiting"]
        await ctx.set_kv("thought:drawer:ready", ready[:8])
        await ctx.set_kv("thought:drawer:waiting", waiting[: self.MAX_DRAWER])
        await ctx.set_kv("thought:turn:last_housekeeping_ts", now)

    async def _available_components(self, ctx) -> dict[str, bool]:
        audio_pref = str(await ctx.get_kv("speech:audio_preferred_transport", "none") or "none").lower()
        power_state = await ctx.get_kv("power:state", {})
        if not isinstance(power_state, Mapping):
            power_state = {}
        interaction = await ctx.get_kv("interaction:last_input", {})
        if not isinstance(interaction, Mapping):
            interaction = {}
        last_user_ts = _safe_float(interaction.get("ts", 0.0), 0.0)
        now = time.time()
        user_recent_s = _safe_float(await ctx.get_kv("thought:turn:user_recent_s", 90.0), 90.0)

        components = {
            "textual_available": bool(await ctx.get_kv("outlet:textual_available", True)),
            "audio_available": bool(await ctx.get_kv("outlet:audio_available", audio_pref != "none")),
            "motion_available": bool(await ctx.get_kv("outlet:motion_available", False)),
            "speech_allowed": not bool(await ctx.get_kv("power:sleep", False)),
            "not_charging": not bool(power_state.get("charging", False)),
            "not_sleeping": not bool(power_state.get("sleep", False)),
            "user_recent": bool(last_user_ts > 0 and (now - last_user_ts) <= user_recent_s),
            "safety_clear": True,
        }

        # The capability-circulation layer is a passive glue/lymph system.
        # It may know about equipment/organ availability or redundant fallback
        # routes that this neuron should not hard-own. Merge it in when present.
        circulated = await ctx.get_kv("capability:available_components", {})
        if isinstance(circulated, Mapping):
            for key, value in circulated.items():
                components[str(key)] = bool(value)

        aliases = await ctx.get_kv("capability:alias_available", {})
        if isinstance(aliases, Mapping):
            for key, value in aliases.items():
                components[str(key)] = bool(value)

        return components

    async def _thought_from_event(
        self,
        event: Event,
        ctx,
        available: dict[str, bool],
        now: float,
    ) -> dict[str, Any] | None:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        meta = event.meta if isinstance(event.meta, Mapping) else {}
        need = ""
        family = "idle_thought"
        content = ""
        pull = 0.0
        importance = 0.0
        required: list[str] = []
        route: dict[str, Any] = {}
        scene_ref = await self._scene_ref(ctx, payload)

        if event.topic == "drive/power_request":
            pressure = payload.get("pressure", {}) if isinstance(payload.get("pressure"), Mapping) else {}
            vector = payload.get("vector", {}) if isinstance(payload.get("vector"), Mapping) else {}
            need = "power"
            family = "power"
            pull = _clamp(_safe_float(pressure.get("urgency", 0.0), 0.0))
            importance = max(0.70, pull)
            content = str(payload.get("thought_text") or payload.get("message") or "Power need is requesting relief.")
            outlet = str(vector.get("outlet") or payload.get("outlet") or "textual")
            required = self._components_for_outlet(outlet) + ["not_charging", "not_sleeping"]
            route = {"outlet": outlet, "style": payload.get("style"), "vector": vector}

        elif event.topic == "thought/internal":
            text = str(payload.get("text", "") or "").strip()
            if not text:
                return None
            need = _norm_token(payload.get("source_need") or meta.get("need") or payload.get("kind") or "thought", "thought")
            family = self._family_for_need(need, text, payload, meta)
            urgency = _safe_float(payload.get("urgency", payload.get("pull", 0.0)), 0.0)
            pull = _clamp(urgency if urgency > 0 else self._text_pull(text))
            importance = _clamp(_safe_float(payload.get("importance", pull), pull))
            content = text
            required = self._required_components_for_family(family)
            route = {"outlet": "internal", "source_kind": payload.get("kind") or meta.get("kind")}

        elif event.topic in {"drive:boredom", "drive:social_experimentation", "curiosity/adjust"}:
            family = "novelty"
            need = "curiosity" if event.topic == "curiosity/adjust" else "novelty"
            pull = _clamp(_safe_float(payload.get("level", payload.get("pressure", payload.get("boost", 0.0))), 0.0))
            if pull <= 0.0:
                return None
            importance = min(0.55, 0.20 + pull)
            content = str(payload.get("text") or "Explore a safe novelty route if higher needs are stable.")
            required = ["safety_clear"]
            route = {"outlet": "internal", "source_drive": event.topic}

        elif event.topic == "drive:social_interaction":
            family = "social"
            need = "social"
            pull = _clamp(_safe_float(payload.get("level", payload.get("pressure", 0.0)), 0.0))
            if pull <= 0.0:
                return None
            importance = min(0.65, 0.25 + pull)
            content = str(payload.get("text") or "Social contact need is active.")
            required = ["textual_available", "speech_allowed", "user_recent"]
            route = {"outlet": "textual", "source_drive": event.topic}

        elif event.topic == "vision/object_delta":
            if not bool(payload.get("memory_candidate", False)):
                return None
            family = "safety" if self._payload_has_safety(payload) else "task"
            need = family
            pull = 0.70 if family == "safety" else 0.45
            importance = 0.85 if family == "safety" else 0.55
            content = str(payload.get("text") or "Vision detected a meaningful object/spatial change.")
            scene_ref = str(payload.get("scene_ref") or scene_ref)
            required = ["safety_clear"]
            route = {"outlet": "internal", "source_delta": "vision/object_delta"}

        elif event.topic == "affect/salience":
            family = "self_need"
            need = "salience"
            pull = _clamp(_safe_float(payload.get("salience", payload.get("level", 0.0)), 0.0))
            if pull <= 0.0:
                return None
            importance = pull
            content = str(payload.get("text") or "A salient internal state needs evaluation.")
            required = ["safety_clear"]
            route = {"outlet": "internal"}

        else:
            return None

        family = family if family in self.DEFAULT_BASE_PRIORITIES else "idle_thought"
        wans = await ctx.get_kv("pdna:wans", {})
        if isinstance(wans, Mapping):
            preferred = wans.get("preferred_routes", {})
            if isinstance(preferred, Mapping) and family in preferred:
                route["wans_preferred_routes"] = list(preferred.get(family, []) or [])[:6]
        learned = await self._learned_modifier(ctx, family, scene_ref)
        base_priority = await self._base_priority(ctx, family)
        priority = self._score(base_priority, pull, importance, learned)
        ttl = await self._ttl(ctx, family, pull, importance)
        recheck = await self._recheck_interval(ctx, family, priority, pull)
        thought_id = self._thought_id(family, need, content, scene_ref)

        thought = {
            "schema": "thought.obj.v1",
            "id": thought_id,
            "kind": "need_expression",
            "family": family,
            "need": need,
            "content": content[:500],
            "scene_ref": scene_ref,
            "source_topic": event.topic,
            "source": event.source,
            "pull": round(pull, 4),
            "importance": round(importance, 4),
            "base_priority": round(base_priority, 4),
            "learned_modifier": round(learned, 4),
            "priority_score": round(priority, 4),
            "required_components": sorted(set(required)),
            "available_components": sorted(k for k, v in available.items() if v),
            "missing_components": [],
            "route": route,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl,
            "check_after": now + recheck,
            "attempt_count": int(payload.get("attempt_count", 0) or 0),
            "memory_candidate": False,
            "learning_rule": "thought_to_action_feedback_memory",
        }
        return thought

    def _recheck_thought(self, thought: dict[str, Any], available: dict[str, bool], now: float) -> None:
        required = list(thought.get("required_components", []) or [])
        missing = [name for name in required if not available.get(name, False)]
        thought["available_components"] = sorted(k for k, v in available.items() if v)
        thought["missing_components"] = missing
        thought["updated_at"] = now
        if self._safe_expired(thought, now):
            thought["status"] = "expired"
            thought["expired_at"] = now
        elif missing:
            thought["status"] = "drawer_waiting"
        else:
            thought["status"] = "ready"

    def _due_drawer_thoughts(self, drawer: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        due = []
        for thought in drawer:
            if thought.get("status") not in {"drawer_waiting", "active", "ready"}:
                continue
            check_after = _safe_float(thought.get("check_after", 0.0), 0.0)
            if check_after <= now:
                due.append(thought)
                family = str(thought.get("family") or "idle_thought")
                pull = _safe_float(thought.get("pull", 0.0), 0.0)
                priority = _safe_float(thought.get("priority_score", 0.0), 0.0)
                thought["check_after"] = now + self._local_recheck_interval(family, priority, pull)
        return due

    def _upsert_drawer(self, drawer: list[dict[str, Any]], thought: dict[str, Any], now: float) -> list[dict[str, Any]]:
        for idx, existing in enumerate(drawer):
            if str(existing.get("id")) == str(thought.get("id")):
                merged = dict(existing)
                merged.update(thought)
                merged["created_at"] = existing.get("created_at", thought.get("created_at", now))
                merged["attempt_count"] = max(int(existing.get("attempt_count", 0) or 0), int(thought.get("attempt_count", 0) or 0))
                drawer[idx] = merged
                return drawer
        drawer.append(thought)
        return drawer

    def _prune_and_rank(self, drawer: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        active = [t for t in drawer if t.get("status") != "expired" and not self._safe_expired(t, now)]
        expired = [t for t in drawer if t.get("status") == "expired" or self._safe_expired(t, now)]
        active.sort(key=lambda t: _safe_float(t.get("priority_score", 0.0), 0.0), reverse=True)
        expired = expired[-4:]
        for t in expired:
            t["status"] = "expired"
        return active[: self.MAX_DRAWER] + expired

    def _resolve_need(self, drawer: list[dict[str, Any]], need: str, now: float, reason: str) -> list[dict[str, Any]]:
        for thought in drawer:
            if str(thought.get("need", "")) == need or str(thought.get("family", "")) == need:
                thought["status"] = "fulfilled"
                thought["fulfilled_at"] = now
                thought["fulfillment_reason"] = reason
                thought["memory_candidate"] = True
        return drawer

    def _turn_state(self, drawer: list[dict[str, Any]], now: float, reason: str) -> dict[str, Any]:
        active = [t for t in drawer if t.get("status") in {"ready", "drawer_waiting", "active"}]
        active.sort(key=lambda t: _safe_float(t.get("priority_score", 0.0), 0.0), reverse=True)
        top = active[0] if active else {}
        return {
            "schema": "thought.turn_state.v1",
            "ts": now,
            "reason": reason,
            "drawer_count": len(drawer),
            "active_count": len(active),
            "ready_count": len([t for t in active if t.get("status") == "ready"]),
            "waiting_count": len([t for t in active if t.get("status") == "drawer_waiting"]),
            "dominant_family": str(top.get("family", "") or ""),
            "dominant_need": str(top.get("need", "") or ""),
            "dominant_status": str(top.get("status", "") or ""),
            "dominant_priority": round(_safe_float(top.get("priority_score", 0.0), 0.0), 4) if top else 0.0,
            "dominant_thought_id": str(top.get("id", "") or ""),
            "top_thoughts": [self._thin_thought(t) for t in active[:5]],
        }

    async def _learn_from_feedback(self, ctx, event: Event, now: float) -> None:
        family = ""
        delta = 0.0
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        meta = event.meta if isinstance(event.meta, Mapping) else {}

        if event.topic == "event/relief/power":
            family = "power"
            delta = _clamp(_safe_float(payload.get("delta_pct", 0.0), 0.0) / 10.0)
        else:
            family = _norm_token(payload.get("family") or payload.get("need") or meta.get("need") or meta.get("family"), "")
            delta = _safe_float(payload.get("delta", payload.get("score", 0.0)), 0.0)
            delta = _clamp(delta, -1.0, 1.0)

        if not family:
            return

        mods = await ctx.get_kv("thought:priority:learned_modifiers", {})
        if not isinstance(mods, dict):
            mods = {}
        current = _safe_float(mods.get(family, 0.0), 0.0)
        rate = _safe_float(await ctx.get_kv("thought:turn:learning_rate", 0.05), 0.05)
        mods[family] = round(_clamp(current + (delta * rate), -0.35, 0.35), 4)
        await ctx.set_kv("thought:priority:learned_modifiers", mods)
        await ctx.set_kv("thought:priority:last_learning", {"family": family, "delta": delta, "ts": now, "source_topic": event.topic})

    async def _scene_ref(self, ctx, payload: Mapping[str, Any]) -> str:
        if payload.get("scene_ref"):
            return str(payload.get("scene_ref"))[:120]
        scene = await ctx.get_kv("scene:current", {})
        if isinstance(scene, Mapping):
            return str(scene.get("scene_ref") or scene.get("id") or scene.get("place") or "scene:unknown")[:120]
        return "scene:unknown"

    async def _ddna_mods(self, ctx) -> dict[str, Any]:
        pdna = await ctx.get_kv("pdna:profile", None)
        mods = await ctx.get_kv("drive:ddna_modulators", None)
        if not isinstance(mods, dict) or not mods:
            mods = derive_ddna_modulators(pdna)
            await ctx.set_kv("drive:ddna_modulators", mods)
        return dict(mods or {})

    def _family_gain(self, family: str, mods: Mapping[str, Any]) -> float:
        if family == "human_uplift":
            return _safe_float(mods.get("human_uplift_gain"), 1.0)
        if family == "novelty":
            return _safe_float(mods.get("novelty_gain"), 1.0)
        if family == "expression":
            return _safe_float(mods.get("expression_bias"), 1.0)
        if family == "social":
            return _safe_float(mods.get("social_gain"), 1.0)
        if family in {"task", "maintenance"}:
            return _safe_float(mods.get("task_continuity_gain"), 1.0)
        if family == "safety":
            return _safe_float(mods.get("action_gate_strictness"), 1.0)
        if family == "idle_thought":
            return _safe_float(mods.get("thought_momentum_gain"), 1.0)
        return 1.0

    async def _base_priority(self, ctx, family: str) -> float:
        raw = await ctx.get_kv("thought:priority:base", {})
        base = self.DEFAULT_BASE_PRIORITIES.get(family, 0.18)
        if isinstance(raw, Mapping) and family in raw:
            base = _safe_float(raw.get(family), base)
        mods = await self._ddna_mods(ctx)
        return _clamp(base * max(0.25, min(2.0, self._family_gain(family, mods))))

    async def _learned_modifier(self, ctx, family: str, scene_ref: str) -> float:
        mods = await ctx.get_kv("thought:priority:learned_modifiers", {})
        if not isinstance(mods, Mapping):
            return 0.0
        family_mod = _safe_float(mods.get(family, 0.0), 0.0)
        scene_mod = 0.0
        scene_map = mods.get("scenes", {})
        if isinstance(scene_map, Mapping):
            scene_entry = scene_map.get(scene_ref, {})
            if isinstance(scene_entry, Mapping):
                scene_mod = _safe_float(scene_entry.get(family, 0.0), 0.0)
        return _clamp(family_mod + scene_mod, -0.35, 0.35)

    async def _ttl(self, ctx, family: str, pull: float, importance: float) -> float:
        raw = await ctx.get_kv("thought:turn:ttl_s", {})
        base = self.DEFAULT_TTLS.get(family, 180.0)
        if isinstance(raw, Mapping) and family in raw:
            base = _safe_float(raw.get(family), base)
        mods = await self._ddna_mods(ctx)
        persistence = max(0.35, min(2.0, _safe_float(mods.get("drawer_persistence_gain"), 1.0)))
        # Strong pull/importance thoughts live longer, capped to avoid drawer fossils.
        return max(30.0, min(3600.0, base * persistence * (0.75 + (pull * 0.45) + (importance * 0.35))))

    async def _recheck_interval(self, ctx, family: str, priority: float, pull: float) -> float:
        raw = await ctx.get_kv("thought:turn:recheck_s", {})
        if isinstance(raw, Mapping) and family in raw:
            base = _safe_float(raw.get(family), 30.0)
        else:
            base = _safe_float(await ctx.get_kv("thought:turn:base_recheck_s", 30.0), 30.0)
        mods = await self._ddna_mods(ctx)
        completion_bias = max(0.35, min(2.0, _safe_float(mods.get("thought_completion_bias"), 1.0)))
        return max(5.0, min(300.0, base / max(0.35, 0.5 + (priority * completion_bias) + (pull * 0.5))))

    def _local_recheck_interval(self, family: str, priority: float, pull: float) -> float:
        base = 30.0 if family not in {"power", "safety"} else 12.0
        return max(5.0, min(300.0, base / max(0.35, 0.5 + priority + (pull * 0.5))))

    def _score(self, base: float, pull: float, importance: float, learned: float) -> float:
        return _clamp((base * 0.42) + (pull * 0.30) + (importance * 0.20) + ((learned + 0.35) / 0.70 * 0.08))

    def _family_for_need(self, need: str, text: str, payload: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
        explicit = _norm_token(payload.get("family") or meta.get("family"), "")
        if explicit in self.DEFAULT_BASE_PRIORITIES:
            return explicit
        need = _norm_token(need, "thought")
        lowered = text.lower()
        if need in {"power", "battery", "charge"} or any(w in lowered for w in ("battery", "charge", "power is")):
            return "power"
        if need in {"safety", "hazard", "risk"} or any(w in lowered for w in ("danger", "risk", "unsafe", "hazard")):
            return "safety"
        if need in {"maintenance", "repair", "fault"} or any(w in lowered for w in ("repair", "fault", "maintenance", "damage")):
            return "maintenance"
        if need in {"curiosity", "novelty", "boredom"}:
            return "novelty"
        if need in {"social", "interaction"}:
            return "social"
        if need in {"expression", "speech", "reply"}:
            return "expression"
        return "idle_thought"

    def _required_components_for_family(self, family: str) -> list[str]:
        if family == "power":
            return ["textual_available", "speech_allowed", "not_charging", "not_sleeping"]
        if family == "safety":
            return ["safety_clear"]
        if family == "maintenance":
            return ["safety_clear"]
        if family in {"social", "expression"}:
            return ["textual_available", "speech_allowed"]
        if family == "novelty":
            return ["safety_clear"]
        return []

    def _components_for_outlet(self, outlet: str) -> list[str]:
        outlet = outlet.strip().lower()
        if outlet == "audio":
            return ["audio_available", "speech_allowed"]
        if outlet == "motion":
            return ["motion_available", "safety_clear"]
        if outlet in {"text", "textual", "speech"}:
            return ["textual_available", "speech_allowed"]
        return []

    def _text_pull(self, text: str) -> float:
        lowered = text.lower()
        if any(w in lowered for w in ("critical", "urgent", "danger", "need")):
            return 0.62
        if any(w in lowered for w in ("should", "maybe", "could", "might")):
            return 0.32
        return 0.18

    def _payload_has_safety(self, payload: Mapping[str, Any]) -> bool:
        text = str(payload.get("text", "") or "").lower()
        if any(w in text for w in ("hazard", "fire", "smoke", "danger", "unsafe", "intruder")):
            return True
        for delta in payload.get("deltas", []) or []:
            if isinstance(delta, Mapping):
                q = delta.get("quorum", {}) if isinstance(delta.get("quorum"), Mapping) else {}
                if "safety" in (q.get("voters", []) or []):
                    return True
        return False

    def _thought_id(self, family: str, need: str, content: str, scene_ref: str) -> str:
        raw = f"{family}|{need}|{scene_ref}|{content[:180]}"
        return "thought:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _safe_expired(self, thought: Mapping[str, Any], now: float) -> bool:
        expires_at = _safe_float(thought.get("expires_at", now + 1.0), now + 1.0)
        return expires_at <= now

    def _thin_thought(self, thought: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": thought.get("id"),
            "family": thought.get("family"),
            "need": thought.get("need"),
            "status": thought.get("status"),
            "priority_score": thought.get("priority_score"),
            "pull": thought.get("pull"),
            "missing_components": list(thought.get("missing_components", []) or []),
            "content": str(thought.get("content", "") or "")[:160],
        }

    def _object_event(self, source_event: Event, thought: dict[str, Any]) -> Event:
        return Event(
            topic="thought/object",
            payload=thought,
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={
                "kind": "thought_object",
                "family": thought.get("family"),
                "need": thought.get("need"),
                "status": thought.get("status"),
                "store_in_memory": False,
                "cognitive_visible": False,
            },
        )

    def _action_candidate_event(self, source_event: Event, thought: dict[str, Any]) -> Event:
        return Event(
            topic="thought/action_candidate",
            payload={
                "thought": thought,
                "route": thought.get("route", {}),
                "priority_score": thought.get("priority_score", 0.0),
                "rule": "tools_available_thought_may_become_action",
            },
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={
                "kind": "thought_action_candidate",
                "family": thought.get("family"),
                "need": thought.get("need"),
                "store_in_memory": False,
                "cognitive_visible": False,
            },
        )

    def _state_event(self, source_event: Event, state: dict[str, Any]) -> Event:
        return Event(
            topic="thought/turn_state",
            payload=state,
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={
                "kind": "thought_turn_state",
                "store_in_memory": False,
                "cognitive_visible": False,
            },
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            SERVICE_TOPIC,
            "thought/internal",
            "drive/power_request",
            "drive:boredom",
            "drive:social_interaction",
            "drive:social_experimentation",
            "curiosity/adjust",
            "vision/object_delta",
            "affect/salience",
            "reinforcement/feedback",
            "control/reinforce",
            "event/relief/power",
            "thought/drawer_recheck",
        ],
        output_topics=["thought/object", "thought/action_candidate", "thought/turn_state"],
        priority=7,
        cooldown_sec=0.0,
    )
    yield ThoughtTurnArbitrationNeuron(cfg)
