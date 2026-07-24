from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.heartbeat_stream import PRIMARY_HEARTBEAT_TOPIC, is_heartbeat_event

NEURON_NAME = Path(__file__).stem


READY_STATUSES = {"ready", "available", "online", "idle", "ok", "clear", "present", "enabled", "active"}
BLOCKED_STATUSES = {"blocked", "missing", "offline", "busy", "crashed", "stalled", "disabled", "unavailable", "error"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "ready", "available", "online", "idle", "ok", "clear", "present", "enabled", "active"}:
            return True
        if lowered in {"0", "false", "no", "off", "blocked", "missing", "offline", "busy", "crashed", "stalled", "disabled", "unavailable", "error"}:
            return False
    return bool(default)


def _norm_name(value: Any, fallback: str = "") -> str:
    text = str(value or fallback or "").strip().lower()
    text = text.replace("/", "_").replace(":", "_").replace("-", "_")
    text = "_".join(part for part in text.split() if part)
    return text[:96]


def _status_available(payload: Mapping[str, Any]) -> bool:
    if "available" in payload:
        return _safe_bool(payload.get("available"), False)
    if "ready" in payload:
        return _safe_bool(payload.get("ready"), False)
    if "enabled" in payload:
        return _safe_bool(payload.get("enabled"), False)
    status = _norm_name(payload.get("status") or payload.get("state") or payload.get("mode"), "")
    if status in READY_STATUSES:
        return True
    if status in BLOCKED_STATUSES:
        return False
    return False


class CapabilityCirculationNeuron(BaseNeuron):
    """
    Passive capability/readiness glue layer.

    This is intentionally not a controller. It acts like a small lymphatic
    system for MB: it circulates component availability, redundant fallback
    routes, action requirements, and readiness summaries so thought/action
    systems do not each need to own every equipment/status rule.

    It emits no speech. Outputs are state/readiness/recheck events only.
    """

    DEFAULT_COMPONENT_TTL_S = 60.0

    DEFAULT_FALLBACKS: dict[str, list[str]] = {
        # Communication redundancy: text and audio can often stand in for each other.
        "textual_available": ["textual_available", "audio_available"],
        "audio_available": ["audio_available", "textual_available"],
        # Motion can be fulfilled directly later, or by requesting/receiving user assist.
        "motion_available": ["motion_available", "user_assist_available"],
        # Spatial perception should degrade gracefully.
        "lidar_available": ["lidar_available", "depth_available", "vision_available"],
        "depth_available": ["depth_available", "lidar_available", "vision_available"],
        "vision_available": ["vision_available"],
        # Safety has deliberate redundancy; any clear safety path can satisfy basic clearance.
        "safety_clear": ["safety_clear", "guardian_clear", "hazard_clear"],
        # Sleep/speech are usually strict, but aliases let external organs replace the source.
        "speech_allowed": ["speech_allowed", "expression_allowed"],
        "not_sleeping": ["not_sleeping", "awake"],
        "not_charging": ["not_charging"],
    }

    ROUTE_REQUIREMENTS: dict[str, list[str]] = {
        "textual": ["textual_available", "speech_allowed"],
        "text": ["textual_available", "speech_allowed"],
        "speech": ["textual_available", "speech_allowed"],
        "audio": ["audio_available", "speech_allowed"],
        "motion": ["motion_available", "safety_clear", "not_sleeping"],
        "internal": [],
        "memory": [],
    }

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        enabled = bool(await ctx.get_kv("capability:circulation:enabled", True))
        if not enabled:
            return []

        now = time.time()
        changed = False
        outputs: list[Event] = []

        components = await self._load_components(ctx, now)
        components, kv_changed = await self._refresh_from_kv(ctx, components, now)
        changed = changed or kv_changed

        if event.topic in {"component/status", "equipment/status", "organ/status", "control/capability"}:
            components, event_changed = self._ingest_status_event(event, components, now)
            changed = changed or event_changed

        if event.topic == "power/state":
            components, power_changed = self._ingest_power_event(event, components, now)
            changed = changed or power_changed

        components, pruned_changed = self._prune_expired(components, now)
        changed = changed or pruned_changed

        available = self._available_map(components)
        fallbacks = await self._fallbacks(ctx)
        alias_available = self._alias_available_map(available, fallbacks)
        state = self._state_payload(components, available, alias_available, now)

        if event.topic in {"thought/object", "thought/action_candidate"}:
            readiness = await self._readiness_for_event(ctx, event, available, fallbacks, now)
            if readiness is not None:
                await ctx.set_kv("capability:last_readiness", readiness)
                if readiness.get("thought_id"):
                    await ctx.set_kv(f"capability:readiness:{readiness['thought_id']}", readiness)
                outputs.append(self._readiness_event(event, readiness))

        if changed or is_heartbeat_event(event):
            await self._save_components(ctx, components, available, alias_available, state)
            outputs.append(self._state_event(event, state))
            if changed:
                outputs.append(self._drawer_recheck_event(event, state, reason="capability_changed"))
        else:
            await self._save_components(ctx, components, available, alias_available, state)

        return outputs

    async def _load_components(self, ctx, now: float) -> dict[str, dict[str, Any]]:
        raw = await ctx.get_kv("capability:components", {})
        if not isinstance(raw, Mapping):
            raw = {}
        components: dict[str, dict[str, Any]] = {}
        for name, value in raw.items():
            key = _norm_name(name)
            if not key:
                continue
            if isinstance(value, Mapping):
                entry = dict(value)
            else:
                entry = {"available": _safe_bool(value), "source": "legacy_bool"}
            entry.setdefault("name", key)
            entry.setdefault("ts", now)
            entry["available"] = _safe_bool(entry.get("available"), False)
            components[key] = entry
        return components

    async def _save_components(
        self,
        ctx,
        components: dict[str, dict[str, Any]],
        available: dict[str, bool],
        alias_available: dict[str, bool],
        state: dict[str, Any],
    ) -> None:
        await ctx.set_kv("capability:components", components)
        await ctx.set_kv("capability:available_components", available)
        await ctx.set_kv("capability:alias_available", alias_available)
        await ctx.set_kv("capability:state", state)

    async def _refresh_from_kv(
        self,
        ctx,
        components: dict[str, dict[str, Any]],
        now: float,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        changed = False
        power_state = await ctx.get_kv("power:state", {})
        if not isinstance(power_state, Mapping):
            power_state = {}
        sleeping = _safe_bool(power_state.get("sleep", await ctx.get_kv("power:sleep", False)), False)
        charging = _safe_bool(power_state.get("charging", await ctx.get_kv("power:charging", False)), False)
        pct = _safe_float(power_state.get("pct", await ctx.get_kv("power:battery_pct", 100.0)), 100.0)
        low_threshold = _safe_float(await ctx.get_kv("capability:low_power_threshold_pct", 15.0), 15.0)

        defaults = {
            "textual_available": _safe_bool(await ctx.get_kv("outlet:textual_available", True), True),
            "audio_available": _safe_bool(await ctx.get_kv("outlet:audio_available", False), False),
            "motion_available": _safe_bool(await ctx.get_kv("outlet:motion_available", False), False),
            "speech_allowed": not sleeping,
            "expression_allowed": not sleeping,
            "not_sleeping": not sleeping,
            "awake": not sleeping,
            "not_charging": not charging,
            "safety_clear": _safe_bool(await ctx.get_kv("safety:clear", True), True),
            "guardian_clear": not _safe_bool(await ctx.get_kv("safety:crisis_mode", False), False),
            "hazard_clear": not _safe_bool(await ctx.get_kv("hazard:active", False), False),
            "power_available": pct > low_threshold,
            "vision_available": _safe_bool(await ctx.get_kv("vision:available", await ctx.get_kv("camera:available", False)), False),
            "lidar_available": _safe_bool(await ctx.get_kv("lidar:available", False), False),
            "depth_available": _safe_bool(await ctx.get_kv("depth:available", False), False),
            "user_assist_available": _safe_bool(await ctx.get_kv("user:assist_available", False), False),
        }
        for name, available in defaults.items():
            prior = components.get(name)
            # Explicit organ/equipment/status events are stronger than passive KV
            # snapshots. Keep them until they expire or another explicit event
            # updates them, so the glue layer does not overwrite a live status
            # with a default on the next pass.
            if isinstance(prior, Mapping) and prior.get("source") != "kv_snapshot":
                expires_at = prior.get("expires_at")
                if expires_at is None or _safe_float(expires_at, now + 1.0) > now:
                    continue
            changed = self._set_component(
                components,
                name,
                available,
                now,
                source="kv_snapshot",
                confidence=0.75,
                expires_at=None,
            ) or changed
        return components, changed

    def _ingest_status_event(
        self,
        event: Event,
        components: dict[str, dict[str, Any]],
        now: float,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        if not payload:
            return components, False

        ttl_default = _safe_float(payload.get("ttl_s", self.DEFAULT_COMPONENT_TTL_S), self.DEFAULT_COMPONENT_TTL_S)
        changed = False

        if isinstance(payload.get("components"), Mapping):
            for name, value in payload["components"].items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("component", name)
                else:
                    item = {"component": name, "available": value}
                changed = self._set_component_from_payload(item, components, now, event.topic, ttl_default) or changed
            return components, changed

        if isinstance(payload.get("components"), list):
            for item in payload.get("components", []):
                if isinstance(item, Mapping):
                    changed = self._set_component_from_payload(dict(item), components, now, event.topic, ttl_default) or changed
            return components, changed

        changed = self._set_component_from_payload(payload, components, now, event.topic, ttl_default) or changed
        return components, changed

    def _set_component_from_payload(
        self,
        payload: Mapping[str, Any],
        components: dict[str, dict[str, Any]],
        now: float,
        source_topic: str,
        ttl_default: float,
    ) -> bool:
        name = _norm_name(
            payload.get("component")
            or payload.get("equipment")
            or payload.get("organ")
            or payload.get("name")
            or payload.get("tool")
        )
        if not name:
            return False
        available = _status_available(payload)
        ttl_s = _safe_float(payload.get("ttl_s", ttl_default), ttl_default)
        expires_at = None if ttl_s <= 0 else now + ttl_s
        confidence = _safe_float(payload.get("confidence", 0.85), 0.85)
        return self._set_component(
            components,
            name,
            available,
            now,
            source=str(payload.get("source") or source_topic),
            confidence=confidence,
            expires_at=expires_at,
            status=str(payload.get("status") or payload.get("state") or ("available" if available else "unavailable")),
            detail=payload.get("detail"),
        )

    def _ingest_power_event(
        self,
        event: Event,
        components: dict[str, dict[str, Any]],
        now: float,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        charging = _safe_bool(payload.get("charging", False), False)
        state = _norm_name(payload.get("state") or payload.get("mode"), "active")
        sleeping = state in {"sleep", "sleeping"} or _safe_bool(payload.get("sleep", False), False)
        changed = False
        changed = self._set_component(components, "not_charging", not charging, now, source="power/state", confidence=0.9, expires_at=None) or changed
        changed = self._set_component(components, "not_sleeping", not sleeping, now, source="power/state", confidence=0.9, expires_at=None) or changed
        changed = self._set_component(components, "awake", not sleeping, now, source="power/state", confidence=0.9, expires_at=None) or changed
        changed = self._set_component(components, "speech_allowed", not sleeping, now, source="power/state", confidence=0.9, expires_at=None) or changed
        return components, changed

    def _set_component(
        self,
        components: dict[str, dict[str, Any]],
        name: str,
        available: bool,
        now: float,
        *,
        source: str,
        confidence: float,
        expires_at: float | None,
        status: str | None = None,
        detail: Any = None,
    ) -> bool:
        key = _norm_name(name)
        if not key:
            return False
        prior = components.get(key, {})
        changed = bool(prior.get("available") != bool(available)) or str(prior.get("source", "")) != source
        entry = dict(prior)
        entry.update(
            {
                "name": key,
                "available": bool(available),
                "source": source,
                "confidence": max(0.0, min(1.0, confidence)),
                "ts": now,
                "expires_at": expires_at,
                "status": status or ("available" if available else "unavailable"),
            }
        )
        if detail is not None:
            entry["detail"] = detail
        components[key] = entry
        return changed

    def _prune_expired(self, components: dict[str, dict[str, Any]], now: float) -> tuple[dict[str, dict[str, Any]], bool]:
        changed = False
        for name, entry in list(components.items()):
            expires_at = entry.get("expires_at")
            if expires_at is None:
                continue
            if _safe_float(expires_at, now + 1.0) <= now and bool(entry.get("available", False)):
                entry = dict(entry)
                entry["available"] = False
                entry["status"] = "expired"
                entry["expired_at"] = now
                components[name] = entry
                changed = True
        return components, changed

    async def _fallbacks(self, ctx) -> dict[str, list[str]]:
        raw = await ctx.get_kv("capability:fallbacks", {})
        merged = {k: list(v) for k, v in self.DEFAULT_FALLBACKS.items()}
        if isinstance(raw, Mapping):
            for name, values in raw.items():
                key = _norm_name(name)
                if not key:
                    continue
                if isinstance(values, (list, tuple)):
                    merged[key] = [_norm_name(v) for v in values if _norm_name(v)]
        return merged

    def _available_map(self, components: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
        return {str(name): bool(entry.get("available", False)) for name, entry in components.items()}

    def _alias_available_map(self, available: Mapping[str, bool], fallbacks: Mapping[str, list[str]]) -> dict[str, bool]:
        alias: dict[str, bool] = {}
        for name, options in fallbacks.items():
            alias[name] = any(bool(available.get(opt, False)) for opt in options)
        return alias

    async def _readiness_for_event(
        self,
        ctx,
        event: Event,
        available: dict[str, bool],
        fallbacks: dict[str, list[str]],
        now: float,
    ) -> dict[str, Any] | None:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        if not payload:
            return None
        thought = payload.get("thought") if event.topic == "thought/action_candidate" else payload
        if not isinstance(thought, Mapping):
            return None
        required = list(thought.get("required_components", []) or [])
        route = thought.get("route", {}) if isinstance(thought.get("route", {}), Mapping) else {}
        outlet = _norm_name(route.get("outlet"), "")
        if outlet in self.ROUTE_REQUIREMENTS:
            required.extend(self.ROUTE_REQUIREMENTS[outlet])
        required = sorted(set(_norm_name(req) for req in required if _norm_name(req)))

        satisfied: list[str] = []
        missing: list[str] = []
        fallback_used: dict[str, str] = {}
        for req in required:
            ok, via = self._component_satisfied(req, available, fallbacks)
            if ok:
                satisfied.append(req)
                if via and via != req:
                    fallback_used[req] = via
            else:
                missing.append(req)

        if required:
            ready_score = len(satisfied) / len(required)
        else:
            ready_score = 1.0
        ready = not missing
        thought_id = str(thought.get("id") or payload.get("thought_id") or "")
        readiness = {
            "schema": "capability.readiness.v1",
            "ts": now,
            "source_topic": event.topic,
            "thought_id": thought_id,
            "family": thought.get("family"),
            "need": thought.get("need"),
            "status": "ready" if ready else "waiting",
            "ready": ready,
            "ready_score": round(ready_score, 4),
            "required_components": required,
            "satisfied_components": satisfied,
            "missing_components": missing,
            "fallback_used": fallback_used,
            "rule": "redundant_components_may_satisfy_requirements",
        }
        return readiness

    def _component_satisfied(
        self,
        req: str,
        available: Mapping[str, bool],
        fallbacks: Mapping[str, list[str]],
    ) -> tuple[bool, str | None]:
        if bool(available.get(req, False)):
            return True, req
        for alt in fallbacks.get(req, []):
            if bool(available.get(alt, False)):
                return True, alt
        return False, None

    def _state_payload(
        self,
        components: Mapping[str, Mapping[str, Any]],
        available: Mapping[str, bool],
        alias_available: Mapping[str, bool],
        now: float,
    ) -> dict[str, Any]:
        available_names = sorted(name for name, ok in available.items() if ok)
        unavailable_names = sorted(name for name, ok in available.items() if not ok)
        return {
            "schema": "capability.state.v1",
            "ts": now,
            "component_count": len(components),
            "available_count": len(available_names),
            "unavailable_count": len(unavailable_names),
            "available_components": available_names[:40],
            "unavailable_components": unavailable_names[:40],
            "alias_available": {k: bool(v) for k, v in sorted(alias_available.items())},
            "policy": "passive_capability_circulation_not_controller",
        }

    def _state_event(self, source_event: Event, state: dict[str, Any]) -> Event:
        return Event(
            topic="capability/state",
            payload=state,
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={"kind": "capability_state", "store_in_memory": False, "cognitive_visible": False},
        )

    def _readiness_event(self, source_event: Event, readiness: dict[str, Any]) -> Event:
        return Event(
            topic="capability/readiness",
            payload=readiness,
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={"kind": "capability_readiness", "store_in_memory": False, "cognitive_visible": False},
        )

    def _drawer_recheck_event(self, source_event: Event, state: dict[str, Any], reason: str) -> Event:
        return Event(
            topic="thought/drawer_recheck",
            payload={
                "reason": reason,
                "source_state": "capability/state",
                "available_components": list(state.get("available_components", []) or []),
                "force_all": True,
            },
            source=self.name,
            correlation_id=source_event.correlation_id,
            meta={"kind": "thought_drawer_recheck", "store_in_memory": False, "cognitive_visible": False},
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "clock/tick",
            PRIMARY_HEARTBEAT_TOPIC,
            "power/state",
            "component/status",
            "equipment/status",
            "organ/status",
            "control/capability",
            "thought/object",
            "thought/action_candidate",
        ],
        output_topics=["capability/state", "capability/readiness", "thought/drawer_recheck"],
        priority=6,
        cooldown_sec=0.0,
    )
    yield CapabilityCirculationNeuron(cfg)
