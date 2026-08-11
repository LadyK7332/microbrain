from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.utils.memdir import resolve_memdir_ctx

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("curiosity")

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Present context owns the front of mind.  Only after the user/world has been
# quiet for this long may the light probe drift through memory.
DEFAULT_IDLE_WANDER_THRESHOLD_S = 90.0

# Recent activity below this age is treated as active here-and-now context.  The
# probe remains quiet; live cognition/memory lookup should be context-driven.
DEFAULT_SETTLING_THRESHOLD_S = 20.0

# Recent KV/context anchors collected within this window can bias idle wander so
# it drifts from recent life instead of pure random dictionary debris.
DEFAULT_CURRENT_WINDOW_S = 600.0

# Minimum weighted candidate score required before an idle memory probe is
# allowed to surface into thought/probe.  This is intentionally low by default;
# the runtime dashboard exposes it as the main calibration knob.
DEFAULT_IDLE_CANDIDATE_THRESHOLD = 0.18

# Idle probes are quiet by default.  Speech/release systems may later use this
# as a separate threshold; this neuron only reports it for inspection/tuning.
DEFAULT_SPEAK_THRESHOLD = 0.86

# How far memory wander may drift from current anchors.  v1 uses this as a soft
# weighting scalar rather than graph traversal depth.
DEFAULT_MEMORY_DRIFT_RADIUS = 2

DEFAULT_SCENE_DELTA_ACTIVITY_THRESHOLD = 0.18


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class LightProbeNeuron(BaseNeuron):
    """
    Keeps weak-but-interesting memory cells alive during the day, then lets
    sleep/charge windows run pruning and promotion.

    Daytime:
      - stays quiet while present-facing context is active
      - monitors recent KV/context anchors
      - after an idle threshold, allows low-pressure memory wandering
      - emits quiet, origin-tagged thought/probe events for introspection

    Sleep/charge:
      - no probing
      - runs lifecycle maintenance (promotion/pruning)
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._mem_cells: MemCellStore | None = None

    async def _ensure_store(self, ctx) -> MemCellStore | None:
        if self._mem_cells is not None:
            return self._mem_cells

        shared = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(shared, MemCellStore):
            self._mem_cells = shared
            return self._mem_cells

        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        self._mem_cells = MemCellStore(memdir)
        await ctx.set_kv("memory:mem_cell_store", self._mem_cells)
        return self._mem_cells

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != SERVICE_TOPIC:
            return []

        store = await self._ensure_store(ctx)
        if store is None:
            return []

        now_ts = time.time()
        sleep_mode = bool(await ctx.get_kv("power:sleep", False))
        entropy_allowed = bool(await ctx.get_kv("entropy:allowed", False))

        if sleep_mode or entropy_allowed:
            maintenance_every_s = _safe_float(await ctx.get_kv("probe:maintenance_every_s", 1800.0), 1800.0)
            last_maint = _safe_float(await self.load_state(ctx, "last_maint_ts", 0.0), 0.0)
            if last_maint and (now_ts - last_maint) < maintenance_every_s:
                return []

            retention = {
                "now": _safe_float(await ctx.get_kv("mem_cell:now_hours", 36.0), 36.0),
                "short": _safe_float(await ctx.get_kv("mem_cell:short_hours", 72.0), 72.0),
                "long": _safe_float(await ctx.get_kv("mem_cell:long_hours", 96.0), 96.0),
                "learned": _safe_float(await ctx.get_kv("mem_cell:learned_hours", 336.0), 336.0),
            }
            stats = store.maintain_lifecycle(retention_hours=retention)
            stats["ts"] = now_ts
            await ctx.set_kv("probe:last_maintenance", stats)
            await self.save_state(ctx, "last_maint_ts", now_ts)
            return []

        enabled = bool(await ctx.get_kv("probe:enabled", True))
        wander_enabled = bool(await ctx.get_kv("probe:idle_wander_enabled", True))
        if not enabled or not wander_enabled:
            await self._publish_state(ctx, now_ts=now_ts, mode="disabled", origin="disabled", blocked_reason="probe_disabled")
            return []

        context = await self._read_recent_context(ctx, now_ts)
        mode = str(context.get("mode") or "active")
        blocked_reason = str(context.get("blocked_reason") or "")
        if mode in {"active", "settling", "background_blocked"}:
            await self._publish_state(ctx, now_ts=now_ts, **context)
            return []

        probe_every_s = _safe_float(await ctx.get_kv("probe:every_s", 300.0), 300.0)
        last_probe = _safe_float(await self.load_state(ctx, "last_probe_ts", 0.0), 0.0)
        if last_probe and (now_ts - last_probe) < probe_every_s:
            await self._publish_state(
                ctx,
                now_ts=now_ts,
                **{**context, "blocked_reason": "probe_cooldown", "cooldown_remaining_s": round(max(0.0, probe_every_s - (now_ts - last_probe)), 3)},
            )
            return []

        candidate_limit = _safe_int(await ctx.get_kv("probe:max_idle_candidates", 24), 24)
        candidates = store.probe_candidates(limit=max(1, candidate_limit), tiers=("now", "short", "long"))
        if not candidates:
            await self._publish_state(ctx, now_ts=now_ts, **{**context, "blocked_reason": "no_candidates"})
            return []

        scored = self._score_candidates(candidates, context)
        if not scored:
            await self._publish_state(ctx, now_ts=now_ts, **{**context, "blocked_reason": "no_scored_candidates"})
            return []

        threshold = _clamp01(_safe_float(await ctx.get_kv("probe:idle_candidate_threshold", DEFAULT_IDLE_CANDIDATE_THRESHOLD), DEFAULT_IDLE_CANDIDATE_THRESHOLD))
        best_score = float(scored[0][0])
        if best_score < threshold:
            await self._publish_state(
                ctx,
                now_ts=now_ts,
                **{**context, "blocked_reason": "below_idle_candidate_threshold", "best_score": round(best_score, 4), "threshold": threshold},
            )
            return []

        # Randomized, but biased: avoid making the same top candidate a hard loop.
        top = scored[: min(len(scored), max(3, min(8, candidate_limit)))]
        row = random.choices([r for _s, r in top], weights=[max(0.001, s) for s, _r in top], k=1)[0]
        updated = store.bump_cell(
            str(row.get("id", "") or ""),
            activation_delta=_safe_float(await ctx.get_kv("probe:activation_delta", 0.03), 0.03),
            promotion_delta=_safe_float(await ctx.get_kv("probe:promotion_delta", 0.01), 0.01),
        )
        await self.save_state(ctx, "last_probe_ts", now_ts)

        selected = updated or row
        info: Dict[str, Any] = {
            "cell_id": str(selected.get("id", "") or ""),
            "tier": str(selected.get("tier", "") or ""),
            "kind": str(selected.get("kind", "") or ""),
            "anchor": str(((selected.get("anchor", {}) or {}).get("ref", "") or ""))[:120],
            "activation": float(selected.get("activation", 0.0) or 0.0),
            "origin": "idle_wander",
            "mode": mode,
            "score": round(best_score, 4),
            "threshold": threshold,
            "speak_threshold": _clamp01(_safe_float(await ctx.get_kv("probe:speak_threshold", DEFAULT_SPEAK_THRESHOLD), DEFAULT_SPEAK_THRESHOLD)),
            "context": {
                "activity_age_s": context.get("activity_age_s"),
                "anchors": context.get("anchors", []),
                "blocked_reason": blocked_reason,
            },
            "ts": now_ts,
        }
        await ctx.set_kv("probe:last", info)
        await self._publish_state(ctx, now_ts=now_ts, **{**context, "last_probe": info, "blocked_reason": ""})

        return [
            Event(
                topic="thought/probe",
                payload={
                    "text": info["anchor"],
                    "cell_id": info["cell_id"],
                    "origin": info["origin"],
                    "mode": mode,
                    "score": info["score"],
                    "quiet": True,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "channel": "thought",
                    "kind": "light_probe",
                    "origin": info["origin"],
                    "mode": mode,
                    "quiet": True,
                    "store_in_memory": False,
                    "cognitive_visible": True,
                },
            )
        ]

    async def _read_recent_context(self, ctx, now_ts: float) -> Dict[str, Any]:
        current_window_s = max(1.0, _safe_float(await ctx.get_kv("probe:current_window_s", DEFAULT_CURRENT_WINDOW_S), DEFAULT_CURRENT_WINDOW_S))
        settling_s = max(0.0, _safe_float(await ctx.get_kv("probe:settling_threshold_s", DEFAULT_SETTLING_THRESHOLD_S), DEFAULT_SETTLING_THRESHOLD_S))
        idle_s = max(settling_s, _safe_float(await ctx.get_kv("probe:idle_wander_threshold_s", DEFAULT_IDLE_WANDER_THRESHOLD_S), DEFAULT_IDLE_WANDER_THRESHOLD_S))
        scene_delta_threshold = _clamp01(_safe_float(await ctx.get_kv("probe:scene_delta_activity_threshold", DEFAULT_SCENE_DELTA_ACTIVITY_THRESHOLD), DEFAULT_SCENE_DELTA_ACTIVITY_THRESHOLD))
        block_slearn = bool(await ctx.get_kv("probe:block_during_slearn_enabled", True))
        block_read = bool(await ctx.get_kv("probe:block_during_read_enabled", True))

        anchors: list[str] = []
        activity_ages: list[float] = []
        blockers: list[str] = []

        interaction = await ctx.get_kv("interaction:last_input", {})
        if isinstance(interaction, Mapping):
            last_input_ts = _safe_float(interaction.get("ts", 0.0), 0.0)
            if last_input_ts > 0.0:
                age = max(0.0, now_ts - last_input_ts)
                activity_ages.append(age)
                if age <= current_window_s:
                    text = str(interaction.get("text") or "").strip()
                    if text:
                        anchors.append(text[:120])

        visual_anchor = await ctx.get_kv("vision:attention_anchor", None)
        if isinstance(visual_anchor, Mapping):
            expires_at = _safe_float(visual_anchor.get("expires_at", 0.0), 0.0)
            selected_at = _safe_float(visual_anchor.get("selected_at", 0.0), 0.0)
            if expires_at > now_ts:
                activity_ages.append(max(0.0, now_ts - selected_at) if selected_at > 0.0 else 0.0)
                anchors.append("visual:" + str(visual_anchor.get("track_id") or visual_anchor.get("label_hint") or "selected_object"))

        scene_delta = await ctx.get_kv("scene:expectation:last_delta", {})
        if isinstance(scene_delta, Mapping):
            magnitude = _clamp01(_safe_float(scene_delta.get("magnitude", 0.0), 0.0))
            delta_ts = _safe_float(scene_delta.get("ts", scene_delta.get("observed_ts", 0.0)), 0.0)
            if magnitude >= scene_delta_threshold and delta_ts > 0.0:
                age = max(0.0, now_ts - delta_ts)
                if age <= current_window_s:
                    activity_ages.append(age)
                    anchors.append(f"scene_delta:{magnitude:.2f}")

        unresolved = await ctx.get_kv("scene:expectation:last_unresolved_question", {})
        if isinstance(unresolved, Mapping):
            expires_at = _safe_float(unresolved.get("expires_at", 0.0), 0.0)
            created = _safe_float(unresolved.get("created_at", unresolved.get("ts", 0.0)), 0.0)
            if expires_at > now_ts:
                age = max(0.0, now_ts - created) if created > 0.0 else 0.0
                activity_ages.append(age)
                question = str(unresolved.get("question") or "").strip()
                if question:
                    anchors.append(question[:120])

        if block_slearn:
            slearn_status = str(await ctx.get_kv("slearn:status", "") or "").lower()
            slearn_phase = str(await ctx.get_kv("slearn:phase", "") or "").lower()
            if slearn_status in {"ingesting", "waiting_commit", "waiting_composer", "preflight"} or slearn_phase in {"ingesting", "waiting_commit", "waiting_composer", "preflight"}:
                blockers.append("slearn_background_active")

        if block_read:
            read_status = str(await ctx.get_kv("read:status", "") or "").lower()
            read_active = bool(await ctx.get_kv("read:active_file", ""))
            if read_status in {"reading", "ingesting", "active"} or read_active:
                blockers.append("read_background_active")

        activity_age = min(activity_ages) if activity_ages else float("inf")
        if blockers:
            mode = "background_blocked"
            blocked_reason = ",".join(blockers[:4])
        elif activity_age < settling_s:
            mode = "active"
            blocked_reason = "present_context_active"
        elif activity_age < idle_s:
            mode = "settling"
            blocked_reason = "waiting_for_idle_wander_threshold"
        else:
            mode = "idle_wander"
            blocked_reason = ""

        return {
            "mode": mode,
            "origin": mode,
            "blocked_reason": blocked_reason,
            "activity_age_s": None if activity_age == float("inf") else round(activity_age, 3),
            "settling_threshold_s": settling_s,
            "idle_wander_threshold_s": idle_s,
            "current_window_s": current_window_s,
            "anchors": self._unique_anchors(anchors, limit=8),
            "memory_drift_radius": max(0, _safe_int(await ctx.get_kv("probe:memory_drift_radius", DEFAULT_MEMORY_DRIFT_RADIUS), DEFAULT_MEMORY_DRIFT_RADIUS)),
        }

    def _score_candidates(self, candidates: Iterable[Dict[str, Any]], context: Mapping[str, Any]) -> list[tuple[float, Dict[str, Any]]]:
        anchor_tokens: set[str] = set()
        for anchor in list(context.get("anchors", []) or []):
            anchor_tokens.update(self._tokens(str(anchor)))
        drift_radius = max(0, int(context.get("memory_drift_radius", DEFAULT_MEMORY_DRIFT_RADIUS) or 0))
        anchor_weight = min(0.35, 0.12 + 0.08 * drift_radius) if anchor_tokens else 0.0

        scored: list[tuple[float, Dict[str, Any]]] = []
        for row in candidates:
            anchor = str(((row.get("anchor", {}) or {}).get("ref", "") or ""))
            row_tokens = self._tokens(anchor)
            overlap = 0.0
            if anchor_tokens and row_tokens:
                overlap = len(anchor_tokens & row_tokens) / max(1, min(len(anchor_tokens), len(row_tokens)))
            activation = _clamp01(_safe_float(row.get("activation", 0.2), 0.2))
            encounters = max(1.0, _safe_float(row.get("encounter_count", 1), 1.0))
            novelty = max(0.10, 1.0 - min(encounters / 8.0, 1.0))
            base = ((1.0 - activation) * 0.55) + (novelty * 0.45)
            score = (base * (1.0 - anchor_weight)) + (overlap * anchor_weight)
            scored.append((_clamp01(score), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    async def _publish_state(self, ctx, *, now_ts: float, **state: Any) -> None:
        payload = {
            "schema": "thought.probe.runtime.v1",
            "ts": now_ts,
            "enabled": bool(await ctx.get_kv("probe:enabled", True)),
            "idle_wander_enabled": bool(await ctx.get_kv("probe:idle_wander_enabled", True)),
            "mode": str(state.get("mode") or "unknown"),
            "origin": str(state.get("origin") or state.get("mode") or "unknown"),
            "blocked_reason": str(state.get("blocked_reason") or ""),
            "activity_age_s": state.get("activity_age_s"),
            "settling_threshold_s": state.get("settling_threshold_s"),
            "idle_wander_threshold_s": state.get("idle_wander_threshold_s"),
            "current_window_s": state.get("current_window_s"),
            "memory_drift_radius": state.get("memory_drift_radius"),
            "anchors": list(state.get("anchors", []) or [])[:8],
            "best_score": state.get("best_score"),
            "threshold": state.get("threshold"),
            "cooldown_remaining_s": state.get("cooldown_remaining_s"),
            "last_probe": state.get("last_probe"),
            "policy": "present_context_owns_front_of_mind_idle_opens_memory_wander",
        }
        await ctx.set_kv("probe:runtime_state", payload)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {part.strip().lower() for part in text.replace("_", " ").replace(":", " ").split() if len(part.strip()) >= 2}

    @staticmethod
    def _unique_anchors(values: Iterable[str], *, limit: int = 8) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= limit:
                break
        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[SERVICE_TOPIC],
        output_topics=["thought/probe"],
        priority=1,
        cooldown_sec=0.0,
    )
    yield LightProbeNeuron(cfg)
