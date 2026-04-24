from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: Any) -> float:
    return max(0.0, min(1.0, _safe_float(x, 0.0)))


class PowerReliefLearningNeuron(BaseNeuron):
    """
    Learn which interaction path helped MB get fed.

    Important rule:
      attention is context
      relief is reward

    So we only reinforce route policy when a real relief event happens in a short
    causal window after a need signal.
    """

    async def _append_episode(self, ctx, row: Dict[str, Any]) -> None:
        base = await resolve_memdir_ctx(ctx)
        out_dir = base / "learning"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "power_relief_episodes.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _normalize_stats(self, raw: Any) -> Dict[str, Any]:
        return dict(raw) if isinstance(raw, dict) else {}

    def _update_bucket(self, bucket: Dict[str, Any], relief_delta: float, latency_s: float) -> Dict[str, Any]:
        attempts = int(bucket.get("attempts", 0) or 0) + 1
        successes = int(bucket.get("successes", 0) or 0) + 1
        prev_relief = _safe_float(bucket.get("avg_relief", 0.0), 0.0)
        prev_latency = _safe_float(bucket.get("avg_latency_s", latency_s), latency_s)

        bucket["attempts"] = attempts
        bucket["successes"] = successes
        bucket["avg_relief"] = ((prev_relief * (successes - 1)) + relief_delta) / max(1, successes)
        bucket["avg_latency_s"] = ((prev_latency * (successes - 1)) + latency_s) / max(1, successes)
        bucket["success_rate"] = _clamp01(successes / max(1, attempts))
        bucket["last_success_ts"] = time.time()
        return bucket

    async def _learn_from_relief(self, ctx, event: Event, now: float) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        pending = await ctx.get_kv("drive:power_pending_request", None)
        if not isinstance(pending, dict):
            return

        causal_window_s = _safe_float(await ctx.get_kv("drive:power:causal_window_s", 90.0), 90.0)
        request_ts = _safe_float(pending.get("ts", 0.0), 0.0)
        if request_ts <= 0.0:
            return
        latency_s = max(0.0, now - request_ts)
        if latency_s > causal_window_s:
            return

        relief_delta = max(0.0, _safe_float(payload.get("delta_pct", 0.0), 0.0))
        if relief_delta <= 0.0:
            return

        outlet = str(pending.get("outlet", "textual") or "textual")
        style = str(pending.get("style", "direct_simple") or "direct_simple")
        vector = pending.get("vector", {}) if isinstance(pending.get("vector", {}), dict) else {}
        pressure = pending.get("pressure", {}) if isinstance(pending.get("pressure", {}), dict) else {}

        stats = self._normalize_stats(await ctx.get_kv("route:power_relief_stats", {}))
        outlet_bucket = stats.get(outlet, {}) if isinstance(stats.get(outlet, {}), dict) else {}
        outlet_bucket = self._update_bucket(outlet_bucket, relief_delta, latency_s)

        styles = outlet_bucket.get("styles", {}) if isinstance(outlet_bucket.get("styles", {}), dict) else {}
        style_bucket = styles.get(style, {}) if isinstance(styles.get(style, {}), dict) else {}
        styles[style] = self._update_bucket(style_bucket, relief_delta, latency_s)
        outlet_bucket["styles"] = styles
        stats[outlet] = outlet_bucket

        await ctx.set_kv("route:power_relief_stats", stats)
        await ctx.set_kv("drive:power:last_relief_ts", now)
        await ctx.set_kv("drive:power:last_relief_delta", relief_delta)
        await ctx.set_kv("drive:power_pending_request", {})

        episode = {
            "ts": now,
            "need": "power",
            "pressure": pressure,
            "outlet_used": outlet,
            "utterance_style": style,
            "message": pending.get("message"),
            "user_response": payload.get("reason", "cookie"),
            "relief_delta": round(relief_delta, 4),
            "latency_to_relief": round(latency_s, 4),
            "success": True,
            "vector": vector,
        }
        await self._append_episode(ctx, episode)

    async def _track_emitted_utterance(self, ctx, event: Event) -> None:
        if event.topic != "act/speech":
            return
        meta = event.meta or {}
        if str(meta.get("need", "") or "") != "power":
            return
        if not str(meta.get("kind", "") or "").startswith("speech_reason_"):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        text = str(payload.get("text", "") or "").strip()
        if not text:
            return
        pending = await ctx.get_kv("drive:power_pending_request", None)
        if not isinstance(pending, dict):
            return
        pending["message"] = text
        pending["utterance_source"] = str(meta.get("utterance_source", pending.get("utterance_source", "fallback")) or "fallback")
        pending["utterance_score"] = _safe_float(meta.get("utterance_score", pending.get("utterance_score", 0.0)), 0.0)
        await ctx.set_kv("drive:power_pending_request", pending)

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.time()
        if event.topic == "drive/power_request":
            payload = event.payload if isinstance(event.payload, dict) else {}
            await ctx.set_kv("drive:power_pending_request", payload)
            return []

        if event.topic == "act/speech":
            if bool(await ctx.get_kv("control:t_pending", False)):
                return []
            await self._track_emitted_utterance(ctx, event)
            return []

        if event.topic == "event/relief/power":
            await self._learn_from_relief(ctx, event, now)
            return []

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["drive/power_request", "act/speech", "event/relief/power"],
        output_topics=[],
        priority=9,
        cooldown_sec=0.0,
    )
    yield PowerReliefLearningNeuron(cfg)
