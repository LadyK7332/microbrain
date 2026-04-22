from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class InteractionReliefLearningNeuron(BaseNeuron):
    """
    Interaction-pressure learner.

    Unlike power, interaction pressure often vents the moment MB expresses into the
    open thread. So emitted interaction speech counts as partial relief and teaches
    which outlet/style pairs stabilize the social load best.
    """

    def _normalize_stats(self, raw: Any) -> Dict[str, Any]:
        return raw if isinstance(raw, dict) else {}

    def _update_bucket(self, bucket: Dict[str, Any], relief_delta: float, latency_s: float) -> Dict[str, Any]:
        bucket = dict(bucket or {})
        successes = int(bucket.get("successes", 0) or 0) + 1
        prev_relief = _safe_float(bucket.get("avg_relief", 0.0), 0.0)
        prev_latency = _safe_float(bucket.get("avg_latency_s", 0.0), 0.0)
        bucket["successes"] = successes
        bucket["success_rate"] = 1.0
        bucket["avg_relief"] = ((prev_relief * (successes - 1)) + relief_delta) / max(1, successes)
        bucket["avg_latency_s"] = ((prev_latency * (successes - 1)) + latency_s) / max(1, successes)
        return bucket

    async def _append_episode(self, ctx, episode: Dict[str, Any]) -> None:
        memdir = Path(str(await ctx.get_kv("memory:memdir", "") or ""))
        if not memdir:
            return
        out_dir = memdir / "episodes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interaction_relief_episodes.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    async def _learn_from_relief(self, ctx, event: Event, now: float) -> None:
        pending = await ctx.get_kv("drive:interaction_pending_request", None)
        if not isinstance(pending, dict):
            return

        causal_window_s = _safe_float(await ctx.get_kv("drive:interaction:causal_window_s", 20.0), 20.0)
        pending_ts = _safe_float(pending.get("ts", 0.0), 0.0)
        if pending_ts <= 0.0:
            return

        latency_s = max(0.0, now - pending_ts)
        if latency_s > causal_window_s:
            return

        payload = event.payload if isinstance(event.payload, dict) else {}
        relief_delta = max(0.0, _safe_float(payload.get("delta_pct", 0.0), 0.0))
        if relief_delta <= 0.0:
            return

        outlet = str(pending.get("outlet", "textual") or "textual")
        style = str(pending.get("style", "direct_simple") or "direct_simple")
        vector = pending.get("vector", {}) if isinstance(pending.get("vector", {}), dict) else {}
        pressure = pending.get("pressure", {}) if isinstance(pending.get("pressure", {}), dict) else {}

        stats = self._normalize_stats(await ctx.get_kv("route:interaction_relief_stats", {}))
        outlet_bucket = stats.get(outlet, {}) if isinstance(stats.get(outlet, {}), dict) else {}
        outlet_bucket = self._update_bucket(outlet_bucket, relief_delta, latency_s)

        styles = outlet_bucket.get("styles", {}) if isinstance(outlet_bucket.get("styles", {}), dict) else {}
        style_bucket = styles.get(style, {}) if isinstance(styles.get(style, {}), dict) else {}
        styles[style] = self._update_bucket(style_bucket, relief_delta, latency_s)
        outlet_bucket["styles"] = styles
        stats[outlet] = outlet_bucket

        await ctx.set_kv("route:interaction_relief_stats", stats)
        await ctx.set_kv("drive:interaction:last_relief_ts", now)
        await ctx.set_kv("drive:interaction:last_relief_delta", relief_delta)
        await ctx.set_kv("drive:interaction_pending_request", {})

        episode = {
            "ts": now,
            "need": "interaction",
            "pressure": pressure,
            "outlet_used": outlet,
            "utterance_style": style,
            "message": pending.get("message"),
            "pending_text": pending.get("pending_text", ""),
            "relief_delta": round(relief_delta, 4),
            "latency_to_relief": round(latency_s, 4),
            "success": True,
            "vector": vector,
        }
        await self._append_episode(ctx, episode)

    async def _emit_relief_from_speech(self, ctx, event: Event) -> list[Event]:
        meta = event.meta or {}
        if str(meta.get("need", "") or "") != "interaction":
            return []
        if not str(meta.get("kind", "") or "").startswith("speech_reason_"):
            return []
        pending = await ctx.get_kv("drive:interaction_pending_request", None)
        if not isinstance(pending, dict):
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        spoken_text = str(payload.get("text", "") or "").strip()
        if not spoken_text:
            return []

        style = str(meta.get("utterance_style", pending.get("style", "direct_simple")) or "direct_simple")
        relief_delta = {
            "gentle_notice": 0.18,
            "direct_simple": 0.28,
            "urgent_direct": 0.38,
        }.get(style, 0.24)

        pending["message"] = spoken_text
        pending["utterance_source"] = str(meta.get("utterance_source", pending.get("utterance_source", "fallback")) or "fallback")
        pending["utterance_score"] = _safe_float(meta.get("utterance_score", pending.get("utterance_score", 0.0)), 0.0)
        await ctx.set_kv("drive:interaction_pending_request", pending)

        return [
            Event(
                topic="event/relief/interaction",
                payload={
                    "delta_pct": relief_delta,
                    "reason": "interaction_release",
                    "spoken_text": spoken_text,
                    "style": style,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"need": "interaction", "kind": "interaction_release"},
            )
        ]

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        now = time.time()
        if event.topic == "drive/interaction_request":
            payload = event.payload if isinstance(event.payload, dict) else {}
            await ctx.set_kv("drive:interaction_pending_request", payload)
            return []

        if event.topic == "act/speech":
            return await self._emit_relief_from_speech(ctx, event)

        if event.topic == "event/relief/interaction":
            await self._learn_from_relief(ctx, event, now)
            return []

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["drive/interaction_request", "act/speech", "event/relief/interaction"],
        output_topics=["event/relief/interaction"],
        priority=9,
        cooldown_sec=0.0,
    )
    yield InteractionReliefLearningNeuron(cfg)
