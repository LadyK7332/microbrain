from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from microbrain.objects.base_object import (
    build_scene_expectation_object,
    build_unresolved_question_object,
    diff_scene_signatures,
    scene_signature,
    stable_digest,
)
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class SceneExpectationNeuron(BaseNeuron):
    """
    Ephemeral scene expectation / delta / why-question organ.

    Core rule:
      scene.obj + time = scene.exp

    `scene.exp` is never saved as durable memory by this neuron. It is a
    temporary prediction object used to compare the current scene against the
    last known scene signature for the same place/context. Meaningful deltas
    become thought-stream notes and, if salient enough, a tiny parked
    unresolved question for the day.
    """

    DEFAULT_DELTA_THRESHOLD = 0.18
    DEFAULT_PARK_THRESHOLD = 0.42
    DEFAULT_EXP_TTL_S = 180.0
    DEFAULT_QUESTION_TTL_S = 86400.0

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "object/scene" or not isinstance(event.payload, Mapping):
            return []

        scene = dict(event.payload)
        now = time.time()
        place_key = self._place_key(scene)
        observed_sig = scene_signature(scene)

        last_by_place = await ctx.get_kv("scene:expectation:last_by_place", {})
        if not isinstance(last_by_place, dict):
            last_by_place = {}
        prior_packet = last_by_place.get(place_key, {}) if isinstance(last_by_place.get(place_key, {}), dict) else {}
        prior_scene = prior_packet.get("scene", {}) if isinstance(prior_packet.get("scene", {}), dict) else {}
        prior_sig = prior_packet.get("signature", {}) if isinstance(prior_packet.get("signature", {}), dict) else {}
        prior_ts = self._float(prior_packet.get("ts", 0.0), 0.0)

        # Always update the baseline after comparison. The expected scene is
        # useful only while parsing this one transition.
        last_by_place[place_key] = {"ts": now, "scene": scene, "signature": observed_sig}
        await ctx.set_kv("scene:expectation:last_by_place", last_by_place)
        await ctx.set_kv("scene:expectation:last_place", place_key)
        await ctx.set_kv("scene:expectation:last_observed_signature", observed_sig)

        if not prior_scene or not prior_sig:
            return [
                Event(
                    topic="scene/expectation_state",
                    payload={
                        "schema": "scene.expectation.v1",
                        "status": "baseline_set",
                        "place_key": place_key,
                        "scene_id": scene.get("object_id", ""),
                        "signature": observed_sig,
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta=self._quiet_meta("scene_expectation_baseline"),
                )
            ]

        expectation = build_scene_expectation_object(prior_scene, observed_at=now, place_key=place_key)
        delta = diff_scene_signatures(prior_sig, observed_sig)
        delta["time_since_prior_s"] = round(max(0.0, now - prior_ts), 3)
        delta["place_key"] = place_key

        await ctx.set_kv("scene:expectation:last_exp", expectation)
        await ctx.set_kv("scene:expectation:last_delta", delta)

        delta_threshold = self._float(await ctx.get_kv("scene:expectation:delta_threshold", self.DEFAULT_DELTA_THRESHOLD), self.DEFAULT_DELTA_THRESHOLD)
        park_threshold = self._float(await ctx.get_kv("scene:expectation:park_threshold", self.DEFAULT_PARK_THRESHOLD), self.DEFAULT_PARK_THRESHOLD)
        magnitude = self._float(delta.get("magnitude", 0.0), 0.0)

        out: list[Event] = [
            Event(
                topic="scene/expectation",
                payload={
                    "schema": "scene.expectation.v1",
                    "scene_exp": expectation,
                    "expected_signature": prior_sig,
                    "observed_signature": observed_sig,
                    "delta": delta,
                    "ephemeral": True,
                    "durable_memory": False,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta=self._quiet_meta("scene_exp_ephemeral"),
            )
        ]

        if magnitude < delta_threshold:
            await ctx.set_kv("scene:expectation:last_match", {"ts": now, "place_key": place_key, "magnitude": magnitude})
            return out

        question_text = self._why_question(delta)
        thought_text = f"Expected scene changed here; {question_text}"
        out.append(
            Event(
                topic="thought/internal",
                payload={
                    "text": thought_text,
                    "type": "scene_delta_question",
                    "question": question_text,
                    "source": "scene_expectation",
                    "scene_exp_id": expectation.get("object_id", ""),
                    "expected_scene_id": prior_scene.get("object_id", ""),
                    "observed_scene_id": scene.get("object_id", ""),
                    "place_key": place_key,
                    "delta": delta,
                    "urgency": min(1.0, magnitude),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "scene_delta_thought",
                    "channel": "thought",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                    "cognitive_visible": True,
                },
            )
        )

        if magnitude >= park_threshold:
            q_obj = build_unresolved_question_object(
                question=question_text,
                expected_scene_id=str(prior_scene.get("object_id", "") or ""),
                observed_scene_id=str(scene.get("object_id", "") or ""),
                place_key=place_key,
                delta=delta,
                salience=magnitude,
                expires_at=now + self.DEFAULT_QUESTION_TTL_S,
            )
            await self._park_question(ctx, q_obj)
            await ctx.set_kv("scene:expectation:last_unresolved_question", q_obj)
            out.append(
                Event(
                    topic="question/unresolved",
                    payload=q_obj,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta=self._quiet_meta("unresolved_scene_delta_question"),
                )
            )

        return out

    def _quiet_meta(self, kind: str) -> Dict[str, Any]:
        return {
            "kind": kind,
            "channel": "thought",
            "quiet": True,
            "store_in_memory": False,
            "reinforcement_eligible": False,
            "self_output_track": False,
            "cognitive_visible": False,
        }

    def _place_key(self, scene: Mapping[str, Any]) -> str:
        mods = scene.get("modalities", {}) if isinstance(scene.get("modalities", {}), Mapping) else {}
        internal = mods.get("internal", {}) if isinstance(mods.get("internal", {}), Mapping) else {}
        source_event = scene.get("source_event", {}) if isinstance(scene.get("source_event", {}), Mapping) else {}
        meta = source_event.get("meta", {}) if isinstance(source_event.get("meta", {}), Mapping) else {}
        scene_mod = mods.get("scene", {}) if isinstance(mods.get("scene", {}), Mapping) else {}
        mods_present = scene_mod.get("modalities_present", []) if isinstance(scene_mod.get("modalities_present", []), list) else []
        basis = {
            "place": meta.get("place") or meta.get("room") or internal.get("place") or "default",
            "channel": meta.get("channel", ""),
            "modalities": sorted(str(x) for x in mods_present),
        }
        return "place:" + stable_digest(basis, size=6)

    def _why_question(self, delta: Mapping[str, Any]) -> str:
        missing = list(delta.get("missing_classifiers", []) or [])
        added = list(delta.get("added_classifiers", []) or [])
        missing_kinds = list(delta.get("missing_kinds", []) or [])
        added_kinds = list(delta.get("added_kinds", []) or [])
        if missing and added:
            return f"Why did {', '.join(missing[:3])} change into {', '.join(added[:3])}?"
        if missing:
            return f"Why is {', '.join(missing[:3])} missing from this scene?"
        if added:
            return f"Why did {', '.join(added[:3])} appear in this scene?"
        if missing_kinds or added_kinds:
            bits = []
            if missing_kinds:
                bits.append("missing " + ", ".join(missing_kinds[:3]))
            if added_kinds:
                bits.append("added " + ", ".join(added_kinds[:3]))
            return "Why did the scene structure change: " + "; ".join(bits) + "?"
        return "Why did this scene change?"

    async def _park_question(self, ctx, question_obj: Mapping[str, Any]) -> None:
        existing = await ctx.get_kv("question:unresolved:recent", [])
        if not isinstance(existing, list):
            existing = []
        qid = str(question_obj.get("object_id", "") or "")
        existing = [q for q in existing if not (isinstance(q, Mapping) and str(q.get("object_id", "") or "") == qid)]
        existing.append(dict(question_obj))
        existing = existing[-32:]
        await ctx.set_kv("question:unresolved:recent", existing)

        try:
            memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
            out_dir = memdir / "questions"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "unresolved_questions.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(question_obj), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            self.debug("park_question_failed", error=repr(exc))

    def _float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["object/scene"],
        output_topics=["scene/expectation", "scene/expectation_state", "thought/internal", "question/unresolved"],
        priority=3,
        cooldown_sec=0.15,
    )
    return [SceneExpectationNeuron(cfg)]
