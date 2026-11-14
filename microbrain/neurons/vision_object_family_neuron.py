from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
FAMILY_SCHEMA = "vision.object_family.v1"


class VisionObjectFamilyNeuron(BaseNeuron):
    """Provisional common-motion families without asserting physical identity.

    A family represents evidence of coordinated motion, not proof that two
    regions are one rigid object. Track IDs remain independently addressable.
    State is ephemeral and does not enter semantic or durable memory.
    """

    def __init__(self, config: NeuronConfig) -> None:
        super().__init__(config)
        self._pairs: dict[tuple[str, str], dict[str, Any]] = {}
        self._focus: dict[str, Any] = {}

    @staticmethod
    def _center(row: Mapping[str, Any]) -> tuple[float, float] | None:
        bbox = row.get("bbox")
        if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return None
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
            return x + w / 2.0, y + h / 2.0
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _motion(row: Mapping[str, Any]) -> tuple[float, float]:
        motion = row.get("motion") or {}
        if not isinstance(motion, Mapping):
            return 0.0, 0.0
        return float(motion.get("dx") or 0.0), float(motion.get("dy") or 0.0)

    def _update_pair(self, a: Mapping[str, Any], b: Mapping[str, Any], now: float) -> None:
        ida, idb = str(a.get("track_id") or ""), str(b.get("track_id") or "")
        ca, cb = self._center(a), self._center(b)
        if not ida or not idb or ca is None or cb is None:
            return
        key = tuple(sorted((ida, idb)))
        record = self._pairs.setdefault(
            key, {"hits": 0, "contradictions": 0, "first_seen": now, "last_seen": now, "offset": None}
        )
        record["last_seen"] = now
        va, vb = self._motion(a), self._motion(b)
        sa, sb = math.hypot(*va), math.hypot(*vb)
        offset = (ca[0] - cb[0], ca[1] - cb[1])
        prior = record.get("offset")
        record["offset"] = offset
        # Neither stationary proximity nor elapsed time alone establishes a family.
        if max(sa, sb) < 1.5:
            return
        if min(sa, sb) < 0.5 or math.hypot(va[0] - vb[0], va[1] - vb[1]) > max(2.0, max(sa, sb) * 0.32):
            record["contradictions"] += 1
            record["hits"] = max(0, record["hits"] - 2)
            return
        if prior is not None and math.dist(offset, prior) > max(3.0, max(sa, sb) * 0.50):
            record["contradictions"] += 1
            record["hits"] = max(0, record["hits"] - 1)
            return
        record["hits"] = min(60, record["hits"] + 1)

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "vision/object_isolation":
            return []
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        now = float(payload.get("ts") or time.time())
        objects = [
            row for row in (payload.get("objects") or [])
            if isinstance(row, Mapping) and str(row.get("track_id") or "")
            and row.get("status") != "lost"
        ][:24]
        for key, row in list(self._pairs.items()):
            if now - float(row.get("last_seen") or now) > 6.0:
                self._pairs.pop(key, None)
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                ca, cb = self._center(a), self._center(b)
                if ca is None or cb is None:
                    continue
                boxes = (a.get("bbox"), b.get("bbox"))
                scale = max(1.0, sum(
                    max(float(box[2]), float(box[3]))
                    for box in boxes if box and len(box) >= 4
                ))
                if math.dist(ca, cb) > scale * 1.5:
                    continue
                self._update_pair(a, b, now)
        families = []
        for ids, row in self._pairs.items():
            if row["hits"] < 2 or now - row["last_seen"] > 2.0:
                continue
            hits, negatives = int(row["hits"]), int(row["contradictions"])
            if negatives >= hits + 2:
                continue
            tier = "ESTABLISHED" if hits >= 12 and negatives <= 1 else "BELIEVE" if hits >= 5 else "MAYBE"
            families.append({
                "family_id": "family:" + "|".join(ids),
                "members": list(ids),
                "tier": tier,
                "evidence": {"common_motion_hits": hits, "contradictions": negatives},
                "physical_identity": "unresolved",
                "note": "Shared motion alone does not establish one rigid object",
            })
        gaze = await ctx.get_kv("vision:gaze_state", {})
        target = str(gaze.get("target_motion_id") or "") if isinstance(gaze, Mapping) else ""
        if target:
            if self._focus.get("target") != target:
                self._focus = {"target": target, "since": now, "last_change": now}
            elif any(target in family["members"] for family in families):
                self._focus["last_change"] = now
        state = {
            "schema": FAMILY_SCHEMA, "ts": now, "families": families,
            "focus": dict(self._focus), "ephemeral": True,
        }
        await ctx.set_kv("vision:object_families", state)
        return []


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield VisionObjectFamilyNeuron(NeuronConfig(
        name=NEURON_NAME, subscribed_topics=["vision/object_isolation"],
        output_topics=[], priority=3,
    ))
