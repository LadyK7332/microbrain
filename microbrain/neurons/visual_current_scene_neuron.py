from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.vision_state import normalize_visual_object

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Drop detector-owned current-object records that have not been refreshed for
# this long. Unit: seconds. Default: 2.0.
VISUAL_CURRENT_STALE_S = 2.0

# Proto-object snapshots are intentionally sparse; retain them longer between
# tracker emissions while still keeping them ephemeral. Unit: seconds.
VISUAL_CURRENT_PROTO_STALE_S = 45.0

# Motion-isolated object tracks can survive brief occlusion/search windows. Unit: seconds.
VISUAL_CURRENT_ISOLATION_STALE_S = 4.0

# Bound RAM/current-scene growth if an upstream detector produces bad IDs.
VISUAL_CURRENT_MAX_OBJECTS = 64

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

VISUAL_CURRENT_SCHEMA = "visual.current.v1"


class VisualCurrentSceneNeuron(BaseNeuron):
    """Maintain the live visual object map as ephemeral RAM-resident state.

    This organ consumes object extraction/recognition results and updates
    ``visual:current`` in KV. It emits no cognitive/bus event for ordinary
    frame-to-frame state. The dashboard samples this RAM state directly through
    its bridge, while ``vision/object_delta`` remains the meaningful durable
    scene-change path.

    ``visual:exp`` is deliberately untouched; that name remains reserved for a
    predicted visual state, not the current observation.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic not in {
            "percept/vision/features",
            "vision/proto_object",
            "vision/percept_commit",
            "vision/object_isolation",
            "vision/object_delta",
        }:
            return []

        now = float(event.timestamp or time.time())
        stale_s = float(await ctx.get_kv("vision:current:stale_s", VISUAL_CURRENT_STALE_S) or VISUAL_CURRENT_STALE_S)
        proto_stale_s = float(
            await ctx.get_kv("vision:current:proto_stale_s", VISUAL_CURRENT_PROTO_STALE_S)
            or VISUAL_CURRENT_PROTO_STALE_S
        )
        isolation_stale_s = float(
            await ctx.get_kv("vision:current:isolation_stale_s", VISUAL_CURRENT_ISOLATION_STALE_S)
            or VISUAL_CURRENT_ISOLATION_STALE_S
        )
        max_objects = int(await ctx.get_kv("vision:current:max_objects", VISUAL_CURRENT_MAX_OBJECTS) or VISUAL_CURRENT_MAX_OBJECTS)

        previous = await ctx.get_kv("visual:current", {})
        previous_objects = previous.get("objects", []) if isinstance(previous, Mapping) else []
        object_map: dict[str, dict[str, Any]] = {
            str(obj.get("track_id") or ""): dict(obj)
            for obj in previous_objects
            if isinstance(obj, Mapping) and str(obj.get("track_id") or "")
        }
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        frame_ref = str(
            payload.get("data_ref")
            or payload.get("image_ref")
            or payload.get("frame_ref")
            or (previous.get("frame_ref") if isinstance(previous, Mapping) else "")
            or ""
        )

        if event.topic == "percept/vision/features":
            raw = payload.get("objects") or payload.get("features") or []
            if isinstance(raw, list):
                # Detector results replace only detector-owned records for this
                # frame. Proto-object tracks remain until their own stale TTL.
                detector_ids: set[str] = set()
                for index, item in enumerate(raw):
                    if not isinstance(item, Mapping):
                        continue
                    obj = normalize_visual_object(item, fallback_index=index, source=event.topic, timestamp=now)
                    obj["last_seen"] = now
                    if frame_ref and not obj.get("source_ref"):
                        obj["source_ref"] = frame_ref
                    object_map[obj["track_id"]] = obj
                    detector_ids.add(obj["track_id"])
                for track_id, obj in list(object_map.items()):
                    if obj.get("source") == event.topic and track_id not in detector_ids:
                        object_map.pop(track_id, None)

        elif event.topic in {"vision/proto_object", "vision/percept_commit"}:
            obj = normalize_visual_object(payload, source=event.topic, timestamp=now)
            obj["last_seen"] = now
            if frame_ref and not obj.get("source_ref"):
                obj["source_ref"] = frame_ref
            object_map[obj["track_id"]] = obj

        elif event.topic == "vision/object_isolation":
            raw = payload.get("objects") if isinstance(payload.get("objects"), list) else []
            isolation_ids: set[str] = set()
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping):
                    continue
                obj = normalize_visual_object(item, fallback_index=index, source=event.topic, timestamp=now)
                obj["last_seen"] = float(item.get("last_seen", now) or now)
                if frame_ref and not obj.get("source_ref"):
                    obj["source_ref"] = frame_ref
                object_map[obj["track_id"]] = obj
                isolation_ids.add(obj["track_id"])
            # Isolation tracks own their IDs; if the motion organ has retired a
            # track, let its short scene TTL handle disappearance instead of
            # rebuilding/removing the whole map each frame.

        elif event.topic == "vision/object_delta":
            raw_deltas = payload.get("deltas") if isinstance(payload.get("deltas"), list) else []
            for index, delta in enumerate(raw_deltas):
                if not isinstance(delta, Mapping):
                    continue
                change_type = str(delta.get("change_type") or "")
                current = delta.get("current") if isinstance(delta.get("current"), Mapping) else None
                previous_obj = delta.get("previous") if isinstance(delta.get("previous"), Mapping) else None
                source_obj = current or previous_obj
                if not isinstance(source_obj, Mapping):
                    continue
                merged = dict(source_obj)
                if delta.get("object_key") and not merged.get("object_key"):
                    merged["object_key"] = delta.get("object_key")
                obj = normalize_visual_object(merged, fallback_index=index, source=event.topic, timestamp=now)
                if change_type == "object_missing":
                    object_map.pop(obj["track_id"], None)
                else:
                    obj["last_seen"] = now
                    object_map[obj["track_id"]] = obj

        # Expire only the ephemeral scene map. No file/delete work occurs here.
        for track_id, obj in list(object_map.items()):
            try:
                age = now - float(obj.get("last_seen", now) or now)
            except (TypeError, ValueError):
                age = 0.0
            source = str(obj.get("source") or "")
            if source in {"vision/proto_object", "vision/percept_commit"}:
                ttl = proto_stale_s
            elif source == "vision/object_isolation":
                ttl = isolation_stale_s
            else:
                ttl = stale_s
            if age > ttl:
                object_map.pop(track_id, None)

        objects = self._coalesce_spatial_duplicates(list(object_map.values()))
        objects = sorted(
            objects,
            key=lambda obj: (
                bool(obj.get("hazard", False)),
                float(obj.get("confidence", 0.0) or 0.0),
                float(obj.get("last_seen", 0.0) or 0.0),
            ),
            reverse=True,
        )[:max_objects]

        state = {
            "schema": VISUAL_CURRENT_SCHEMA,
            "ts": now,
            "frame_ref": frame_ref,
            "object_count": len(objects),
            "objects": objects,
            "storage_policy": "ram_current_state_only",
            "prediction_key": "visual:exp_reserved_for_prediction",
        }
        await ctx.set_kv("visual:current", state)
        await ctx.set_kv("vision:current_objects", objects)
        return []

    @classmethod
    def _coalesce_spatial_duplicates(cls, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse near-identical boxes without collapsing legitimate sub-objects.

        The proto tracker historically could create several IDs around the same
        crop.  Current-scene state should represent one believed object/region,
        not every detector attempt.  We only coalesce boxes with both very high
        overlap and similar area, so an eye inside a face remains a possible
        sub-region rather than being silently erased.
        """
        kept: list[dict[str, Any]] = []
        for obj in sorted(
            objects,
            key=lambda row: (float(row.get("last_seen", 0.0) or 0.0), float(row.get("confidence", 0.0) or 0.0)),
            reverse=True,
        ):
            match = None
            for prior in kept:
                iou = cls._bbox_iou(prior.get("bbox"), obj.get("bbox"))
                ratio = cls._area_ratio(prior.get("bbox"), obj.get("bbox"))
                same_family = str(prior.get("source") or "") == str(obj.get("source") or "")
                if iou >= (0.82 if same_family else 0.90) and 0.72 <= ratio <= 1.38:
                    match = prior
                    break
            if match is None:
                kept.append(dict(obj))
                continue
            aliases = list(match.get("alias_track_ids") or [])
            alias_id = str(obj.get("track_id") or "")
            if alias_id and alias_id != str(match.get("track_id") or "") and alias_id not in aliases:
                aliases.append(alias_id)
            match["alias_track_ids"] = aliases[-8:]
            if float(obj.get("confidence", 0.0) or 0.0) > float(match.get("confidence", 0.0) or 0.0):
                for key in ("label", "confidence", "status", "contour", "snippet_ref", "isolation_confidence"):
                    if obj.get(key) not in (None, "", []):
                        match[key] = obj.get(key)
        return kept

    @staticmethod
    def _bbox_tuple(box: Any) -> tuple[float, float, float, float] | None:
        try:
            if isinstance(box, Mapping):
                if all(key in box for key in ("left", "top", "right", "bottom")):
                    x = float(box.get("left", 0.0) or 0.0)
                    y = float(box.get("top", 0.0) or 0.0)
                    return x, y, float(box.get("right", x) or x) - x, float(box.get("bottom", y) or y) - y
                if all(key in box for key in ("left", "top", "width", "height")):
                    return (
                        float(box.get("left", 0.0) or 0.0),
                        float(box.get("top", 0.0) or 0.0),
                        float(box.get("width", 0.0) or 0.0),
                        float(box.get("height", 0.0) or 0.0),
                    )
                if all(key in box for key in ("x", "y", "w", "h")):
                    return tuple(float(box.get(key, 0.0) or 0.0) for key in ("x", "y", "w", "h"))
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                return tuple(float(v) for v in box[:4])
        except Exception:
            return None
        return None

    @classmethod
    def _bbox_iou(cls, a: Any, b: Any) -> float:
        aa = cls._bbox_tuple(a)
        bb = cls._bbox_tuple(b)
        if aa is None or bb is None:
            return 0.0
        ax, ay, aw, ah = aa
        bx, by, bw, bh = bb
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = max(1e-9, aw * ah + bw * bh - inter)
        return inter / union

    @classmethod
    def _area_ratio(cls, a: Any, b: Any) -> float:
        aa = cls._bbox_tuple(a)
        bb = cls._bbox_tuple(b)
        if aa is None or bb is None:
            return 0.0
        return max(1e-9, aa[2] * aa[3]) / max(1e-9, bb[2] * bb[3])


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield VisualCurrentSceneNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=[
                "percept/vision/features",
                "vision/proto_object",
                "vision/percept_commit",
                "vision/object_isolation",
                "vision/object_delta",
            ],
            output_topics=[],
            priority=4,
        )
    )
