from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class VisionObjectDeltaNeuron(BaseNeuron):
    """
    Object-level vision delta gate.

    Core rule:
      vision identifies objects; LiDAR/depth only attaches when active,
      available, and fresh; durable memory candidates are compact deltas,
      not raw frames or point clouds.

    Inputs:
      - percept/vision
      - percept/vision/features
      - vision/percept_commit
      - percept/lidar
      - percept/depth

    Output:
      - vision/object_delta
    """

    HAZARD_TERMS = {
        "fire",
        "smoke",
        "burning",
        "blood",
        "weapon",
        "gun",
        "knife",
        "fall",
        "fallen",
        "glass",
        "broken_glass",
        "sparks",
        "wire",
        "exposed_wire",
        "leak",
        "water_leak",
        "intruder",
        "unknown_person",
    }

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic in ("percept/lidar", "percept/depth"):
            await self._handle_spatial_sensor(event, ctx)
            return []

        if event.topic not in (
            "percept/vision",
            "percept/vision/features",
            "vision/percept_commit",
        ):
            return []

        enabled = bool(await ctx.get_kv("vision:delta:enabled", True))
        if not enabled:
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        objects = self._normalize_objects(event.topic, payload)
        if not objects:
            return []

        now = float(payload.get("ts", event.timestamp) or time.time())
        scene_ref = self._scene_ref(event, payload)
        previous_scene = dict(await ctx.get_kv(f"vision:delta:scene:{scene_ref}", {}) or {})
        current_scene: dict[str, dict[str, Any]] = {}

        confidence_floor = self._float(
            await ctx.get_kv("vision:delta:confidence_floor", 0.35),
            0.35,
        )
        move_threshold_m = self._float(
            await ctx.get_kv("vision:delta:move_threshold_m", 0.15),
            0.15,
        )

        deltas: list[dict[str, Any]] = []
        for obj in objects:
            confidence = self._float(obj.get("confidence", 1.0), 1.0)
            if confidence < confidence_floor:
                continue

            key = self._object_key(scene_ref, obj)
            obj["object_key"] = key
            current_scene[key] = obj

            old = previous_scene.get(key)
            if not isinstance(old, Mapping):
                deltas.append(self._build_delta("object_appeared", scene_ref, None, obj, now))
                continue

            movement = self._movement_delta(old, obj, move_threshold_m)
            if movement is not None:
                delta = self._build_delta("object_moved", scene_ref, old, obj, now)
                delta["movement"] = movement
                deltas.append(delta)

            state_changes = self._state_delta(old, obj)
            if state_changes:
                delta = self._build_delta("object_state_changed", scene_ref, old, obj, now)
                delta["state_changes"] = state_changes
                deltas.append(delta)

        for key, old in previous_scene.items():
            if key in current_scene or not isinstance(old, Mapping):
                continue
            deltas.append(self._build_delta("object_missing", scene_ref, old, None, now))

        if current_scene:
            await ctx.set_kv(f"vision:delta:scene:{scene_ref}", current_scene)
            await ctx.set_kv("vision:delta:last_scene_ref", scene_ref)

        if not deltas:
            return []

        spatial = await self._fresh_spatial_attachment(ctx, now)
        if spatial is not None:
            for delta in deltas:
                delta.setdefault("confirmers", []).append("spatial")

        safe_space = await self._safe_space(ctx, event, payload)
        required_voters = int(
            await ctx.get_kv(
                "vision:delta:safe_space_required_voters" if safe_space else "vision:delta:base_required_voters",
                3 if safe_space else 2,
            )
            or (3 if safe_space else 2)
        )

        any_candidate = False
        for delta in deltas:
            voters = await self._quorum_voters(ctx, event, payload, delta, spatial, safe_space)
            emergency = "safety" in voters and self._hazard_delta(delta, payload)
            memory_candidate = bool(emergency or len(voters) >= required_voters)
            delta["memory_candidate"] = memory_candidate
            delta["memory_policy"] = "delta_description_reference_only"
            delta["quorum"] = {
                "voters": voters,
                "voter_count": len(voters),
                "required_voters": required_voters,
                "safe_space": safe_space,
                "emergency_override": emergency,
            }
            any_candidate = any_candidate or memory_candidate

        text = self._summary(scene_ref, deltas, any_candidate)
        out_payload = {
            "schema": "vision.object_delta.v1",
            "scene_ref": scene_ref,
            "previous_ref": payload.get("previous_ref") or payload.get("prev_ref") or "",
            "source_topic": event.topic,
            "image_ref": payload.get("data_ref") or payload.get("image_ref") or payload.get("frame_ref") or "",
            "image_ref_policy": "reference_only_do_not_hardsave_by_default",
            "raw_image_policy": "temporary_unless_evidence_or_quorum_gate_requests_save",
            "pointcloud_policy": "summary_or_ref_only_do_not_hardsave_by_default",
            "memory_candidate": any_candidate,
            "delta_count": len(deltas),
            "text": text,
            "deltas": deltas,
            "spatial": spatial,
        }

        await ctx.set_kv("vision:delta:last", out_payload)
        await self._write_state(ctx, out_payload)

        return [
            Event(
                topic="vision/object_delta",
                payload=out_payload,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "vision_object_delta",
                    "memory_candidate": any_candidate,
                    "store_in_memory": any_candidate,
                    "cognitive_visible": any_candidate,
                    "lidar_attached": spatial is not None,
                },
            )
        ]

    async def _handle_spatial_sensor(self, event: Event, ctx) -> None:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        active = bool(payload.get("active", payload.get("enabled", True)))
        available = bool(payload.get("available", True))

        if not active or not available:
            await ctx.set_kv("vision:delta:last_spatial", None)
            return

        packet = {
            "ts": float(payload.get("ts", event.timestamp) or time.time()),
            "source_topic": event.topic,
            "source": str(payload.get("source") or event.source or event.topic),
            "frame_ref": payload.get("frame_ref") or payload.get("scan_ref") or "",
            "map_ref": payload.get("map_ref") or "",
            "occupancy_ref": payload.get("occupancy_ref") or "",
            "resolution": payload.get("resolution"),
            "range_m": payload.get("range_m"),
            "summary": payload.get("summary") or payload.get("description") or "",
        }
        await ctx.set_kv("vision:delta:last_spatial", packet)

    async def _fresh_spatial_attachment(self, ctx, now: float) -> dict[str, Any] | None:
        packet = await ctx.get_kv("vision:delta:last_spatial", None)
        if not isinstance(packet, Mapping):
            return None

        ttl_s = self._float(await ctx.get_kv("vision:delta:lidar_ttl_s", 1.5), 1.5)
        ts = self._float(packet.get("ts", 0.0), 0.0)
        age_s = max(0.0, now - ts)
        if age_s > ttl_s:
            return None

        return {
            "attached": True,
            "age_s": round(age_s, 3),
            "source": packet.get("source"),
            "source_topic": packet.get("source_topic"),
            "frame_ref": packet.get("frame_ref"),
            "map_ref": packet.get("map_ref"),
            "occupancy_ref": packet.get("occupancy_ref"),
            "resolution": packet.get("resolution"),
            "range_m": packet.get("range_m"),
            "summary": packet.get("summary"),
            "policy": "optional_attachment_summary_or_ref_only",
        }

    def _normalize_objects(self, topic: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        if topic == "vision/percept_commit":
            label = str(payload.get("resolved_label") or payload.get("fallback_ref") or "that thing").strip()
            return [
                {
                    "label": label,
                    "object_id": payload.get("proto_id") or "",
                    "confidence": self._float(payload.get("max_stability", 0.65), 0.65),
                    "state": {
                        "status": payload.get("status", "unknown"),
                        "count": payload.get("count", 1),
                        "brightness": payload.get("brightness"),
                        "edge_energy": payload.get("edge_energy"),
                        "contrast_energy": payload.get("contrast_energy"),
                    },
                    "bbox": payload.get("crop_box"),
                    "position": payload.get("position") or payload.get("xyz"),
                    "focus_xy": payload.get("focus_xy"),
                    "focus_radius": payload.get("focus_radius"),
                    "source_ref": payload.get("frame_ref") or "",
                    "commit_confirmed": True,
                }
            ]

        raw_objects = payload.get("objects") or payload.get("features") or []
        if isinstance(raw_objects, (str, int, float)):
            raw_objects = [raw_objects]
        if not isinstance(raw_objects, list):
            return []

        out: list[dict[str, Any]] = []
        for item in raw_objects:
            if isinstance(item, Mapping):
                label = str(item.get("label") or item.get("name") or item.get("class") or item.get("type") or "").strip()
                if not label:
                    continue
                state = item.get("state") if isinstance(item.get("state"), Mapping) else {}
                out.append(
                    {
                        "label": label,
                        "object_id": item.get("object_id") or item.get("id") or "",
                        "confidence": self._float(item.get("confidence", item.get("conf", 1.0)), 1.0),
                        "state": dict(state),
                        "bbox": item.get("bbox") or item.get("box"),
                        "position": item.get("position") or item.get("xyz"),
                        "pose": item.get("pose"),
                        "distance_m": item.get("distance_m"),
                        "source_ref": item.get("source_ref") or payload.get("data_ref") or payload.get("image_ref") or "",
                    }
                )
            else:
                label = str(item or "").strip()
                if not label:
                    continue
                out.append(
                    {
                        "label": label,
                        "object_id": "",
                        "confidence": 1.0,
                        "state": {},
                        "bbox": None,
                        "position": None,
                        "pose": None,
                        "distance_m": None,
                        "source_ref": payload.get("data_ref") or payload.get("image_ref") or "",
                    }
                )
        return out

    def _scene_ref(self, event: Event, payload: Mapping[str, Any]) -> str:
        explicit = payload.get("scene_ref") or payload.get("scene_id")
        if explicit:
            return self._safe_key(str(explicit))

        meta = event.meta or {}
        place = meta.get("place") or meta.get("room") or payload.get("place") or payload.get("room")
        if place:
            return self._safe_key(f"place:{place}")

        window = payload.get("window") if isinstance(payload.get("window"), Mapping) else {}
        title = window.get("title") if isinstance(window, Mapping) else ""
        if title:
            return self._safe_key(f"window:{title}")

        camera = payload.get("camera") if isinstance(payload.get("camera"), Mapping) else {}
        camera_id = camera.get("id") or camera.get("name") if isinstance(camera, Mapping) else ""
        if camera_id:
            return self._safe_key(f"camera:{camera_id}")

        channel = payload.get("channel") or event.source or event.topic or "default"
        return self._safe_key(f"scene:{channel}")

    def _object_key(self, scene_ref: str, obj: Mapping[str, Any]) -> str:
        object_id = str(obj.get("object_id", "") or "").strip()
        if object_id:
            return self._safe_key(object_id)

        label = self._safe_key(str(obj.get("label", "object") or "object"))
        # Fallback identity is intentionally label + scene, not location.
        # Location belongs in the delta; if it is baked into identity then movement
        # looks like "missing + appeared" instead of "moved."
        basis = {"scene_ref": scene_ref, "label": label}
        digest = hashlib.blake2b(
            json.dumps(basis, sort_keys=True, default=str).encode("utf-8"),
            digest_size=6,
        ).hexdigest()
        return f"obj:{label}:{digest}"

    def _coarse_location(self, obj: Mapping[str, Any]) -> Any:
        position = obj.get("position")
        xyz = self._xyz_or_none(position)
        if xyz is not None:
            return tuple(round(v, 1) for v in xyz)

        bbox = obj.get("bbox")
        if isinstance(bbox, Mapping):
            vals = [bbox.get(k) for k in ("x", "y", "w", "h")]
            if all(v is not None for v in vals):
                return tuple(round(self._float(v, 0.0), 1) for v in vals)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return tuple(round(self._float(v, 0.0), 1) for v in bbox[:4])

        focus = obj.get("focus_xy")
        if isinstance(focus, Mapping):
            return (round(self._float(focus.get("x"), 0.5), 1), round(self._float(focus.get("y"), 0.5), 1))

        return "unknown"

    def _build_delta(
        self,
        change_type: str,
        scene_ref: str,
        old: Mapping[str, Any] | None,
        new: Mapping[str, Any] | None,
        ts: float,
    ) -> dict[str, Any]:
        obj = dict(new or old or {})
        label = str(obj.get("label", "object") or "object")
        return {
            "change_type": change_type,
            "scene_ref": scene_ref,
            "object_key": obj.get("object_key"),
            "label": label,
            "timestamp": ts,
            "previous": self._memory_safe_object(old),
            "current": self._memory_safe_object(new),
            "description": self._describe_delta(change_type, old, new),
            "confirmers": ["novelty"],
        }

    def _memory_safe_object(self, obj: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if obj is None:
            return None
        return {
            "label": obj.get("label"),
            "object_key": obj.get("object_key"),
            "confidence": obj.get("confidence"),
            "state": obj.get("state"),
            "bbox": obj.get("bbox"),
            "position": obj.get("position"),
            "pose": obj.get("pose"),
            "distance_m": obj.get("distance_m"),
            "source_ref": obj.get("source_ref"),
        }

    def _movement_delta(
        self,
        old: Mapping[str, Any],
        new: Mapping[str, Any],
        threshold_m: float,
    ) -> dict[str, Any] | None:
        old_xyz = self._xyz_or_none(old.get("position"))
        new_xyz = self._xyz_or_none(new.get("position"))
        if old_xyz is None or new_xyz is None:
            return None

        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(old_xyz, new_xyz)))
        if distance < threshold_m:
            return None

        return {"from": old.get("position"), "to": new.get("position"), "distance_m": round(distance, 3)}

    def _state_delta(self, old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        old_state = old.get("state") if isinstance(old.get("state"), Mapping) else {}
        new_state = new.get("state") if isinstance(new.get("state"), Mapping) else {}
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(set(old_state) | set(new_state)):
            old_value = old_state.get(key)
            new_value = new_state.get(key)
            if old_value != new_value:
                changes[str(key)] = {"from": old_value, "to": new_value}
        return changes

    async def _quorum_voters(
        self,
        ctx,
        event: Event,
        payload: Mapping[str, Any],
        delta: Mapping[str, Any],
        spatial: Mapping[str, Any] | None,
        safe_space: bool,
    ) -> list[str]:
        voters = list(delta.get("confirmers", []) or ["novelty"])

        if spatial is not None:
            voters.append("spatial")
        if event.topic == "vision/percept_commit":
            voters.append("stability")
        if payload.get("resolved_label") or payload.get("proto_id"):
            voters.append("object_identity")
        if self._hazard_delta(delta, payload):
            voters.append("safety")
        if self._task_relevant(event, payload):
            voters.append("task_relevance")
        if self._user_attention(event, payload):
            voters.append("user_attention")
        if payload.get("cross_modal") or (event.meta or {}).get("cross_modal"):
            voters.append("cross_modal")
        if safe_space:
            voters.append("known_safe_place")

        return self._dedupe(voters)

    async def _safe_space(self, ctx, event: Event, payload: Mapping[str, Any]) -> bool:
        if bool((event.meta or {}).get("safe_space", False) or payload.get("safe_space", False)):
            return True
        if bool(await ctx.get_kv("location:safe", False)):
            return True

        places = await ctx.get_kv("vision:delta:safe_places", ["home", "workshop", "base", "lab"])
        safe_places = {str(p).lower() for p in places if str(p or "").strip()} if isinstance(places, list) else set()
        place = str(
            (event.meta or {}).get("place")
            or (event.meta or {}).get("room")
            or payload.get("place")
            or payload.get("room")
            or ""
        ).lower()
        return bool(place and any(p in place for p in safe_places))

    def _task_relevant(self, event: Event, payload: Mapping[str, Any]) -> bool:
        meta = event.meta or {}
        return bool(
            payload.get("task_id")
            or payload.get("goal_id")
            or payload.get("task_relevant")
            or meta.get("task_id")
            or meta.get("goal_id")
            or meta.get("task_relevant")
        )

    def _user_attention(self, event: Event, payload: Mapping[str, Any]) -> bool:
        meta = event.meta or {}
        return bool(
            payload.get("user_attention")
            or payload.get("user_pointed")
            or payload.get("user_requested")
            or meta.get("user_attention")
            or meta.get("user_pointed")
            or meta.get("user_requested")
        )

    def _hazard_delta(self, delta: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
        bits: list[str] = []
        bits.append(str(delta.get("label", "") or ""))
        bits.append(str(delta.get("description", "") or ""))
        bits.append(str(payload.get("description", "") or payload.get("note", "") or ""))
        state_changes = delta.get("state_changes") if isinstance(delta.get("state_changes"), Mapping) else {}
        bits.extend(str(k) for k in state_changes.keys())
        haystack = " ".join(bits).lower().replace("-", "_").replace(" ", "_")
        return any(term in haystack for term in self.HAZARD_TERMS)

    def _describe_delta(
        self,
        change_type: str,
        old: Mapping[str, Any] | None,
        new: Mapping[str, Any] | None,
    ) -> str:
        obj = dict(new or old or {})
        label = str(obj.get("label", "object") or "object")
        if change_type == "object_appeared":
            return f"{label} appeared in the scene."
        if change_type == "object_missing":
            return f"{label} is missing from the expected scene."
        if change_type == "object_moved":
            return f"{label} moved from its previous position."
        if change_type == "object_state_changed":
            return f"{label} changed state."
        return f"{label} changed."

    def _summary(self, scene_ref: str, deltas: list[Mapping[str, Any]], any_candidate: bool) -> str:
        labels = [str(d.get("label", "object") or "object") for d in deltas[:4]]
        action = "memory candidate" if any_candidate else "temporary observation"
        if len(labels) == 1:
            return f"Vision delta in {scene_ref}: {labels[0]} changed; {action}."
        return f"Vision delta in {scene_ref}: {', '.join(labels)} changed; {action}."

    async def _write_state(self, ctx, packet: Mapping[str, Any]) -> None:
        memdir = await resolve_memdir_ctx(ctx, fallback=None)
        if not memdir:
            return
        out_dir = Path(memdir) / "state"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            (out_dir / "vision_object_delta_last.json").write_text(
                json.dumps(packet, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def _xyz_or_none(value: Any) -> tuple[float, float, float] | None:
        try:
            if isinstance(value, Mapping):
                return (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None
        return None

    @staticmethod
    def _safe_key(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "_.:-" else "_" for ch in str(value).strip().lower())[:96] or "unknown"

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = str(item or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/vision",
            "percept/vision/features",
            "vision/percept_commit",
            "percept/lidar",
            "percept/depth",
        ],
        output_topics=["vision/object_delta"],
        priority=7,
    )
    yield VisionObjectDeltaNeuron(cfg)
