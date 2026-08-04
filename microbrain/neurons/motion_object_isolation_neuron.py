from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.mb_vision.ram_frames import get_ram_frame

NEURON_NAME = Path(__file__).stem

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Motion isolation is deliberately cheaper/slower than camera capture.  It is
# a perceptual comparison organ, not another full-frame recognition pass.
MOTION_ANALYSIS_MAX_HZ = 8.0
MOTION_ANALYSIS_WIDTH = 320
MOTION_PIXEL_THRESHOLD = 20
MOTION_MIN_AREA_FRAC = 0.0012
MOTION_MAX_AREA_FRAC = 0.45
MOTION_ASSOCIATION_GATE = 0.34
MOTION_PROMOTE_HITS = 2
MOTION_LOST_GRACE_S = 0.65
MOTION_SEARCH_TIMEOUT_S = 2.75
MOTION_SNIPPET_MAX_PX = 160
MOTION_SNIPPET_QUALITY = 68
MOTION_MAX_TRACKS = 24
MOTION_CURIOSITY_BOOST = 0.34

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

MOTION_ISOLATION_SCHEMA = "vision.object_isolation.v1"
MOTION_ATTENTION_SCHEMA = "vision.motion_attention.v1"


class MotionObjectIsolationNeuron(BaseNeuron):
    """Isolate candidate objects from coherent frame-to-frame change.

    This organ intentionally answers a simpler question than recognition:

        "Which localized visual material changed together?"

    Coherent local motion is evidence that a region belongs to one independent
    object.  Stable runtime IDs are associated across frames; recognition may
    later attach a semantic label to the same track.  Raw frames and snippets
    remain RAM-only.
    """

    def __init__(self, config: NeuronConfig) -> None:
        super().__init__(config)
        self._prev_gray = None
        self._prev_ts = 0.0
        self._last_analysis_ts = 0.0
        self._tracks: Dict[str, Dict[str, Any]] = {}
        self._track_counter = 0

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/vision":
            return []
        if not bool(await ctx.get_kv("vision:isolation:enabled", True)):
            return []

        payload = event.payload if isinstance(event.payload, Mapping) else {}
        frame_ref = str(payload.get("data_ref") or payload.get("frame_ref") or "").strip()
        if not frame_ref:
            return []

        now = float(payload.get("ts", event.timestamp) or time.time())
        max_hz = float(await ctx.get_kv("vision:isolation:max_hz", MOTION_ANALYSIS_MAX_HZ) or MOTION_ANALYSIS_MAX_HZ)
        if max_hz > 0 and self._last_analysis_ts > 0 and (now - self._last_analysis_ts) < (1.0 / max_hz):
            return []

        decoded = await self._decode_frame(ctx, frame_ref)
        if decoded is None:
            return []
        frame_bgr, source_w, source_h = decoded

        try:
            import cv2
        except Exception:
            return []

        analysis_width = int(await ctx.get_kv("vision:isolation:analysis_width", MOTION_ANALYSIS_WIDTH) or MOTION_ANALYSIS_WIDTH)
        analysis_width = max(96, min(640, analysis_width))
        scale = min(1.0, analysis_width / max(1, source_w))
        aw = max(1, int(round(source_w * scale)))
        ah = max(1, int(round(source_h * scale)))
        small = cv2.resize(frame_bgr, (aw, ah), interpolation=cv2.INTER_AREA) if (aw, ah) != (source_w, source_h) else frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self._last_analysis_ts = now
        if self._prev_gray is None or getattr(self._prev_gray, "shape", None) != gray.shape:
            self._prev_gray = gray
            self._prev_ts = now
            return []

        global_shift, aligned_prev = self._align_previous(self._prev_gray, gray)
        regions = self._motion_regions(
            aligned_prev,
            gray,
            source_w=source_w,
            source_h=source_h,
            analysis_w=aw,
            analysis_h=ah,
            pixel_threshold=int(await ctx.get_kv("vision:isolation:pixel_threshold", MOTION_PIXEL_THRESHOLD) or MOTION_PIXEL_THRESHOLD),
            min_area_frac=float(await ctx.get_kv("vision:isolation:min_area_frac", MOTION_MIN_AREA_FRAC) or MOTION_MIN_AREA_FRAC),
            max_area_frac=float(await ctx.get_kv("vision:isolation:max_area_frac", MOTION_MAX_AREA_FRAC) or MOTION_MAX_AREA_FRAC),
        )

        dt = max(1e-3, now - self._prev_ts) if self._prev_ts > 0 else 0.125
        attention_events = await self._update_tracks(
            ctx,
            frame_bgr=frame_bgr,
            frame_ref=frame_ref,
            regions=regions,
            now=now,
            dt=dt,
            source_w=source_w,
            source_h=source_h,
        )

        self._prev_gray = gray
        self._prev_ts = now

        current = self._public_tracks(now)
        await ctx.set_kv("vision:isolation:objects", current)
        await ctx.set_kv(
            "vision:isolation:last",
            {
                "schema": MOTION_ISOLATION_SCHEMA,
                "ts": now,
                "frame_ref": frame_ref,
                "objects": current,
                "global_motion": {
                    "dx": round(float(global_shift[0]) * (source_w / max(1, aw)), 3),
                    "dy": round(float(global_shift[1]) * (source_h / max(1, ah)), 3),
                    "policy": "estimated_camera_motion_subtracted_before_local_motion",
                },
                "storage_policy": "ram_ephemeral_object_state",
            },
        )

        out: List[Event] = []
        if current:
            out.append(
                Event(
                    topic="vision/object_isolation",
                    payload={
                        "schema": MOTION_ISOLATION_SCHEMA,
                        "ts": now,
                        "frame_ref": frame_ref,
                        "objects": current,
                        "global_motion": {
                            "dx": round(float(global_shift[0]) * (source_w / max(1, aw)), 3),
                            "dy": round(float(global_shift[1]) * (source_h / max(1, ah)), 3),
                        },
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "kind": "vision_object_isolation",
                        "store_in_memory": False,
                        "cognitive_visible": False,
                        "reinforcement_eligible": False,
                        "ui_instrument": True,
                    },
                )
            )
        out.extend(attention_events)
        return out

    async def _decode_frame(self, ctx, frame_ref: str) -> Optional[Tuple[Any, int, int]]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return None

        raw: Optional[bytes] = None
        if frame_ref.startswith("ram:vision:"):
            packet = await get_ram_frame(ctx, frame_ref)
            if not isinstance(packet, Mapping):
                return None
            data = packet.get("jpeg_bytes")
            if not isinstance(data, (bytes, bytearray)):
                return None
            raw = bytes(data)
        else:
            try:
                frame = cv2.imread(frame_ref, cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            if frame is None:
                return None
            h, w = frame.shape[:2]
            return frame, int(w), int(h)

        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return frame, int(w), int(h)

    @staticmethod
    def _align_previous(previous, current) -> Tuple[Tuple[float, float], Any]:
        """Estimate global camera translation and align prior frame to current."""
        try:
            import cv2
            import numpy as np

            shift, response = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
            dx, dy = float(shift[0]), float(shift[1])
            if not math.isfinite(dx) or not math.isfinite(dy) or float(response or 0.0) < 0.08:
                return (0.0, 0.0), previous
            max_shift = max(previous.shape[:2]) * 0.20
            if abs(dx) > max_shift or abs(dy) > max_shift:
                return (0.0, 0.0), previous
            matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            aligned = cv2.warpAffine(previous, matrix, (current.shape[1], current.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return (dx, dy), aligned
        except Exception:
            return (0.0, 0.0), previous

    @staticmethod
    def _motion_regions(
        previous,
        current,
        *,
        source_w: int,
        source_h: int,
        analysis_w: int,
        analysis_h: int,
        pixel_threshold: int,
        min_area_frac: float,
        max_area_frac: float,
    ) -> List[Dict[str, Any]]:
        try:
            import cv2
        except Exception:
            return []

        diff = cv2.absdiff(previous, current)
        threshold = max(5, min(80, int(pixel_threshold)))
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(max(1, analysis_w * analysis_h))
        sx = source_w / max(1.0, float(analysis_w))
        sy = source_h / max(1.0, float(analysis_h))
        out: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            frac = area / frame_area
            if frac < max(0.0001, min_area_frac) or frac > max(min_area_frac, max_area_frac):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 4 or h < 4:
                continue
            perimeter = float(cv2.arcLength(contour, True) or 0.0)
            epsilon = max(1.0, 0.018 * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points: List[List[float]] = []
            for p in approx[:32]:
                px = float(p[0][0]) * sx
                py = float(p[0][1]) * sy
                points.append([round(px, 2), round(py, 2)])
            bbox = [round(x * sx, 2), round(y * sy, 2), round(w * sx, 2), round(h * sy, 2)]
            fill = area / max(1.0, float(w * h))
            out.append(
                {
                    "bbox": bbox,
                    "contour": points,
                    "area_frac": round(frac, 6),
                    "coherence": round(max(0.0, min(1.0, fill)), 4),
                }
            )
        out = MotionObjectIsolationNeuron._merge_nearby_motion_regions(out, source_w=source_w, source_h=source_h)
        out.sort(key=lambda row: float(row.get("area_frac", 0.0) or 0.0), reverse=True)
        return out[:MOTION_MAX_TRACKS]

    @staticmethod
    def _merge_nearby_motion_regions(
        regions: List[Dict[str, Any]],
        *,
        source_w: int,
        source_h: int,
    ) -> List[Dict[str, Any]]:
        """Join complementary motion lobes produced by one translated object.

        Frame differencing often sees the leading and trailing edges of a
        translated object as two separate islands.  When those islands are
        strongly aligned and close enough to plausibly be one coherent moving
        body, merge them before assigning object IDs.  This is intentionally
        conservative so nearby independent objects do not collapse together.
        """
        work = [dict(row) for row in regions]

        def metrics(a: Mapping[str, Any], b: Mapping[str, Any]):
            ax, ay, aw, ah = (float(v) for v in list(a.get("bbox") or [])[:4])
            bx, by, bw, bh = (float(v) for v in list(b.get("bbox") or [])[:4])
            h_overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            v_overlap = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            h_ratio = h_overlap / max(1.0, min(aw, bw))
            v_ratio = v_overlap / max(1.0, min(ah, bh))
            h_gap = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
            v_gap = max(0.0, max(ay, by) - min(ay + ah, by + bh))
            close_h = v_ratio >= 0.62 and h_gap <= max(10.0, min(source_w * 0.15, 1.85 * max(aw, bw)))
            close_v = h_ratio >= 0.62 and v_gap <= max(10.0, min(source_h * 0.15, 1.85 * max(ah, bh)))
            return close_h or close_v

        changed = True
        while changed:
            changed = False
            for i in range(len(work)):
                if changed:
                    break
                for j in range(i + 1, len(work)):
                    try:
                        should_merge = metrics(work[i], work[j])
                    except Exception:
                        should_merge = False
                    if not should_merge:
                        continue
                    a, b = work[i], work[j]
                    ax, ay, aw, ah = (float(v) for v in list(a.get("bbox") or [])[:4])
                    bx, by, bw, bh = (float(v) for v in list(b.get("bbox") or [])[:4])
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    points = []
                    for row in (a, b):
                        for point in list(row.get("contour") or []):
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                points.append([float(point[0]), float(point[1])])
                    hull_points: List[List[float]] = []
                    if len(points) >= 3:
                        try:
                            import cv2
                            import numpy as np
                            hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape((-1, 1, 2)))
                            hull_points = [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in hull[:32]]
                        except Exception:
                            hull_points = []
                    if not hull_points:
                        hull_points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    merged = {
                        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                        "contour": hull_points,
                        "area_frac": round(float(a.get("area_frac", 0.0) or 0.0) + float(b.get("area_frac", 0.0) or 0.0), 6),
                        "coherence": round(max(float(a.get("coherence", 0.0) or 0.0), float(b.get("coherence", 0.0) or 0.0)), 4),
                    }
                    work[i] = merged
                    work.pop(j)
                    changed = True
                    break
        return work

    async def _update_tracks(
        self,
        ctx,
        *,
        frame_bgr,
        frame_ref: str,
        regions: List[Dict[str, Any]],
        now: float,
        dt: float,
        source_w: int,
        source_h: int,
    ) -> List[Event]:
        association_gate = float(await ctx.get_kv("vision:isolation:association_gate", MOTION_ASSOCIATION_GATE) or MOTION_ASSOCIATION_GATE)
        promote_hits = int(await ctx.get_kv("vision:isolation:promote_hits", MOTION_PROMOTE_HITS) or MOTION_PROMOTE_HITS)
        lost_grace_s = float(await ctx.get_kv("vision:isolation:lost_grace_s", MOTION_LOST_GRACE_S) or MOTION_LOST_GRACE_S)
        search_timeout_s = float(await ctx.get_kv("vision:isolation:search_timeout_s", MOTION_SEARCH_TIMEOUT_S) or MOTION_SEARCH_TIMEOUT_S)
        max_tracks = int(await ctx.get_kv("vision:isolation:max_tracks", MOTION_MAX_TRACKS) or MOTION_MAX_TRACKS)
        curiosity_boost = float(await ctx.get_kv("vision:isolation:curiosity_boost", MOTION_CURIOSITY_BOOST) or MOTION_CURIOSITY_BOOST)

        unused_tracks = set(self._tracks)
        matches: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]] = []
        for region in regions:
            signature = self._appearance_signature(frame_bgr, region["bbox"])
            region["signature"] = signature
            best: Optional[Dict[str, Any]] = None
            best_score = -1.0
            for track_id in list(unused_tracks):
                track = self._tracks.get(track_id)
                if not track:
                    continue
                age = max(0.0, now - float(track.get("last_seen", now) or now))
                if age > search_timeout_s:
                    continue
                score = self._association_score(track, region, source_w=source_w, source_h=source_h, age_s=age)
                if score > best_score:
                    best_score = score
                    best = track
            if best is not None and best_score >= association_gate:
                unused_tracks.discard(str(best.get("track_id") or ""))
                matches.append((region, best, best_score))
            else:
                matches.append((region, None, best_score))

        gaze_state = await ctx.get_kv("vision:gaze_state", {})
        target_motion_id = str((gaze_state or {}).get("target_motion_id") or "") if isinstance(gaze_state, Mapping) else ""

        out: List[Event] = []
        matched_ids: set[str] = set()
        for region, track, score in matches:
            if track is None:
                track = self._new_track(region, frame_ref=frame_ref, now=now)
                self._tracks[track["track_id"]] = track
                was_lost = False
                is_new = True
            else:
                is_new = False
                was_lost = str(track.get("motion_state") or "") in {"lost", "searching"}
                old_bbox = list(track.get("bbox") or region["bbox"])
                old_center = self._bbox_center(old_bbox)
                new_center = self._bbox_center(region["bbox"])
                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]
                track["velocity_px_s"] = [dx / max(dt, 1e-3), dy / max(dt, 1e-3)]
                track["motion"] = {
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "speed_px_s": round(math.hypot(dx, dy) / max(dt, 1e-3), 3),
                    "speed_norm_s": round(math.hypot(dx / max(1, source_w), dy / max(1, source_h)) / max(dt, 1e-3), 5),
                }
                track["bbox"] = list(region["bbox"])
                track["contour"] = list(region.get("contour") or [])
                track["signature"] = int(region.get("signature", 0) or 0)
                track["coherence"] = float(region.get("coherence", track.get("coherence", 0.0)) or 0.0)
                track["area_frac"] = float(region.get("area_frac", track.get("area_frac", 0.0)) or 0.0)
                track["last_seen"] = now
                track["seen_count"] = int(track.get("seen_count", 1) or 1) + 1
                track["miss_count"] = 0
                track["frame_ref"] = frame_ref
                track["motion_state"] = "moving"
                track["status"] = "isolated" if int(track["seen_count"]) >= promote_hits else "candidate"
                track["confidence"] = self._track_confidence(track, promote_hits)
                track["association_score"] = round(float(score), 4)

            matched_ids.add(str(track["track_id"]))
            await self._store_snippet(ctx, frame_bgr, track, now=now)

            just_promoted = int(track.get("seen_count", 1) or 1) == promote_hits
            if just_promoted or (was_lost and not is_new):
                attention_kind = "reacquired" if was_lost else "motion_onset"
                out.append(self._motion_attention_event(track, kind=attention_kind, now=now, source_w=source_w, source_h=source_h))
                if just_promoted:
                    out.append(
                        Event(
                            topic="curiosity/adjust",
                            payload={
                                "boost": max(0.0, min(1.0, curiosity_boost)),
                                "reason": "coherent_visual_motion_isolated",
                                "visual_ref": str(track.get("track_id") or ""),
                            },
                            source=self.name,
                            meta={"store_in_memory": False, "cognitive_visible": False, "kind": "motion_curiosity"},
                        )
                    )
            elif target_motion_id and str(track.get("track_id") or "") == target_motion_id:
                speed = float((track.get("motion") or {}).get("speed_norm_s", 0.0) or 0.0)
                if speed >= 0.002:
                    out.append(self._motion_attention_event(track, kind="follow", now=now, source_w=source_w, source_h=source_h))

        # Tracks that were not matched are not immediately destroyed: a moving
        # object can be occluded or momentarily blend into the background.
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            last_seen = float(track.get("last_seen", now) or now)
            age = max(0.0, now - last_seen)
            track["miss_count"] = int(track.get("miss_count", 0) or 0) + 1
            vx, vy = self._velocity(track)
            predicted = self._shift_bbox(track.get("bbox"), vx * age, vy * age)
            if predicted is not None:
                track["predicted_bbox"] = predicted
            if age >= lost_grace_s and str(track.get("motion_state") or "") not in {"lost", "searching"}:
                track["motion_state"] = "searching"
                track["status"] = "lost"
                out.append(self._motion_attention_event(track, kind="lost", now=now, source_w=source_w, source_h=source_h))
            if age > search_timeout_s:
                self._tracks.pop(track_id, None)

        if len(self._tracks) > max_tracks:
            ordered = sorted(
                self._tracks.values(),
                key=lambda row: (float(row.get("last_seen", 0.0) or 0.0), float(row.get("confidence", 0.0) or 0.0)),
                reverse=True,
            )[:max_tracks]
            self._tracks = {str(row["track_id"]): row for row in ordered}
        return out

    def _new_track(self, region: Mapping[str, Any], *, frame_ref: str, now: float) -> Dict[str, Any]:
        self._track_counter += 1
        seed = f"{now:.6f}|{self._track_counter}|{region.get('bbox')}|{region.get('signature', 0)}"
        digest = hashlib.blake2b(seed.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()
        track_id = f"vobj:motion:{digest}"
        return {
            "track_id": track_id,
            "kind": "vision_motion_object.v1",
            "label": "unknown",
            "status": "candidate",
            "motion_state": "moving",
            "bbox": list(region.get("bbox") or []),
            "contour": list(region.get("contour") or []),
            "signature": int(region.get("signature", 0) or 0),
            "coherence": float(region.get("coherence", 0.0) or 0.0),
            "area_frac": float(region.get("area_frac", 0.0) or 0.0),
            "confidence": 0.28,
            "isolation_confidence": 0.28,
            "seen_count": 1,
            "miss_count": 0,
            "first_seen": now,
            "last_seen": now,
            "frame_ref": frame_ref,
            "motion": {"dx": 0.0, "dy": 0.0, "speed_px_s": 0.0, "speed_norm_s": 0.0},
            "velocity_px_s": [0.0, 0.0],
            "source": "motion_isolation",
        }

    @staticmethod
    def _track_confidence(track: Mapping[str, Any], promote_hits: int) -> float:
        hits = int(track.get("seen_count", 1) or 1)
        coherence = float(track.get("coherence", 0.0) or 0.0)
        persistence = min(1.0, hits / max(2.0, float(promote_hits + 3)))
        return round(max(0.0, min(0.94, 0.30 + 0.42 * persistence + 0.20 * coherence)), 4)

    def _association_score(self, track: Mapping[str, Any], region: Mapping[str, Any], *, source_w: int, source_h: int, age_s: float) -> float:
        vx, vy = self._velocity(track)
        predicted = self._shift_bbox(track.get("bbox"), vx * age_s, vy * age_s) or track.get("bbox")
        bbox = region.get("bbox")
        iou = self._iou(predicted, bbox)
        tc = self._bbox_center(predicted)
        rc = self._bbox_center(bbox)
        diag = max(1.0, math.hypot(source_w, source_h))
        center_similarity = max(0.0, 1.0 - (math.hypot(tc[0] - rc[0], tc[1] - rc[1]) / (diag * 0.30)))
        old_sig = int(track.get("signature", 0) or 0)
        new_sig = int(region.get("signature", 0) or 0)
        bit_dist = (old_sig ^ new_sig).bit_count() if old_sig or new_sig else 32
        appearance = max(0.0, 1.0 - bit_dist / 64.0)
        lost_bonus = 0.08 if str(track.get("motion_state") or "") in {"lost", "searching"} else 0.0
        return 0.50 * iou + 0.32 * center_similarity + 0.18 * appearance + lost_bonus

    @staticmethod
    def _appearance_signature(frame_bgr, bbox: Any) -> int:
        try:
            import cv2
            import numpy as np

            vals = [float(v) for v in list(bbox)[:4]]
            x, y, w, h = vals
            fh, fw = frame_bgr.shape[:2]
            x0 = max(0, min(fw - 1, int(round(x))))
            y0 = max(0, min(fh - 1, int(round(y))))
            x1 = max(x0 + 1, min(fw, int(round(x + w))))
            y1 = max(y0 + 1, min(fh, int(round(y + h))))
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return 0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sample = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            diff = sample[:, 1:] > sample[:, :-1]
            bits = 0
            for flag in diff.flatten():
                bits = (bits << 1) | int(bool(flag))
            return int(bits)
        except Exception:
            return 0

    async def _store_snippet(self, ctx, frame_bgr, track: Dict[str, Any], *, now: float) -> None:
        try:
            import cv2
        except Exception:
            return
        bbox = track.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
            fh, fw = frame_bgr.shape[:2]
            x0 = max(0, min(fw - 1, int(round(x))))
            y0 = max(0, min(fh - 1, int(round(y))))
            x1 = max(x0 + 1, min(fw, int(round(x + w))))
            y1 = max(y0 + 1, min(fh, int(round(y + h))))
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return
            max_px = MOTION_SNIPPET_MAX_PX
            ch, cw = crop.shape[:2]
            scale = min(1.0, max_px / max(1, max(ch, cw)))
            if scale < 1.0:
                crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), MOTION_SNIPPET_QUALITY])
            if not ok:
                return
            track_id = str(track.get("track_id") or "")
            ref = f"ram:vision:object:{track_id}"
            snippets = dict(await ctx.get_kv("vision:object_snippets", {}) or {})
            snippets[track_id] = {
                "ref": ref,
                "track_id": track_id,
                "ts": now,
                "frame_ref": str(track.get("frame_ref") or ""),
                "bbox": list(bbox[:4]),
                "format": "jpeg",
                "jpeg_bytes": bytes(encoded.tobytes()),
            }
            if len(snippets) > MOTION_MAX_TRACKS:
                rows = sorted(snippets.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0.0) or 0.0), reverse=True)
                snippets = dict(rows[:MOTION_MAX_TRACKS])
            await ctx.set_kv("vision:object_snippets", snippets)
            track["snippet_ref"] = ref
        except Exception:
            return

    def _public_tracks(self, now: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for track in self._tracks.values():
            status = str(track.get("status") or "candidate")
            if status == "candidate" and int(track.get("seen_count", 1) or 1) < MOTION_PROMOTE_HITS:
                continue
            bbox = track.get("predicted_bbox") if status == "lost" and track.get("predicted_bbox") else track.get("bbox")
            row = {
                "track_id": str(track.get("track_id") or ""),
                "label": str(track.get("label") or "unknown"),
                "status": status,
                "confidence": float(track.get("confidence", 0.0) or 0.0),
                "isolation_confidence": float(track.get("confidence", 0.0) or 0.0),
                "bbox": list(bbox or []),
                "contour": list(track.get("contour") or []),
                "motion": dict(track.get("motion") or {}),
                "motion_state": str(track.get("motion_state") or ""),
                "position": self._normalized_center(bbox),
                "snippet_ref": str(track.get("snippet_ref") or ""),
                "source_ref": str(track.get("frame_ref") or ""),
                "first_seen": float(track.get("first_seen", now) or now),
                "last_seen": float(track.get("last_seen", now) or now),
                "seen_count": int(track.get("seen_count", 1) or 1),
                "source": "vision/object_isolation",
                "objecthood_evidence": ["coherent_motion", "frame_delta", "spatial_persistence"],
            }
            rows.append(row)
        rows.sort(key=lambda row: (row["status"] != "lost", row["confidence"], row["last_seen"]), reverse=True)
        return rows[:MOTION_MAX_TRACKS]

    def _motion_attention_event(
        self,
        track: Mapping[str, Any],
        *,
        kind: str,
        now: float,
        source_w: int,
        source_h: int,
    ) -> Event:
        bbox = track.get("predicted_bbox") if kind == "lost" and track.get("predicted_bbox") else track.get("bbox")
        center = self._bbox_center(bbox)
        position = {
            "x": round(center[0] / max(1.0, float(source_w)), 6),
            "y": round(center[1] / max(1.0, float(source_h)), 6),
            "x_px": round(center[0], 2),
            "y_px": round(center[1], 2),
        }
        return Event(
            topic="vision/motion_attention",
            payload={
                "schema": MOTION_ATTENTION_SCHEMA,
                "kind": kind,
                "track_id": str(track.get("track_id") or ""),
                "bbox": list(bbox or []),
                "position": position,
                "source_width": int(source_w),
                "source_height": int(source_h),
                "motion": dict(track.get("motion") or {}),
                "confidence": float(track.get("confidence", 0.0) or 0.0),
                "ts": now,
            },
            source=self.name,
            meta={
                "kind": "vision_motion_attention",
                "store_in_memory": False,
                "cognitive_visible": False,
                "reinforcement_eligible": False,
            },
        )

    @staticmethod
    def _velocity(track: Mapping[str, Any]) -> Tuple[float, float]:
        raw = track.get("velocity_px_s")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                return float(raw[0]), float(raw[1])
            except Exception:
                pass
        return 0.0, 0.0

    @staticmethod
    def _bbox_center(bbox: Any) -> Tuple[float, float]:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x, y, w, h = (float(v) for v in bbox[:4])
                return x + w / 2.0, y + h / 2.0
            except Exception:
                pass
        return 0.0, 0.0

    @staticmethod
    def _normalized_center(bbox: Any) -> Dict[str, float]:
        # Source dimensions are not carried here; bbox is still the canonical
        # coordinate. This normalized-looking field is only filled when a
        # caller later supplies width/height, so avoid inventing a scale.
        cx, cy = MotionObjectIsolationNeuron._bbox_center(bbox)
        return {"x_px": round(cx, 2), "y_px": round(cy, 2)}

    @staticmethod
    def _shift_bbox(bbox: Any, dx: float, dy: float) -> Optional[List[float]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
            return [x + float(dx), y + float(dy), w, h]
        except Exception:
            return None

    @staticmethod
    def _iou(a: Any, b: Any) -> float:
        if not isinstance(a, (list, tuple)) or len(a) < 4 or not isinstance(b, (list, tuple)) or len(b) < 4:
            return 0.0
        try:
            ax, ay, aw, ah = (float(v) for v in a[:4])
            bx, by, bw, bh = (float(v) for v in b[:4])
        except Exception:
            return 0.0
        x0 = max(ax, bx)
        y0 = max(ay, by)
        x1 = min(ax + aw, bx + bw)
        y1 = min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = max(1e-9, aw * ah + bw * bh - inter)
        return inter / union


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    yield MotionObjectIsolationNeuron(
        NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=["percept/vision"],
            output_topics=["vision/object_isolation", "vision/motion_attention", "curiosity/adjust"],
            priority=4,
        )
    )
