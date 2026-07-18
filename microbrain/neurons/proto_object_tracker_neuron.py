from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class ProtoObjectTrackerNeuron(BaseNeuron):
    """
    Slow, curiosity-led proto-object tracker for vision.

    Purpose:
      - Avoid instant hard labels for unknown visual content.
      - Form weak, revisitable proto-object tracks around the current focus point.
      - Keep curiosity high, importance low, and only grow confidence with repeats.
      - Accept later teacher labels via ``vision/proto_label``.

    This is intentionally conservative: it does not try to be a fast classifier.
    It builds tentative visual threads the rest of the system can revisit.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            'received',
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic == 'vision/proto_label':
            return await self._apply_label(event, ctx)

        if event.topic != 'percept/vision':
            return []

        enabled = bool(await ctx.get_kv('vision:proto:enabled', True))
        if not enabled:
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        frame_ref = str(payload.get('data_ref', '') or '').strip()
        if not frame_ref:
            return []

        image = self._load_image(frame_ref)
        if image is None:
            await ctx.log_debug(f'[{self.name}] could not open frame', frame_ref=frame_ref)
            return []

        focus_xy = payload.get('focus') if isinstance(payload.get('focus'), dict) else await ctx.get_kv('vision:gaze_state', None)
        if not isinstance(focus_xy, dict):
            focus_xy = await ctx.get_kv('vision:focus_xy', {'x': 0.5, 'y': 0.5})
        crop_px = int(await ctx.get_kv('vision:proto:crop_px', 160) or 160)
        stale_s = float(await ctx.get_kv('vision:proto:stale_s', 45.0) or 45.0)
        max_tracks = int(await ctx.get_kv('vision:proto:max_tracks', 32) or 32)
        min_repeat = int(await ctx.get_kv('vision:proto:min_repeat', 3) or 3)
        match_max_bits = int(await ctx.get_kv('vision:proto:match_max_bits', 10) or 10)
        focus_proximity = float(await ctx.get_kv('vision:proto:focus_proximity', 0.18) or 0.18)
        curiosity_start = float(await ctx.get_kv('vision:proto:curiosity_start', 0.72) or 0.72)
        importance_start = float(await ctx.get_kv('vision:proto:importance_start', 0.15) or 0.15)
        self_reinforce = float(await ctx.get_kv('vision:proto:self_reinforce', 0.05) or 0.05)
        internal_notes = bool(await ctx.get_kv('vision:proto:emit_internal_notes', True))
        speak_questions = bool(await ctx.get_kv('vision:proto:speak_questions', False))
        question_threshold = float(await ctx.get_kv('vision:proto:question_threshold', 0.86) or 0.86)
        question_cooldown_s = float(await ctx.get_kv('vision:proto:question_cooldown_s', 90.0) or 90.0)

        now = time.time()
        obs = self._make_observation(image=image, frame_ref=frame_ref, focus_xy=focus_xy, crop_px=crop_px)
        if obs is None:
            return []

        tracks = list(await ctx.get_kv('vision:proto:tracks', []) or [])
        track, is_new = self._attach_or_create_track(
            tracks=tracks,
            obs=obs,
            now=now,
            stale_s=stale_s,
            match_max_bits=match_max_bits,
            focus_proximity=focus_proximity,
            curiosity_start=curiosity_start,
            importance_start=importance_start,
        )

        importance_delta = self._importance_delta(event=event, payload=payload)
        if not is_new:
            track['seen_count'] = int(track.get('seen_count', 1) or 1) + 1
            track['revision'] = int(track.get('revision', 0) or 0) + 1
            track['last_seen'] = now
            track['frame_ref'] = frame_ref
            track['focus_xy'] = dict(obs['focus_xy'])
            track['focus_radius'] = float(obs.get('focus_radius', track.get('focus_radius', 0.08)) or track.get('focus_radius', 0.08) or 0.08)
            track['crop_box'] = dict(obs['crop_box'])
            track['brightness'] = float(obs['brightness'])
            track['edge_energy'] = float(obs['edge_energy'])
            track['contrast_energy'] = float(obs.get('contrast_energy', obs['edge_energy']) or obs['edge_energy'])
            track['signature'] = int(obs['signature'])
            track['stability'] = min(1.0, float(track.get('stability', 0.0) or 0.0) + (1.0 / max(1, min_repeat + 1)))
            if not str(track.get('resolved_label', '') or '').strip():
                track['curiosity'] = self._clamp(
                    float(track.get('curiosity', curiosity_start) or curiosity_start) + self_reinforce,
                    0.0,
                    1.0,
                )
            else:
                track['curiosity'] = self._clamp(float(track.get('curiosity', 0.25) or 0.25) - 0.08, 0.0, 1.0)
        else:
            track['curiosity'] = self._clamp(curiosity_start, 0.0, 1.0)
            track['importance'] = self._clamp(importance_start, 0.0, 1.0)
            track['stability'] = 0.18

        if importance_delta:
            track['importance'] = self._clamp(float(track.get('importance', importance_start) or importance_start) + importance_delta, 0.0, 1.0)
        else:
            baseline = float(await ctx.get_kv('vision:proto:importance_start', importance_start) or importance_start)
            current_importance = float(track.get('importance', baseline) or baseline)
            track['importance'] = self._clamp(current_importance * 0.98 + baseline * 0.02, 0.0, 1.0)

        track['fallback_ref'] = self._render_fallback_ref(track)
        track['status'] = 'labeled' if str(track.get('resolved_label', '') or '').strip() else 'unknown'
        track['should_ask'] = bool(
            str(track.get('resolved_label', '') or '').strip() == ''
            and float(track.get('curiosity', 0.0) or 0.0) >= question_threshold
            and float(track.get('stability', 0.0) or 0.0) >= 0.55
        )

        tracks = self._prune_tracks(tracks=tracks, now=now, stale_s=stale_s, max_tracks=max_tracks)
        await ctx.set_kv('vision:proto:tracks', tracks)

        snapshot = self._snapshot(track)
        await ctx.set_kv('vision:proto:last_focus', snapshot)
        await self._write_state(ctx, tracks)

        should_emit = self._should_emit_snapshot(track=track, is_new=is_new, min_repeat=min_repeat, now=now)
        out: List[Event] = []
        if should_emit:
            track['last_emit_ts'] = now
            snapshot = self._snapshot(track)
            out.append(
                Event(
                    topic='vision/proto_object',
                    payload=snapshot,
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        'kind': 'vision_proto_object',
                        'status': snapshot.get('status', 'unknown'),
                        'new_track': is_new,
                    },
                )
            )
            if internal_notes:
                out.append(
                    Event(
                        topic='reason/output',
                        payload={'text': self._internal_note(snapshot)},
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={'channel': 'thought', 'kind': 'vision_proto_note', 'lobe': 'vision'},
                    )
                )

        if speak_questions and snapshot.get('should_ask'):
            last_question_ts = float(track.get('last_question_ts', 0.0) or 0.0)
            if (now - last_question_ts) >= question_cooldown_s:
                track['last_question_ts'] = now
                out.append(
                    Event(
                        topic='thought/internal',
                        payload={
                            'kind': 'vision_proto_question',
                            'track': snapshot,
                        },
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={'channel': 'thought', 'kind': 'vision_proto_question', 'store_in_memory': False},
                    )
                )

        return out

    async def _apply_label(self, event: Event, ctx) -> List[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        proto_id = str(payload.get('proto_id', '') or '').strip()
        label = str(payload.get('label', '') or '').strip().lower()
        if not proto_id or not label:
            return []

        tracks = list(await ctx.get_kv('vision:proto:tracks', []) or [])
        now = time.time()
        changed = False
        snapshot: Optional[Dict[str, Any]] = None
        for track in tracks:
            if str(track.get('id', '') or '') != proto_id:
                continue
            labels = [str(x).strip().lower() for x in list(track.get('labels', []) or []) if str(x).strip()]
            if label not in labels:
                labels.append(label)
            track['labels'] = labels[:8]
            track['resolved_label'] = label
            track['status'] = 'labeled'
            track['curiosity'] = self._clamp(float(track.get('curiosity', 0.5) or 0.5) * 0.35, 0.0, 1.0)
            track['importance'] = self._clamp(max(float(track.get('importance', 0.15) or 0.15), 0.22), 0.0, 1.0)
            track['stability'] = max(float(track.get('stability', 0.0) or 0.0), 0.75)
            track['revision'] = int(track.get('revision', 0) or 0) + 1
            track['last_labeled_ts'] = now
            track['label_source_text'] = str(payload.get('source_text', '') or '')[:160]
            track['fallback_ref'] = self._render_fallback_ref(track)
            snapshot = self._snapshot(track)
            changed = True
            break

        if not changed or snapshot is None:
            return []

        await ctx.set_kv('vision:proto:tracks', tracks)
        await ctx.set_kv('vision:proto:last_focus', snapshot)
        await self._write_state(ctx, tracks)

        return [
            Event(
                topic='vision/proto_object',
                payload=snapshot,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={'kind': 'vision_proto_object', 'status': 'labeled', 'label_applied': True},
            ),
            Event(
                topic='reason/output',
                payload={'text': f"visual proto-object resolved: {snapshot.get('fallback_ref', 'that thing')} is {label}"},
                source=self.name,
                correlation_id=event.correlation_id,
                meta={'channel': 'thought', 'kind': 'vision_proto_labeled', 'lobe': 'vision'},
            ),
        ]

    def _attach_or_create_track(
        self,
        *,
        tracks: List[Dict[str, Any]],
        obs: Dict[str, Any],
        now: float,
        stale_s: float,
        match_max_bits: int,
        focus_proximity: float,
        curiosity_start: float,
        importance_start: float,
    ) -> tuple[Dict[str, Any], bool]:
        best_track = None
        best_score = -1.0
        obs_sig = int(obs['signature'])
        obs_focus = obs['focus_xy']

        for track in tracks:
            last_seen = float(track.get('last_seen', 0.0) or 0.0)
            if (now - last_seen) > stale_s:
                continue
            try:
                dist = (int(track.get('signature', 0) or 0) ^ obs_sig).bit_count()
            except Exception:
                dist = 64
            if dist > match_max_bits:
                continue
            tfocus = track.get('focus_xy', {}) or {}
            dx = float(tfocus.get('x', 0.5) or 0.5) - float(obs_focus.get('x', 0.5) or 0.5)
            dy = float(tfocus.get('y', 0.5) or 0.5) - float(obs_focus.get('y', 0.5) or 0.5)
            proximity = math.sqrt(dx * dx + dy * dy)
            if proximity > focus_proximity:
                continue
            score = (1.0 - (dist / max(1, match_max_bits))) + (1.0 - (proximity / max(0.001, focus_proximity)))
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None:
            return best_track, False

        seed = f"{obs['frame_ref']}|{now}|{obs_sig}|{obs_focus.get('x', 0.5):.3f}|{obs_focus.get('y', 0.5):.3f}"
        proto_id = 'vobj:' + hashlib.blake2b(seed.encode('utf-8', errors='ignore'), digest_size=8).hexdigest()
        track = {
            'id': proto_id,
            'kind': 'vision_proto_object.v1',
            'first_seen': now,
            'last_seen': now,
            'frame_ref': obs['frame_ref'],
            'signature': obs_sig,
            'focus_xy': dict(obs_focus),
            'focus_radius': float(obs.get('focus_radius', 0.08) or 0.08),
            'crop_box': dict(obs['crop_box']),
            'brightness': float(obs['brightness']),
            'edge_energy': float(obs['edge_energy']),
            'contrast_energy': float(obs.get('contrast_energy', obs['edge_energy']) or obs['edge_energy']),
            'seen_count': 1,
            'revision': 0,
            'stability': 0.0,
            'curiosity': curiosity_start,
            'importance': importance_start,
            'deictic': 'that',
            'term': 'thing',
            'modifiers': ['unknown'],
            'labels': [],
            'resolved_label': '',
            'status': 'unknown',
            'last_emit_ts': 0.0,
            'last_question_ts': 0.0,
        }
        track['fallback_ref'] = self._render_fallback_ref(track)
        tracks.append(track)
        return track, True

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    def _importance_delta(self, *, event: Event, payload: Dict[str, Any]) -> float:
        tags: List[str] = []
        raw_tags = payload.get('tags', [])
        if isinstance(raw_tags, list):
            tags.extend(str(t).strip().lower() for t in raw_tags if str(t).strip())
        meta_tags = event.meta.get('tags', []) if isinstance(event.meta, dict) else []
        if isinstance(meta_tags, list):
            tags.extend(str(t).strip().lower() for t in meta_tags if str(t).strip())
        tagset = set(tags)
        delta = 0.0
        if {'hazard', 'danger', 'boundary_warning'} & tagset:
            delta += 0.45
        if {'power', 'maintenance', 'task_relevant'} & tagset:
            delta += 0.20
        if {'background', 'routine', 'low_value'} & tagset:
            delta -= 0.08
        return delta

    def _should_emit_snapshot(self, *, track: Dict[str, Any], is_new: bool, min_repeat: int, now: float) -> bool:
        if is_new:
            return True
        if str(track.get('resolved_label', '') or '').strip() and float(track.get('last_emit_ts', 0.0) or 0.0) <= 0.0:
            return True
        seen_count = int(track.get('seen_count', 1) or 1)
        last_emit_ts = float(track.get('last_emit_ts', 0.0) or 0.0)
        if seen_count in {min_repeat, min_repeat + 2}:
            return True
        if (now - last_emit_ts) >= 20.0 and float(track.get('stability', 0.0) or 0.0) >= 0.55:
            return True
        return False

    def _snapshot(self, track: Dict[str, Any]) -> Dict[str, Any]:
        resolved = str(track.get('resolved_label', '') or '').strip().lower()
        labels = [str(x).strip().lower() for x in list(track.get('labels', []) or []) if str(x).strip()]
        return {
            'proto_id': str(track.get('id', '') or ''),
            'kind': 'vision_proto_object.v1',
            'status': 'labeled' if resolved else 'unknown',
            'deictic': str(track.get('deictic', 'that') or 'that'),
            'term': str(track.get('term', 'thing') or 'thing'),
            'modifiers': list(track.get('modifiers', []) or []),
            'fallback_ref': self._render_fallback_ref(track),
            'resolved_label': resolved,
            'labels': labels,
            'curiosity': round(float(track.get('curiosity', 0.0) or 0.0), 4),
            'importance': round(float(track.get('importance', 0.0) or 0.0), 4),
            'stability': round(float(track.get('stability', 0.0) or 0.0), 4),
            'seen_count': int(track.get('seen_count', 1) or 1),
            'focus_xy': dict(track.get('focus_xy', {}) or {}),
            'focus_radius': round(float(track.get('focus_radius', 0.08) or 0.08), 4),
            'crop_box': dict(track.get('crop_box', {}) or {}),
            'brightness': round(float(track.get('brightness', 0.0) or 0.0), 4),
            'edge_energy': round(float(track.get('edge_energy', 0.0) or 0.0), 4),
            'contrast_energy': round(float(track.get('contrast_energy', 0.0) or 0.0), 4),
            'frame_ref': str(track.get('frame_ref', '') or ''),
            'first_seen': float(track.get('first_seen', 0.0) or 0.0),
            'last_seen': float(track.get('last_seen', 0.0) or 0.0),
            'should_ask': bool(track.get('should_ask', False)),
        }

    def _internal_note(self, snapshot: Dict[str, Any]) -> str:
        if snapshot.get('resolved_label'):
            return (
                f"visual proto-object stabilized: {snapshot.get('fallback_ref', 'that thing')} "
                f"maps to {snapshot.get('resolved_label')}"
            )
        return (
            f"visual proto-object: {snapshot.get('fallback_ref', 'that thing')} "
            f"seen {snapshot.get('seen_count', 1)} times | curiosity={snapshot.get('curiosity', 0.0):.2f} "
            f"importance={snapshot.get('importance', 0.0):.2f} focus_r={snapshot.get('focus_radius', 0.08):.2f}"
        )

    def _question_for_track(self, snapshot: Dict[str, Any]) -> str:
        """Canned proto-object questions removed; use thought payload instead."""
        return ""

    def _render_fallback_ref(self, track: Dict[str, Any]) -> str:
        resolved = str(track.get('resolved_label', '') or '').strip().lower()
        if resolved:
            return f"that {resolved}"
        deictic = str(track.get('deictic', 'that') or 'that').strip().lower()
        term = str(track.get('term', 'thing') or 'thing').strip().lower()
        modifiers = [str(x).strip().lower() for x in list(track.get('modifiers', []) or []) if str(x).strip()]
        parts = [deictic]
        if modifiers:
            parts.extend(modifiers[:2])
        parts.append(term)
        return ' '.join(parts)

    def _prune_tracks(self, *, tracks: List[Dict[str, Any]], now: float, stale_s: float, max_tracks: int) -> List[Dict[str, Any]]:
        keep: List[Dict[str, Any]] = []
        for track in tracks:
            last_seen = float(track.get('last_seen', 0.0) or 0.0)
            resolved = bool(str(track.get('resolved_label', '') or '').strip())
            ttl = stale_s * (3.0 if resolved else 1.0)
            if (now - last_seen) <= ttl:
                keep.append(track)
        keep.sort(key=lambda row: (float(row.get('importance', 0.0) or 0.0), float(row.get('last_seen', 0.0) or 0.0)), reverse=True)
        return keep[:max_tracks]

    async def _write_state(self, ctx, tracks: List[Dict[str, Any]]) -> None:
        memdir = await resolve_memdir_ctx(ctx, fallback=None)
        if not memdir:
            return
        out_dir = Path(memdir) / 'state'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vision_proto_tracks.json'
        payload = {
            'schema': 'vision_proto_tracks.v1',
            'ts': time.time(),
            'tracks': [self._snapshot(track) for track in tracks],
        }
        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    def _load_image(self, frame_ref: str):
        try:
            from PIL import Image
        except Exception:
            return None
        try:
            image = Image.open(frame_ref)
            image.load()
            return image.convert('RGB')
        except Exception:
            return None

    def _make_observation(self, *, image, frame_ref: str, focus_xy: Any, crop_px: int) -> Optional[Dict[str, Any]]:
        try:
            from PIL import Image
        except Exception:
            return None

        width, height = image.size
        if width <= 0 or height <= 0:
            return None

        fx = 0.5
        fy = 0.5
        focus_radius = 0.08
        if isinstance(focus_xy, dict):
            try:
                fx = float(focus_xy.get('x', focus_xy.get('cx', 0.5)) or 0.5)
                fy = float(focus_xy.get('y', focus_xy.get('cy', 0.5)) or 0.5)
                focus_radius = float(focus_xy.get('radius', 0.08) or 0.08)
            except Exception:
                fx, fy, focus_radius = 0.5, 0.5, 0.08
        fx = self._clamp(fx, 0.0, 1.0)
        fy = self._clamp(fy, 0.0, 1.0)
        focus_radius = self._clamp(focus_radius, 0.03, 0.35)

        derived_half = int(round(min(width, height) * focus_radius))
        half = max(24, min(int(crop_px // 2), max(24, derived_half)))
        cx = int(round(fx * (width - 1)))
        cy = int(round(fy * (height - 1)))
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(width, cx + half)
        bottom = min(height, cy + half)
        if right <= left or bottom <= top:
            return None

        crop = image.crop((left, top, right, bottom)).convert('L')
        resized = crop.resize((9, 9), Image.Resampling.BILINEAR)
        pixels = list(resized.getdata())
        rows = [pixels[idx * 9:(idx + 1) * 9] for idx in range(9)]

        bits = 0
        edge_sum = 0.0
        contrast_accum = 0.0
        for y in range(8):
            for x in range(8):
                a = int(rows[y][x])
                bx = int(rows[y][x + 1])
                by = int(rows[y + 1][x])
                dx = abs(bx - a)
                dy = abs(by - a)
                edge_sum += dx + dy
                contrast_accum += max(dx, dy)
                bits = (bits << 1) | int(bx > a)

        crop_pixels = list(crop.getdata())
        brightness = sum(int(p) for p in crop_pixels) / max(1, crop.width * crop.height * 255.0)
        edge_energy = edge_sum / (8.0 * 8.0 * 255.0 * 2.0)
        contrast_energy = contrast_accum / (8.0 * 8.0 * 255.0)

        return {
            'frame_ref': frame_ref,
            'signature': bits,
            'brightness': brightness,
            'edge_energy': edge_energy,
            'contrast_energy': contrast_energy,
            'focus_radius': focus_radius,
            'focus_xy': {'x': fx, 'y': fy, 'radius': focus_radius},
            'crop_box': {
                'left': int(left),
                'top': int(top),
                'right': int(right),
                'bottom': int(bottom),
                'width': int(right - left),
                'height': int(bottom - top),
            },
        }


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['percept/vision', 'vision/proto_label'],
        output_topics=['vision/proto_object', 'reason/output', 'thought/internal'],
        priority=5,
    )
    yield ProtoObjectTrackerNeuron(cfg)
