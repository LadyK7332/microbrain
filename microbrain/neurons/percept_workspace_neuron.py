from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class PerceptWorkspaceNeuron(BaseNeuron):
    """
    Short-lived visual comparison workspace.

    This is the "hold it briefly before memory" layer:
      - keeps recent proto-object observations together
      - compares repeated sightings of the same proto-object
      - only emits a commit event when the percept survives comparison
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != 'vision/proto_object':
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        proto_id = str(payload.get('proto_id', '') or '').strip()
        if not proto_id:
            return []

        now = time.time()
        workspace = dict(await ctx.get_kv('vision:workspace', {}) or {})
        ttl_s = float(await ctx.get_kv('vision:workspace:ttl_s', 25.0) or 25.0)
        min_repeat = int(await ctx.get_kv('vision:workspace:min_repeat', 3) or 3)
        commit_stability = float(await ctx.get_kv('vision:workspace:commit_stability', 0.58) or 0.58)
        candidates = dict(workspace.get('candidates', {}) or {})

        pruned: Dict[str, Any] = {}
        for key, row in candidates.items():
            last_seen = float(row.get('last_seen', 0.0) or 0.0)
            if (now - last_seen) <= ttl_s:
                pruned[key] = row
        candidates = pruned

        row = dict(candidates.get(proto_id, {}) or {})
        row['proto_id'] = proto_id
        row['status'] = str(payload.get('status', row.get('status', 'unknown')) or 'unknown')
        row['fallback_ref'] = str(payload.get('fallback_ref', row.get('fallback_ref', 'that thing')) or 'that thing')
        row['resolved_label'] = str(payload.get('resolved_label', row.get('resolved_label', '')) or '').strip().lower()
        row['first_seen'] = float(row.get('first_seen', now) or now)
        row['last_seen'] = now
        row['count'] = int(row.get('count', 0) or 0) + 1
        row['max_stability'] = max(float(row.get('max_stability', 0.0) or 0.0), float(payload.get('stability', 0.0) or 0.0))
        row['max_curiosity'] = max(float(row.get('max_curiosity', 0.0) or 0.0), float(payload.get('curiosity', 0.0) or 0.0))
        row['mean_brightness'] = self._running_mean(
            float(row.get('mean_brightness', payload.get('brightness', 0.0)) or payload.get('brightness', 0.0) or 0.0),
            float(payload.get('brightness', 0.0) or 0.0),
            row['count'],
        )
        row['mean_edge_energy'] = self._running_mean(
            float(row.get('mean_edge_energy', payload.get('edge_energy', 0.0)) or payload.get('edge_energy', 0.0) or 0.0),
            float(payload.get('edge_energy', 0.0) or 0.0),
            row['count'],
        )
        row['mean_contrast_energy'] = self._running_mean(
            float(row.get('mean_contrast_energy', payload.get('contrast_energy', 0.0)) or payload.get('contrast_energy', 0.0) or 0.0),
            float(payload.get('contrast_energy', 0.0) or 0.0),
            row['count'],
        )
        row['focus_xy'] = dict(payload.get('focus_xy', row.get('focus_xy', {})) or {})
        row['focus_radius'] = float(payload.get('focus_radius', row.get('focus_radius', 0.08)) or row.get('focus_radius', 0.08) or 0.08)
        row['crop_box'] = dict(payload.get('crop_box', row.get('crop_box', {})) or {})
        row['last_frame_ref'] = str(payload.get('frame_ref', row.get('last_frame_ref', '')) or '')
        row['commit_emitted'] = bool(row.get('commit_emitted', False))
        candidates[proto_id] = row

        workspace['ts'] = now
        workspace['candidates'] = candidates
        await ctx.set_kv('vision:workspace', workspace)
        await ctx.set_kv('vision:workspace:last', row)
        await self._write_state(ctx, workspace)

        should_commit = bool(row.get('resolved_label')) or (
            int(row.get('count', 0) or 0) >= min_repeat
            and float(row.get('max_stability', 0.0) or 0.0) >= commit_stability
        )
        if not should_commit or bool(row.get('commit_emitted', False)):
            return []

        row['commit_emitted'] = True
        candidates[proto_id] = row
        workspace['candidates'] = candidates
        await ctx.set_kv('vision:workspace', workspace)
        await self._write_state(ctx, workspace)

        summary = self._summary_for(row)
        return [
            Event(
                topic='vision/percept_commit',
                payload={
                    'kind': 'vision_percept_commit.v1',
                    'proto_id': proto_id,
                    'status': row.get('status', 'unknown'),
                    'text': summary,
                    'resolved_label': row.get('resolved_label', ''),
                    'fallback_ref': row.get('fallback_ref', 'that thing'),
                    'count': int(row.get('count', 0) or 0),
                    'max_stability': float(row.get('max_stability', 0.0) or 0.0),
                    'max_curiosity': float(row.get('max_curiosity', 0.0) or 0.0),
                    'focus_xy': dict(row.get('focus_xy', {}) or {}),
                    'focus_radius': float(row.get('focus_radius', 0.08) or 0.08),
                    'crop_box': dict(row.get('crop_box', {}) or {}),
                    'brightness': float(row.get('mean_brightness', 0.0) or 0.0),
                    'edge_energy': float(row.get('mean_edge_energy', 0.0) or 0.0),
                    'contrast_energy': float(row.get('mean_contrast_energy', 0.0) or 0.0),
                    'frame_ref': row.get('last_frame_ref', ''),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={'kind': 'vision_percept_commit', 'status': row.get('status', 'unknown')},
            )
        ]

    @staticmethod
    def _running_mean(current: float, new_value: float, count: int) -> float:
        if count <= 1:
            return float(new_value)
        prev_n = max(1, count - 1)
        return ((float(current) * prev_n) + float(new_value)) / count

    @staticmethod
    def _summary_for(row: Dict[str, Any]) -> str:
        label = str(row.get('resolved_label', '') or '').strip().lower()
        fallback = str(row.get('fallback_ref', 'that thing') or 'that thing')
        count = int(row.get('count', 0) or 0)
        if label:
            return f"I kept seeing {fallback}; it resolved as {label}."
        return f"I kept seeing {fallback} {count} times, and it seems like the same object."

    async def _write_state(self, ctx, workspace: Dict[str, Any]) -> None:
        memdir = await resolve_memdir_ctx(ctx, fallback=None)
        if not memdir:
            return
        out_dir = Path(memdir) / 'state'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vision_workspace.json'
        try:
            import json
            out_path.write_text(json.dumps(workspace, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['vision/proto_object'],
        output_topics=['vision/percept_commit'],
        priority=5,
    )
    yield PerceptWorkspaceNeuron(cfg)
