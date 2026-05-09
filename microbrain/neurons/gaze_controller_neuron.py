from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class GazeControllerNeuron(BaseNeuron):
    """
    Governed visual attention controller.

    Purpose:
      - make the focus reticle movable without turning it into a free cursor
      - let MB roam slowly, dwell, tighten focus on unknowns, and micro-nudge
      - keep all motion on a budget so vision does not machine-scan a room instantly

    This organ only changes focus state. Other neurons decide what the focused
    region means.
    """

    ROAM_POINTS: Tuple[Tuple[float, float], ...] = (
        (0.50, 0.50),
        (0.36, 0.36),
        (0.64, 0.36),
        (0.36, 0.64),
        (0.64, 0.64),
        (0.50, 0.28),
        (0.72, 0.50),
        (0.50, 0.72),
        (0.28, 0.50),
    )
    MICRO_OFFSETS: Tuple[Tuple[float, float], ...] = (
        (0.00, 0.00),
        (0.015, 0.00),
        (-0.015, 0.00),
        (0.00, 0.015),
        (0.00, -0.015),
        (0.012, 0.012),
        (-0.012, 0.012),
        (0.012, -0.012),
        (-0.012, -0.012),
    )

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic == 'control/focus':
            return await self._manual_focus(event, ctx)
        if event.topic == 'vision/proto_object':
            return await self._on_proto_object(event, ctx)
        if event.topic != 'clock/tick':
            return []
        return await self._tick(event, ctx)

    async def _load_state(self, ctx) -> Dict[str, Any]:
        state = dict(await ctx.get_kv('vision:gaze_state', {}) or {})
        if not state:
            state = {
                'x': 0.5,
                'y': 0.5,
                'radius': 0.12,
                'mode': 'roam',
                'target_proto_id': '',
                'dwell_until': 0.0,
                'settle_until': 0.0,
                'move_budget': 1.0,
                'last_tick_ts': 0.0,
                'last_move_ts': 0.0,
                'roam_index': 0,
                'inspect_step_idx': 0,
                'why': 'init',
                'last_question_text': '',
                'last_question_ts': 0.0,
            }
        state['x'] = self._clamp(float(state.get('x', 0.5) or 0.5), 0.0, 1.0)
        state['y'] = self._clamp(float(state.get('y', 0.5) or 0.5), 0.0, 1.0)
        state['radius'] = self._clamp(float(state.get('radius', 0.12) or 0.12), 0.03, 0.35)
        state['move_budget'] = self._clamp(float(state.get('move_budget', 1.0) or 1.0), 0.0, 1.5)
        state['mode'] = str(state.get('mode', 'roam') or 'roam')
        return state

    async def _save_state(self, ctx, state: Dict[str, Any]) -> None:
        await ctx.set_kv('vision:gaze_state', state)
        await ctx.set_kv(
            'vision:focus_xy',
            {
                'x': state.get('x', 0.5),
                'y': state.get('y', 0.5),
                'radius': state.get('radius', 0.12),
                'mode': state.get('mode', 'roam'),
            },
        )

    async def _manual_focus(self, event: Event, ctx) -> List[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        action = str(payload.get('action', '') or '').strip().lower()
        state = await self._load_state(ctx)
        if action == 'center':
            state.update({
                'x': 0.5,
                'y': 0.5,
                'radius': 0.12,
                'mode': 'manual',
                'target_proto_id': '',
                'dwell_until': time.time() + 1.0,
                'settle_until': time.time() + 0.25,
                'why': 'manual:center',
            })
        elif action == 'set':
            state.update({
                'x': self._clamp(float(payload.get('x', state.get('x', 0.5)) or state.get('x', 0.5)), 0.0, 1.0),
                'y': self._clamp(float(payload.get('y', state.get('y', 0.5)) or state.get('y', 0.5)), 0.0, 1.0),
                'mode': 'manual',
                'target_proto_id': '',
                'dwell_until': time.time() + 1.0,
                'settle_until': time.time() + 0.25,
                'why': 'manual:set',
            })
        await self._save_state(ctx, state)
        return []

    async def _on_proto_object(self, event: Event, ctx) -> List[Event]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        proto_id = str(payload.get('proto_id', '') or '').strip()
        if not proto_id:
            return []
        state = await self._load_state(ctx)
        now = time.time()
        focus_xy = payload.get('focus_xy', {}) if isinstance(payload.get('focus_xy'), dict) else {}
        fx = self._clamp(float(focus_xy.get('x', state.get('x', 0.5)) or state.get('x', 0.5)), 0.0, 1.0)
        fy = self._clamp(float(focus_xy.get('y', state.get('y', 0.5)) or state.get('y', 0.5)), 0.0, 1.0)
        status = str(payload.get('status', 'unknown') or 'unknown').lower()
        curiosity = float(payload.get('curiosity', 0.0) or 0.0)
        stability = float(payload.get('stability', 0.0) or 0.0)
        should_ask = bool(payload.get('should_ask', False))

        state['x'] = fx
        state['y'] = fy
        if status == 'labeled':
            state['mode'] = 'release'
            state['radius'] = self._clamp(max(float(state.get('radius', 0.12) or 0.12), 0.12) + 0.02, 0.03, 0.35)
            state['target_proto_id'] = ''
            state['dwell_until'] = now + 0.8
            state['settle_until'] = now + 0.25
            state['why'] = f'proto:labeled:{proto_id}'
        else:
            state['target_proto_id'] = proto_id
            if should_ask or stability >= 0.55:
                state['mode'] = 'lock'
                state['radius'] = self._clamp(min(float(state.get('radius', 0.12) or 0.12), 0.08), 0.03, 0.35)
                state['dwell_until'] = now + 2.6
                state['why'] = f'proto:lock:{proto_id}'
            elif curiosity >= 0.72 or stability >= 0.30:
                state['mode'] = 'inspect'
                state['radius'] = self._clamp(min(float(state.get('radius', 0.12) or 0.12), 0.10), 0.03, 0.35)
                state['dwell_until'] = now + 1.5
                state['why'] = f'proto:inspect:{proto_id}'
            else:
                state['mode'] = 'hold'
                state['radius'] = self._clamp(min(float(state.get('radius', 0.12) or 0.12), 0.12), 0.03, 0.35)
                state['dwell_until'] = now + 1.0
                state['why'] = f'proto:hold:{proto_id}'
            state['settle_until'] = now + 0.25
        await self._save_state(ctx, state)
        return []

    async def _tick(self, event: Event, ctx) -> List[Event]:
        if not bool(await ctx.get_kv('vision:enabled', False)):
            return []

        state = await self._load_state(ctx)
        now = time.time()
        last_tick_ts = float(state.get('last_tick_ts', 0.0) or 0.0)
        dt = max(0.0, now - last_tick_ts) if last_tick_ts > 0.0 else 0.5
        state['last_tick_ts'] = now

        refill_per_s = float(await ctx.get_kv('vision:gaze:budget_refill_per_s', 0.14) or 0.14)
        state['move_budget'] = self._clamp(float(state.get('move_budget', 1.0) or 1.0) + refill_per_s * dt, 0.0, 1.5)

        if now < float(state.get('settle_until', 0.0) or 0.0):
            await self._save_state(ctx, state)
            return []

        if now < float(state.get('dwell_until', 0.0) or 0.0):
            await self._save_state(ctx, state)
            return []

        mode = str(state.get('mode', 'roam') or 'roam')
        outputs: List[Event] = []
        if mode in {'inspect', 'lock'} and float(state.get('move_budget', 0.0) or 0.0) >= 0.18:
            self._micro_nudge(state)
            state['move_budget'] = self._clamp(float(state.get('move_budget', 0.0) or 0.0) - 0.18, 0.0, 1.5)
            state['settle_until'] = now + float(await ctx.get_kv('vision:gaze:settle_s', 0.22) or 0.22)
            state['dwell_until'] = now + float(await ctx.get_kv('vision:gaze:inspect_dwell_s', 1.0) or 1.0)
            state['last_move_ts'] = now
            state['why'] = f'{mode}:micro'
        elif float(state.get('move_budget', 0.0) or 0.0) >= 0.24 and mode not in {'manual'}:
            self._roam_step(state)
            state['move_budget'] = self._clamp(float(state.get('move_budget', 0.0) or 0.0) - 0.24, 0.0, 1.5)
            state['settle_until'] = now + float(await ctx.get_kv('vision:gaze:settle_s', 0.22) or 0.22)
            state['dwell_until'] = now + float(await ctx.get_kv('vision:gaze:roam_dwell_s', 1.6) or 1.6)
            state['last_move_ts'] = now
            state['why'] = 'roam:step'

        state['radius'] = self._next_radius(state)
        await self._save_state(ctx, state)

        if bool(await ctx.get_kv('vision:gaze:emit_state_notes', False)):
            outputs.append(
                Event(
                    topic='reason/output',
                    payload={
                        'text': (
                            f"gaze {state.get('mode', 'roam')} @ ({state.get('x', 0.5):.3f}, {state.get('y', 0.5):.3f}) "
                            f"r={state.get('radius', 0.12):.3f} budget={state.get('move_budget', 0.0):.2f}"
                        )
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={'channel': 'thought', 'kind': 'vision_gaze_state', 'lobe': 'vision'},
                )
            )
        return outputs

    def _roam_step(self, state: Dict[str, Any]) -> None:
        idx = int(state.get('roam_index', 0) or 0)
        x, y = self.ROAM_POINTS[idx % len(self.ROAM_POINTS)]
        state['roam_index'] = idx + 1
        state['x'] = self._clamp((float(state.get('x', 0.5) or 0.5) * 0.25) + (x * 0.75), 0.0, 1.0)
        state['y'] = self._clamp((float(state.get('y', 0.5) or 0.5) * 0.25) + (y * 0.75), 0.0, 1.0)
        if str(state.get('mode', 'roam') or 'roam') == 'release':
            state['mode'] = 'roam'
        elif str(state.get('mode', 'roam') or 'roam') not in {'inspect', 'lock'}:
            state['mode'] = 'roam'

    def _micro_nudge(self, state: Dict[str, Any]) -> None:
        idx = int(state.get('inspect_step_idx', 0) or 0)
        dx, dy = self.MICRO_OFFSETS[idx % len(self.MICRO_OFFSETS)]
        state['inspect_step_idx'] = idx + 1
        base_x = float(state.get('x', 0.5) or 0.5)
        base_y = float(state.get('y', 0.5) or 0.5)
        state['x'] = self._clamp(base_x + dx, 0.0, 1.0)
        state['y'] = self._clamp(base_y + dy, 0.0, 1.0)

    def _next_radius(self, state: Dict[str, Any]) -> float:
        mode = str(state.get('mode', 'roam') or 'roam')
        current = float(state.get('radius', 0.12) or 0.12)
        if mode == 'roam':
            return self._clamp(current * 0.92 + 0.18 * 0.08, 0.03, 0.35)
        if mode == 'release':
            return self._clamp(current + 0.02, 0.03, 0.35)
        if mode == 'hold':
            return self._clamp(current * 0.96, 0.03, 0.35)
        if mode == 'inspect':
            return self._clamp(current * 0.92, 0.03, 0.35)
        if mode == 'lock':
            return self._clamp(current * 0.88, 0.03, 0.35)
        return self._clamp(current, 0.03, 0.35)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['clock/tick', 'vision/proto_object', 'control/focus'],
        output_topics=['reason/output'],
        priority=6,
    )
    yield GazeControllerNeuron(cfg)
