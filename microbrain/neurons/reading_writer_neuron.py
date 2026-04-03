from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class ReadingWriterNeuron(BaseNeuron):
    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != 'act/speech':
            return []
        payload = event.payload if isinstance(event.payload, dict) else {'text': event.payload}
        text = str(payload.get('text', '') or '').strip()
        channel = str(payload.get('channel', 'repl') or 'repl')
        if not text or channel in ('thought', 'internal'):
            return []
        enabled = bool(await ctx.get_kv('reading:writer_enabled', True))
        if not enabled:
            return []
        memdir = Path(await resolve_memdir_ctx(ctx, fallback=r'Z:\memory'))
        reading_dir = memdir / 'reading'
        queue_dir = reading_dir / 'queue'
        archive_dir = reading_dir / 'archive'
        queue_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = time.time()
        fname = f"read_{int(ts * 1000)}_{event.correlation_id[:8]}.txt"
        path = queue_dir / fname
        path.write_text(text, encoding='utf-8')
        await ctx.set_kv('reading:last_written', {
            'ts': ts,
            'path': str(path),
            'text': text,
            'channel': channel,
        })
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['act/speech'],
        output_topics=[],
        priority=-4,
        cooldown_sec=0.0,
    )
    yield ReadingWriterNeuron(cfg)
