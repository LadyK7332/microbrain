from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class ReadingWriterNeuron(BaseNeuron):
    """
    Explicit reading-queue writer.

    Important separation:
      - mouth/textual output should not be mirrored into reading ingestion
      - only dedicated reading queue events should write files here
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != 'reading/queue_text':
            return []
        payload = event.payload if isinstance(event.payload, dict) else {'text': event.payload}
        text = str(payload.get('text', '') or '').strip()
        if not text:
            return []
        enabled = bool(await ctx.get_kv('reading:writer_enabled', False))
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
            'channel': 'reading',
            'source_topic': event.topic,
        })
        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['reading/queue_text'],
        output_topics=[],
        priority=-4,
        cooldown_sec=0.0,
    )
    yield ReadingWriterNeuron(cfg)
