from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pypdf import PdfReader

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class ReadingModeNeuron(BaseNeuron):
    def _state_path(self, read_dir: Path) -> Path:
        return read_dir / '_read_state.json'

    def _load_state_file(self, read_dir: Path) -> Dict[str, Any]:
        path = self._state_path(read_dir)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _save_state_file(self, read_dir: Path, data: Dict[str, Any]) -> None:
        path = self._state_path(read_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)

    def _list_candidates(self, read_dir: Path) -> List[Path]:
        out: List[Path] = []
        for path in sorted(read_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith('_read_state'):
                continue
            if path.suffix.lower() not in ('.txt', '.pdf', '.md'):
                continue
            out.append(path)
        return out

    def _read_text_lines(self, path: Path) -> List[str]:
        try:
            return path.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return []

    def _read_pdf_pages(self, path: Path) -> List[str]:
        try:
            reader = PdfReader(str(path))
        except Exception:
            return []
        pages: List[str] = []
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or '').strip())
            except Exception:
                pages.append('')
        return pages

    def _txt_chunk(self, path: Path, chunk_index: int, chunk_lines: int) -> Optional[Dict[str, Any]]:
        lines = self._read_text_lines(path)
        if not lines:
            return None
        start = chunk_index * chunk_lines
        if start >= len(lines):
            return None
        end = min(len(lines), start + chunk_lines)
        picked = lines[start:end]
        text = '\n'.join(picked).strip()
        if not text:
            return None
        return {
            'kind': 'text',
            'text': text,
            'chunk_index': chunk_index,
            'start_line': start + 1,
            'end_line': end,
            'summary': f"{path.name} lines {start + 1}-{end}",
        }

    def _pdf_chunk(self, path: Path, chunk_index: int, chunk_chars: int) -> Optional[Dict[str, Any]]:
        pages = self._read_pdf_pages(path)
        flat: List[Tuple[int, str]] = []
        for idx, page_text in enumerate(pages):
            if page_text.strip():
                flat.append((idx + 1, page_text.strip()))
        if not flat:
            return None
        if chunk_index >= len(flat):
            return None
        page_no, page_text = flat[chunk_index]
        clipped = page_text[: max(400, chunk_chars)].strip()
        if not clipped:
            return None
        return {
            'kind': 'pdf',
            'text': clipped,
            'chunk_index': chunk_index,
            'page': page_no,
            'summary': f"{path.name} page {page_no}",
        }

    def _chunk_for(self, path: Path, chunk_index: int, chunk_lines: int, chunk_chars: int) -> Optional[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in ('.txt', '.md'):
            return self._txt_chunk(path, chunk_index, chunk_lines)
        if suffix == '.pdf':
            return self._pdf_chunk(path, chunk_index, chunk_chars)
        return None

    def _ingest_piece(self, mem_cell_store: MemCellStore, text: str, source_name: str) -> int:
        pieces = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not pieces:
            pieces = [text.strip()]
        count = 0
        for piece in pieces[:4]:
            if len(piece) > 900:
                piece = piece[:900].rsplit(' ', 1)[0].strip() or piece[:900]
            result = mem_cell_store.ingest_text(
                text=piece,
                topic='reading/text',
                role='assistant',
                transport_source='reading',
                source=source_name,
                meta={'channel': 'reading'},
                tier='now',
            )
            rows = [result.get('utterance')] + list(result.get('tokens', [])) + list(result.get('patterns', []))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row['activation'] = min(float(row.get('activation', 0.18) or 0.18), 0.18)
                row['promotion'] = min(float(row.get('promotion', 0.0) or 0.0), 0.0)
                row['trust'] = min(float(row.get('trust', 0.3) or 0.3), 0.30)
                mem_cell_store.upsert_cell(row, tier=str(row.get('tier', 'now') or 'now'))
            count += 1
        return count

    async def _mem_cell_store(self, ctx) -> Optional[MemCellStore]:
        store = await ctx.get_kv('memory:mem_cell_store', None)
        if isinstance(store, MemCellStore):
            return store
        try:
            memdir = await resolve_memdir_ctx(ctx, fallback=r'Z:\memory')
            store = MemCellStore(memdir)
            await ctx.set_kv('memory:mem_cell_store', store)
            return store
        except Exception:
            return None

    async def _step_once(self, ctx, force: bool = False) -> List[Event]:
        read_dir = Path(str(await ctx.get_kv('read:dir', '') or '')).expanduser()
        read_dir.mkdir(parents=True, exist_ok=True)
        ready_dir = read_dir / 'ready'
        ready_dir.mkdir(parents=True, exist_ok=True)

        enabled = bool(await ctx.get_kv('read:enabled', False))
        if not enabled and not force:
            return []
        now = time.time()
        idle_after_s = float(await ctx.get_kv('read:idle_after_s', 90.0) or 90.0)
        tick_every_s = float(await ctx.get_kv('read:tick_every_s', 30.0) or 30.0)
        chunk_lines = int(await ctx.get_kv('read:chunk_lines', 40) or 40)
        chunk_chars = int(await ctx.get_kv('read:chunk_chars', 1200) or 1200)
        last_user_ts = float(await ctx.get_kv('read:last_user_ts', 0.0) or 0.0)
        last_activity_ts = float(await ctx.get_kv('read:last_activity_ts', 0.0) or 0.0)
        if not force:
            if (now - last_user_ts) < idle_after_s:
                return []
            if (now - last_activity_ts) < tick_every_s:
                return []

        state = self._load_state_file(read_dir)
        active_file = str(await ctx.get_kv('read:active_file', '') or state.get('active_file', '') or '')
        chunk_index = int(await ctx.get_kv('read:chunk_index', state.get('chunk_index', 0)) or 0)

        path = Path(active_file) if active_file else None
        if path is None or not path.exists():
            candidates = self._list_candidates(read_dir)
            if not candidates:
                result = {'ts': now, 'summary': 'no readable files in read_dir'}
                await ctx.set_kv('read:last_result', result)
                await ctx.set_kv('read:last_activity_ts', now)
                self._save_state_file(read_dir, {'active_file': '', 'chunk_index': 0, 'last_result': result})
                return []
            path = candidates[0]
            chunk_index = 0

        chunk = self._chunk_for(path, chunk_index, chunk_lines, chunk_chars)
        if chunk is None:
            target = ready_dir / path.name
            if target.exists():
                stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(now))
                target = ready_dir / f"{path.stem}-{stamp}{path.suffix}"
            try:
                shutil.move(str(path), str(target))
            except Exception:
                pass
            await ctx.set_kv('read:active_file', '')
            await ctx.set_kv('read:active_kind', '')
            await ctx.set_kv('read:chunk_index', 0)
            result = {'ts': now, 'summary': f'{path.name} moved to ready'}
            await ctx.set_kv('read:last_result', result)
            await ctx.set_kv('read:last_activity_ts', now)
            self._save_state_file(read_dir, {'active_file': '', 'chunk_index': 0, 'last_result': result})
            return []

        mem_cell_store = await self._mem_cell_store(ctx)
        stored = 0
        if mem_cell_store is not None:
            stored = self._ingest_piece(mem_cell_store, chunk['text'], path.name)

        next_index = int(chunk.get('chunk_index', chunk_index)) + 1
        await ctx.set_kv('read:active_file', str(path))
        await ctx.set_kv('read:active_kind', str(chunk.get('kind', '') or ''))
        await ctx.set_kv('read:chunk_index', next_index)
        await ctx.set_kv('read:last_activity_ts', now)
        result = {
            'ts': now,
            'file': str(path),
            'kind': str(chunk.get('kind', '') or ''),
            'chunk_index': int(chunk.get('chunk_index', chunk_index)),
            'stored_count': stored,
            'summary': str(chunk.get('summary', path.name)),
        }
        if 'start_line' in chunk:
            result['start_line'] = int(chunk['start_line'])
            result['end_line'] = int(chunk['end_line'])
        if 'page' in chunk:
            result['page'] = int(chunk['page'])
        await ctx.set_kv('read:last_result', result)
        self._save_state_file(read_dir, {'active_file': str(path), 'chunk_index': next_index, 'last_result': result})
        return []

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic == 'percept/text':
            payload = event.payload if isinstance(event.payload, dict) else {}
            if str(payload.get('source', 'user') or 'user') == 'user':
                await ctx.set_kv('read:last_user_ts', time.time())
            return []

        if event.topic == 'control/read':
            payload = event.payload if isinstance(event.payload, dict) else {}
            cmd = str(payload.get('command', '') or '').lower()
            if cmd in ('on', 'start'):
                await ctx.set_kv('read:enabled', True)
                await ctx.set_kv('read:last_activity_ts', 0.0)
                return []
            if cmd in ('off', 'stop'):
                await ctx.set_kv('read:enabled', False)
                return []
            if cmd in ('next', 'step'):
                return await self._step_once(ctx, force=True)
            return []

        if event.topic == 'clock/tick':
            return await self._step_once(ctx, force=False)

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=['clock/tick', 'percept/text', 'control/read'],
        output_topics=[],
        priority=-3,
        cooldown_sec=0.0,
    )
    yield ReadingModeNeuron(cfg)
