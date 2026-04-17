from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.event_bus import Event
from microbrain.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class _Chunk:
    kind: str
    text: str
    chunk_index: int
    summary: str
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None


class ReadSidecar:
    """
    Background reading organ.

    Purpose:
    - keep /read control in the interaction layer
    - move file chewing / PDF parsing off the Textual hot path
    - nibble one chunk at a time so the rest of the system stays responsive
    """

    def __init__(self, orch: Orchestrator, *, memdir: str) -> None:
        self.orch = orch
        self.memdir = Path(memdir)
        self._wake_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._sub_id: str | None = None
        self._stop = False

    async def start(self) -> None:
        if self._loop_task is not None:
            return

        async def _on_control(ev: Event) -> List[Event]:
            payload = ev.payload if isinstance(ev.payload, dict) else {}
            cmd = str(payload.get("command", "") or "").strip().lower()
            if cmd == "on":
                self.orch.kv_store["read:enabled"] = True
            elif cmd == "off":
                self.orch.kv_store["read:enabled"] = False
            elif cmd in ("next", "step"):
                self.orch.kv_store["read:force_once"] = time.time()
            self._wake_event.set()
            return []

        self._sub_id = self.orch.bus.subscribe(
            "sidecar.read.control",
            ["control/read"],
            _on_control,
            priority=0,
        )
        self._loop_task = asyncio.create_task(self._run_loop(), name="read_sidecar")
        logger.info("Read sidecar started.")

    async def stop(self) -> None:
        self._stop = True
        self._wake_event.set()
        if self._loop_task is not None:
            try:
                await self._loop_task
            except Exception:
                logger.exception("Read sidecar stop failed")
        if self._sub_id:
            try:
                self.orch.bus.unsubscribe(self._sub_id)
            except Exception:
                pass

    async def _run_loop(self) -> None:
        while not self._stop:
            try:
                await self._run_once()
            except Exception:
                logger.exception("Read sidecar cycle failed")

            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self._wake_event.clear()

    async def _run_once(self) -> None:
        enabled = bool(self.orch.kv_store.get("read:enabled", False))
        force_ts = float(self.orch.kv_store.get("read:force_once", 0.0) or 0.0)
        force = force_ts > 0.0
        if not enabled and not force:
            return

        now = time.time()
        idle_after_s = float(self.orch.kv_store.get("read:idle_after_s", 90.0) or 90.0)
        tick_every_s = float(self.orch.kv_store.get("read:tick_every_s", 30.0) or 30.0)
        chunk_lines = int(self.orch.kv_store.get("read:chunk_lines", 40) or 40)
        chunk_chars = int(self.orch.kv_store.get("read:chunk_chars", 1200) or 1200)

        interaction_last = self.orch.kv_store.get("interaction:last_input", {}) or {}
        last_input_ts = float(interaction_last.get("ts", 0.0) or 0.0) if isinstance(interaction_last, dict) else 0.0
        last_activity_ts = float(self.orch.kv_store.get("read:last_activity_ts", 0.0) or 0.0)

        if not force:
            if (now - last_input_ts) < idle_after_s:
                return
            if (now - last_activity_ts) < tick_every_s:
                return

        read_dir = self._resolve_read_dir()
        ready_dir = read_dir / "ready"
        read_dir.mkdir(parents=True, exist_ok=True)
        ready_dir.mkdir(parents=True, exist_ok=True)

        state = await asyncio.to_thread(self._load_state_file, read_dir)
        active_file = str(self.orch.kv_store.get("read:active_file", "") or state.get("active_file", "") or "")
        chunk_index = int(self.orch.kv_store.get("read:chunk_index", state.get("chunk_index", 0)) or 0)

        path = Path(active_file) if active_file else None
        if path is None or not path.exists():
            candidates = await asyncio.to_thread(self._list_candidates, read_dir)
            if not candidates:
                result = {"ts": now, "summary": "no readable files in read_dir"}
                self._apply_status(result=result, active_file="", active_kind="", chunk_index=0, now=now)
                await asyncio.to_thread(
                    self._save_state_file,
                    read_dir,
                    {"active_file": "", "chunk_index": 0, "last_result": result},
                )
                if force:
                    self.orch.kv_store["read:force_once"] = 0.0
                return
            path = candidates[0]
            chunk_index = 0

        chunk = await asyncio.to_thread(self._chunk_for, path, chunk_index, chunk_lines, chunk_chars)
        if chunk is None:
            target = ready_dir / path.name
            if target.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
                target = ready_dir / f"{path.stem}-{stamp}{path.suffix}"
            try:
                await asyncio.to_thread(shutil.move, str(path), str(target))
            except Exception:
                logger.exception("Read sidecar failed to move completed file: %s", path)
            result = {"ts": now, "summary": f"{path.name} moved to ready"}
            self._apply_status(result=result, active_file="", active_kind="", chunk_index=0, now=now)
            await asyncio.to_thread(
                self._save_state_file,
                read_dir,
                {"active_file": "", "chunk_index": 0, "last_result": result},
            )
            if force:
                self.orch.kv_store["read:force_once"] = 0.0
            return

        mem_cell_store = self._mem_cell_store()
        ingested = 0
        if mem_cell_store is not None:
            ingested = await asyncio.to_thread(self._ingest_piece, mem_cell_store, chunk.text, path.name)

        next_index = chunk.chunk_index + 1
        result = {
            "ts": now,
            "summary": f"ingested {ingested} piece(s) from {chunk.summary}",
            "file": str(path),
            "kind": chunk.kind,
            "chunk_index": chunk.chunk_index,
        }
        self._apply_status(result=result, active_file=str(path), active_kind=chunk.kind, chunk_index=next_index, now=now)
        await asyncio.to_thread(
            self._save_state_file,
            read_dir,
            {
                "active_file": str(path),
                "chunk_index": next_index,
                "last_result": result,
            },
        )

        await self.orch.push_event(
            "read/status",
            dict(result),
            source="read_sidecar",
            meta={"source": "read_sidecar", "channel": "internal"},
        )

        if force:
            self.orch.kv_store["read:force_once"] = 0.0

    def _resolve_read_dir(self) -> Path:
        raw = str(self.orch.kv_store.get("read:dir", "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        default = self.memdir / "read_dir"
        self.orch.kv_store["read:dir"] = str(default)
        return default

    def _apply_status(self, *, result: Dict[str, Any], active_file: str, active_kind: str, chunk_index: int, now: float) -> None:
        self.orch.kv_store["read:last_result"] = result
        self.orch.kv_store["read:active_file"] = active_file
        self.orch.kv_store["read:active_kind"] = active_kind
        self.orch.kv_store["read:chunk_index"] = chunk_index
        self.orch.kv_store["read:last_activity_ts"] = now

    def _state_path(self, read_dir: Path) -> Path:
        return read_dir / "_read_state.json"

    def _load_state_file(self, read_dir: Path) -> Dict[str, Any]:
        path = self._state_path(read_dir)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state_file(self, read_dir: Path, data: Dict[str, Any]) -> None:
        path = self._state_path(read_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _list_candidates(self, read_dir: Path) -> List[Path]:
        out: List[Path] = []
        for path in sorted(read_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("_read_state"):
                continue
            if path.suffix.lower() not in (".txt", ".pdf", ".md"):
                continue
            out.append(path)
        return out

    def _read_text_lines(self, path: Path) -> List[str]:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return []

    def _read_pdf_pages(self, path: Path) -> List[str]:
        try:
            from pypdf import PdfReader  # local import: optional dep, heavy path
            reader = PdfReader(str(path))
        except Exception:
            return []
        pages: List[str] = []
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception:
                pages.append("")
        return pages

    def _txt_chunk(self, path: Path, chunk_index: int, chunk_lines: int) -> Optional[_Chunk]:
        lines = self._read_text_lines(path)
        if not lines:
            return None
        start = chunk_index * chunk_lines
        if start >= len(lines):
            return None
        end = min(len(lines), start + chunk_lines)
        picked = lines[start:end]
        text = "\n".join(picked).strip()
        if not text:
            return None
        return _Chunk(
            kind="text",
            text=text,
            chunk_index=chunk_index,
            start_line=start + 1,
            end_line=end,
            summary=f"{path.name} lines {start + 1}-{end}",
        )

    def _pdf_chunk(self, path: Path, chunk_index: int, chunk_chars: int) -> Optional[_Chunk]:
        pages = self._read_pdf_pages(path)
        flat: List[Tuple[int, str]] = []
        for idx, page_text in enumerate(pages):
            if page_text.strip():
                flat.append((idx + 1, page_text.strip()))
        if not flat or chunk_index >= len(flat):
            return None
        page_no, page_text = flat[chunk_index]
        clipped = page_text[: max(400, chunk_chars)].strip()
        if not clipped:
            return None
        return _Chunk(
            kind="pdf",
            text=clipped,
            chunk_index=chunk_index,
            page=page_no,
            summary=f"{path.name} page {page_no}",
        )

    def _chunk_for(self, path: Path, chunk_index: int, chunk_lines: int, chunk_chars: int) -> Optional[_Chunk]:
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            return self._txt_chunk(path, chunk_index, chunk_lines)
        if suffix == ".pdf":
            return self._pdf_chunk(path, chunk_index, chunk_chars)
        return None

    def _mem_cell_store(self) -> Optional[MemCellStore]:
        store = self.orch.kv_store.get("memory:mem_cell_store")
        if isinstance(store, MemCellStore):
            return store
        try:
            store = MemCellStore(str(self.memdir))
            self.orch.kv_store["memory:mem_cell_store"] = store
            return store
        except Exception:
            logger.exception("Read sidecar could not initialize MemCellStore")
            return None

    def _ingest_piece(self, mem_cell_store: MemCellStore, text: str, source_name: str) -> int:
        pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not pieces:
            pieces = [text.strip()]
        count = 0
        for piece in pieces[:4]:
            if len(piece) > 900:
                piece = piece[:900].rsplit(" ", 1)[0].strip() or piece[:900]
            result = mem_cell_store.ingest_text(
                text=piece,
                topic="reading/text",
                role="assistant",
                transport_source="reading",
                source=source_name,
                meta={"channel": "reading"},
                tier="now",
            )
            rows = [result.get("utterance")] + list(result.get("tokens", [])) + list(result.get("patterns", []))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row["activation"] = min(float(row.get("activation", 0.18) or 0.18), 0.18)
                row["promotion"] = min(float(row.get("promotion", 0.0) or 0.0), 0.0)
                row["trust"] = min(float(row.get("trust", 0.3) or 0.3), 0.30)
                mem_cell_store.upsert_cell(row, tier=str(row.get("tier", "now") or "now"))
            count += 1
        return count
