from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.syntax_learning_neuron import SyntaxLearningNeuron, strip_slearn_inline_comment
from microbrain.orchestrator.event_bus import Event
from microbrain.orchestrator.neuron_base import NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sidecars.slearn_workbench import (
    SLEARN_BUCKET_BATCH_LINES,
    SLEARN_BUCKET_MIN_BYTES,
    SLEARN_BUCKET_MIN_LINES,
    SLEARN_COMPOSER_FLUSH_BATCHES,
    SLEARN_MAX_INFLIGHT_BATCHES,
    SLEARN_NORMAL_BATCH_LINES,
    SlearnWorkspaceCleaner,
    byte_offset_for_line,
    read_line_batch,
    scan_slearn_file,
    stable_job_id,
)

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
        self._slearn_store: MemCellStore | None = None
        self._slearn_store_job_id = ""
        self._slearn_seen_rule_hashes: set[str] = set()
        self._slearn_seen_job_id = ""
        self._slearn_processor = SyntaxLearningNeuron(
            NeuronConfig(
                name="slearn_sidecar_processor",
                subscribed_topics=[],
                output_topics=[],
                priority=0,
            )
        )
        self._slearn_cleaner = SlearnWorkspaceCleaner(self.memdir)

    async def start(self) -> None:
        if self._loop_task is not None:
            return

        async def _on_control(ev: Event) -> List[Event]:
            payload = ev.payload if isinstance(ev.payload, dict) else {}
            cmd = str(payload.get("command", "") or "").strip().lower()
            if cmd == "on":
                self.orch.kv_store["read:enabled"] = True
                # Kick one immediate read pass so /read on feels alive instead of
                # waiting for the idle timer. The cadence gate still controls
                # later background chewing.
                self.orch.kv_store["read:force_once"] = time.time()
            elif cmd == "off":
                self.orch.kv_store["read:enabled"] = False
                self.orch.kv_store["read:force_once"] = 0.0
            elif cmd in ("next", "step"):
                self.orch.kv_store["read:force_once"] = time.time()
            self._wake_event.set()
            return []

        async def _on_slearn_control(ev: Event) -> List[Event]:
            payload = ev.payload if isinstance(ev.payload, dict) else {}
            cmd = str(payload.get("command", "") or "").strip().lower()
            if cmd == "on":
                self.orch.kv_store["slearn:enabled"] = True
                # Preflight immediately; the heavy work itself remains in the
                # sidecar worker and never blocks the dashboard event loop.
                self.orch.kv_store["slearn:force_once"] = time.time()
            elif cmd == "off":
                self.orch.kv_store["slearn:enabled"] = False
            elif cmd in ("next", "step"):
                self.orch.kv_store["slearn:force_once"] = time.time()
            self._wake_event.set()
            return []

        self._sub_id = self.orch.bus.subscribe(
            "sidecar.read.control",
            ["control/read"],
            _on_control,
            priority=0,
        )
        self.orch.bus.subscribe(
            "sidecar.slearn.control",
            ["control/slearn"],
            _on_slearn_control,
            priority=0,
        )

        # SLEARN behavioral tuning is centralized in KV so Window 2 can inspect
        # and adjust the workbench without editing source.
        self.orch.kv_store.setdefault("slearn:bucket_min_bytes", SLEARN_BUCKET_MIN_BYTES)
        self.orch.kv_store.setdefault("slearn:bucket_min_lines", SLEARN_BUCKET_MIN_LINES)
        self.orch.kv_store.setdefault("slearn:normal_batch_lines", SLEARN_NORMAL_BATCH_LINES)
        self.orch.kv_store.setdefault("slearn:bucket_batch_lines", SLEARN_BUCKET_BATCH_LINES)
        self.orch.kv_store.setdefault("slearn:max_inflight_batches", SLEARN_MAX_INFLIGHT_BATCHES)
        self.orch.kv_store.setdefault("slearn:composer_flush_batches", SLEARN_COMPOSER_FLUSH_BATCHES)

        # Seed the visible intake paths immediately so /read status can show
        # real folders before the first background cycle runs.
        read_dir = self._resolve_read_dir()
        read_ready_dir = self._resolve_read_ready_dir(read_dir)
        slearn_dir = self._resolve_slearn_dir()
        for path in (read_dir, read_ready_dir, self._legacy_read_dir(), slearn_dir, slearn_dir / "ready"):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Read sidecar failed to create intake folder: %s", path)

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
        slearn_did_work = await self._run_slearn_once()
        if slearn_did_work:
            return

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
        ready_dir = self._resolve_read_ready_dir(read_dir)
        read_dir.mkdir(parents=True, exist_ok=True)
        ready_dir.mkdir(parents=True, exist_ok=True)

        state = await asyncio.to_thread(self._load_state_file, read_dir)
        active_file = str(self.orch.kv_store.get("read:active_file", "") or state.get("active_file", "") or "")
        chunk_index = int(self.orch.kv_store.get("read:chunk_index", state.get("chunk_index", 0)) or 0)

        path = Path(active_file) if active_file else None
        if path is None or not path.exists():
            candidates = await asyncio.to_thread(self._list_candidates, read_dir)
            if not candidates:
                result = {
                    "ts": now,
                    "summary": "no readable files in read queue",
                    "read_dir": str(read_dir),
                    "ready_dir": str(ready_dir),
                    "legacy_dir": str(self._legacy_read_dir()),
                }
                self._apply_status(result=result, active_file="", active_kind="", chunk_index=0, now=now)
                await asyncio.to_thread(
                    self._save_state_file,
                    read_dir,
                    {"active_file": "", "chunk_index": 0, "last_result": result},
                )
                await self._publish_read_status(result, text=f"/read: no readable files in {read_dir}")
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
            await self._publish_read_status(result, text=f"/read: completed {path.name}; moved to {target}")
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

        await self._publish_read_status(
            result,
            text=(
                f"/read: {path.name} chunk {chunk.chunk_index} ingested "
                f"{ingested} piece(s); next_chunk={next_index}"
            ),
        )

        if force:
            self.orch.kv_store["read:force_once"] = 0.0


    async def _run_slearn_once(self) -> bool:
        """Run one structured-learning workbench step.

        SLEARN is an organ, not a cognition flood.  Files are preflighted first,
        then consumed through a resumable streaming cursor.  Rule parsing/memory
        staging happens in a worker thread and only compact job-status events are
        returned to the main event bus.
        """
        enabled = bool(self.orch.kv_store.get("slearn:enabled", False))
        force_ts = float(self.orch.kv_store.get("slearn:force_once", 0.0) or 0.0)
        force = force_ts > 0.0
        if not enabled and not force:
            return False

        now = time.time()
        slearn_dir = self._resolve_slearn_dir()
        ready_dir = slearn_dir / "ready"
        slearn_dir.mkdir(parents=True, exist_ok=True)
        ready_dir.mkdir(parents=True, exist_ok=True)

        state = await asyncio.to_thread(self._load_slearn_state_file, slearn_dir)
        # The persisted job file is the crash-recovery source of truth. Runtime KV
        # can lag behind it across abrupt UI/process shutdowns, so never let a stale
        # in-memory active_file graft one job's cursor/receipts onto another file.
        persisted_active_file = str(state.get("active_file", "") or "")
        runtime_active_file = str(self.orch.kv_store.get("slearn:active_file", "") or "")
        active_file = persisted_active_file or runtime_active_file
        path = Path(active_file) if active_file else None

        # Pick a new job only when there is no valid active file.  No SLEARN work
        # means normal /read is free to use the same sidecar loop.
        if path is None or not path.exists():
            candidates = await asyncio.to_thread(self._list_slearn_candidates, slearn_dir)
            if not candidates:
                result = {"ts": now, "summary": "no structured learning files in slearn_dir", "status": "idle"}
                self._apply_slearn_status(result=result, active_file="", chunk_index=0, now=now)
                self.orch.kv_store["slearn:status"] = "idle"
                if force:
                    self.orch.kv_store["slearn:force_once"] = 0.0
                return False
            path = candidates[0]
            state = await self._begin_slearn_job(path, slearn_dir, now=now)
            if force:
                self.orch.kv_store["slearn:force_once"] = 0.0
            return True

        # Migrate a legacy chunk-index state into the streaming cursor once.
        if str(state.get("schema", "") or "") != "slearn.job.v2":
            state = await self._migrate_legacy_slearn_state(path, state, slearn_dir, now=now)

        # A v2 cursor belongs to one exact source revision. If the file changed or
        # runtime state pointed at a different file after a crash, start a clean
        # attempt instead of inheriting another job's cursor/receipt obligations.
        expected_job_id = stable_job_id(path)
        state_job_id = str(state.get("job_id", "") or "")
        if state_job_id and state_job_id != expected_job_id:
            state = await self._begin_slearn_job(path, slearn_dir, now=now)
            if force:
                self.orch.kv_store["slearn:force_once"] = 0.0
            return True
        job_id = state_job_id or expected_job_id
        state["job_id"] = job_id

        # A previously blocked job can become runnable once the composer returns.
        if str(state.get("phase", "") or "") == "blocked":
            store = self._slearn_store_for_job(job_id)
            if store.composer_enabled and not bool(self.orch.kv_store.get("mem_cell:composer:started", False)):
                self.orch.kv_store["slearn:status"] = "blocked"
                return True
            workspace = await asyncio.to_thread(
                self._slearn_cleaner.prepare,
                job_id=job_id,
                source_path=path,
            )
            state["workspace"] = workspace
            state["phase"] = "ingesting"
            state["status"] = "ingesting"
            state.pop("blocked_reason", None)
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            await self._publish_slearn_diagnostic(
                "slearn/preflight",
                {**dict(state.get("preflight", {}) or {}), "status": "preflight", "workspace": workspace, "summary": "SLEARN preflight passed; workspace prepared"},
                kind="slearn_preflight",
            )
            self._wake_event.set()
            return True

        mode = str(state.get("mode", "normal") or "normal").lower()
        idle_after_s = float(self.orch.kv_store.get("slearn:idle_after_s", 2.0) or 2.0)
        interaction_last = self.orch.kv_store.get("interaction:last_input", {}) or {}
        last_input_ts = float(interaction_last.get("ts", 0.0) or 0.0) if isinstance(interaction_last, dict) else 0.0
        if not force and (now - last_input_ts) < idle_after_s:
            return True

        # Normal curriculum remains intentionally gentle.  Bucket mode is already
        # isolated in a worker and can advance continuously while the user is idle.
        if mode != "bucket" and not force:
            tick_every_s = float(self.orch.kv_store.get("slearn:tick_every_s", 2.0) or 2.0)
            last_activity_ts = float(self.orch.kv_store.get("slearn:last_activity_ts", 0.0) or 0.0)
            if (now - last_activity_ts) < tick_every_s:
                return True

        outstanding, detached = self._reconcile_slearn_receipts(
            list(state.get("receipts", []) or []),
            job_id=job_id,
            started_at=float(state.get("started_at", 0.0) or 0.0),
        )
        state["receipts"] = outstanding
        if detached:
            recovery = dict(state.get("receipt_recovery", {}) or {})
            recovery["detached_total"] = int(recovery.get("detached_total", 0) or 0) + len(detached)
            recovery["last_detached"] = len(detached)
            recovery["last_ts"] = now
            reasons: Dict[str, int] = {}
            for item in detached:
                reason = str(item.get("reason", "foreign") or "foreign")
                reasons[reason] = reasons.get(reason, 0) + 1
            recovery["last_reasons"] = reasons
            recovery["last_receipts"] = [str(item.get("receipt", "") or "") for item in detached[-8:]]
            state["receipt_recovery"] = recovery
            workspace = dict(state.get("workspace", {}) or {})
            actions = list(workspace.get("actions", []) or [])
            actions.append(f"detached_foreign_receipts:{len(detached)}")
            workspace["actions"] = actions[-32:]
            state["workspace"] = workspace
            state["updated_at"] = now
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            recovery_payload = self._slearn_status_payload(
                state,
                summary=f"recovered SLEARN receipt ownership; detached {len(detached)} stale/foreign receipt(s)",
            )
            recovery_payload["detached_receipts"] = detached[-8:]
            await asyncio.to_thread(self._append_slearn_audit, {"event": "receipt_recovery", **recovery_payload})
            await self._publish_slearn_diagnostic("slearn/recovery", recovery_payload, kind="slearn_receipt_recovery")

        max_inflight = int(self.orch.kv_store.get("slearn:max_inflight_batches", SLEARN_MAX_INFLIGHT_BATCHES) or SLEARN_MAX_INFLIGHT_BATCHES)
        if len(outstanding) >= max(1, max_inflight):
            state["phase"] = "waiting_composer"
            state["status"] = "waiting_composer"
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            await self._publish_slearn_diagnostic(
                "slearn/status",
                self._slearn_status_payload(state, summary=f"waiting for memory composer ({len(outstanding)} batch receipt(s) outstanding)"),
                kind="slearn_waiting_composer",
            )
            return True

        if str(state.get("phase", "") or "") in {"waiting_commit", "waiting_composer"} and bool(state.get("eof", False)):
            finalized = await self._finalize_slearn_job(path, ready_dir, slearn_dir, state, now=now)
            if not finalized:
                return True
            if force:
                self.orch.kv_store["slearn:force_once"] = 0.0
            return True

        preflight = dict(state.get("preflight", {}) or {})
        batch_lines_default = SLEARN_BUCKET_BATCH_LINES if mode == "bucket" else SLEARN_NORMAL_BATCH_LINES
        batch_key = "slearn:bucket_batch_lines" if mode == "bucket" else "slearn:normal_batch_lines"
        batch_lines = int(self.orch.kv_store.get(batch_key, batch_lines_default) or batch_lines_default)
        batch = await asyncio.to_thread(
            read_line_batch,
            path,
            byte_offset=int(state.get("byte_offset", 0) or 0),
            line_number=int(state.get("line_number", 0) or 0),
            max_lines=max(1, batch_lines),
        )

        if not batch.lines and batch.eof:
            state["eof"] = True
            state["phase"] = "waiting_commit"
            state["status"] = "waiting_commit"
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            await self._finalize_slearn_job(path, ready_dir, slearn_dir, state, now=now)
            return True

        if self._slearn_seen_job_id != job_id:
            self._slearn_seen_rule_hashes.clear()
            self._slearn_seen_job_id = job_id

        rule_items: List[Dict[str, Any]] = []
        pre_rejected = 0
        ignored = 0
        duplicates = 0
        for line_no, raw_line in batch.lines:
            line = str(raw_line or "").strip()
            if not line or line.startswith("#") or line.startswith("//"):
                ignored += 1
                continue
            extracted = self._extract_slearn_rules(line)
            if not extracted:
                if line.startswith("IF ") or " THEN " in line:
                    pre_rejected += 1
                else:
                    ignored += 1
                continue
            rule = extracted[0][1]
            rule_hash = hashlib.blake2b(rule.encode("utf-8", errors="ignore"), digest_size=12).hexdigest()
            if rule_hash in self._slearn_seen_rule_hashes:
                duplicates += 1
                continue
            self._slearn_seen_rule_hashes.add(rule_hash)
            rule_items.append({
                "teaching_note": rule,
                "source_name": path.name,
                "source_path": str(path),
                "source_line": line_no,
                "ingest_mode": f"structured_learning_{mode}",
            })

        weight = int(self.orch.kv_store.get("slearn:default_weight", 3) or 3)
        weight = max(1, min(5, weight))
        store = self._slearn_store_for_job(job_id)
        try:
            applied = await asyncio.to_thread(
                self._slearn_processor.apply_slearn_batch,
                store,
                rule_items,
                weight=weight,
            )
        except Exception as exc:
            await self._fail_slearn_job(path, slearn_dir, state, exc, now=now)
            return True

        accepted_now = int(applied.get("accepted", 0) or 0)
        rejected_now = pre_rejected + int(applied.get("rejected", 0) or 0)
        saved_cells_now = int(applied.get("saved_cells", 0) or 0)
        receipts = outstanding + [str(p) for p in list(applied.get("staged_paths", []) or []) if str(p)]

        state["byte_offset"] = int(batch.byte_offset)
        state["line_number"] = int(batch.end_line)
        state["batch_index"] = int(state.get("batch_index", 0) or 0) + 1
        state["chunk_index"] = state["batch_index"]
        state["accepted"] = int(state.get("accepted", 0) or 0) + accepted_now
        state["rejected"] = int(state.get("rejected", 0) or 0) + rejected_now
        state["duplicates"] = int(state.get("duplicates", 0) or 0) + duplicates
        state["ignored"] = int(state.get("ignored", 0) or 0) + ignored
        state["saved_cells"] = int(state.get("saved_cells", 0) or 0) + saved_cells_now
        state["receipts"] = receipts
        state["eof"] = bool(batch.eof)
        state["phase"] = "waiting_commit" if batch.eof else "ingesting"
        state["status"] = state["phase"]
        state["updated_at"] = time.time()

        total_size = max(1, int(preflight.get("file_size_bytes", 0) or 0))
        progress_pct = min(100.0, (float(state["byte_offset"]) / float(total_size)) * 100.0)
        state["progress_pct"] = progress_pct

        # Compatibility counters remain visible, but no rule events are emitted to
        # cognition.  "emitted" now means staged by the SLEARN organ.
        self.orch.kv_store["slearn:rules_emitted_total"] = int(self.orch.kv_store.get("slearn:rules_emitted_total", 0) or 0) + accepted_now
        self.orch.kv_store["slearn:rules_staged_total"] = int(self.orch.kv_store.get("slearn:rules_staged_total", 0) or 0) + accepted_now
        self.orch.kv_store["slearn:saved_cells_total"] = int(self.orch.kv_store.get("slearn:saved_cells_total", 0) or 0) + saved_cells_now
        await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
        self._sync_slearn_runtime_state(state, now=now)

        progress_payload = self._slearn_status_payload(
            state,
            summary=(
                f"{path.name} {mode} batch {state['batch_index']}: "
                f"accepted {accepted_now}, rejected {rejected_now}, duplicates {duplicates}"
            ),
        )
        progress_payload.update({
            "progress_pct": progress_pct,
            "processed": int(state["line_number"]),
            "total": int(preflight.get("line_count", 0) or 0),
            "total_lines": int(preflight.get("line_count", 0) or 0),
            "batch_accepted": accepted_now,
            "batch_rejected": rejected_now,
            "batch_duplicates": duplicates,
            "batch_ignored": ignored,
            "batch_saved_cells": saved_cells_now,
            "outstanding_batches": len(receipts),
        })
        await asyncio.to_thread(self._append_slearn_audit, {"event": "batch_staged", **progress_payload})
        await self._publish_slearn_diagnostic("slearn/progress", progress_payload, kind="slearn_progress")

        if batch.eof:
            await self._finalize_slearn_job(path, ready_dir, slearn_dir, state, now=time.time())
        elif mode == "bucket":
            # Keep the bulk organ moving without waiting for the 1 s sidecar poll.
            # Heavy work still happens in to_thread(), so Qt/cognition keep breathing.
            self._wake_event.set()

        if force:
            self.orch.kv_store["slearn:force_once"] = 0.0
        return True

    async def _begin_slearn_job(self, path: Path, slearn_dir: Path, *, now: float) -> Dict[str, Any]:
        bucket_min_bytes = int(self.orch.kv_store.get("slearn:bucket_min_bytes", SLEARN_BUCKET_MIN_BYTES) or SLEARN_BUCKET_MIN_BYTES)
        bucket_min_lines = int(self.orch.kv_store.get("slearn:bucket_min_lines", SLEARN_BUCKET_MIN_LINES) or SLEARN_BUCKET_MIN_LINES)
        preflight_obj = await asyncio.to_thread(
            scan_slearn_file,
            path,
            bucket_min_bytes=max(1, bucket_min_bytes),
            bucket_min_lines=max(1, bucket_min_lines),
        )
        preflight = preflight_obj.to_dict()
        job_id = stable_job_id(path)
        store = self._slearn_store_for_job(job_id)

        state: Dict[str, Any] = {
            "schema": "slearn.job.v2",
            "active_file": str(path),
            "job_id": job_id,
            "mode": preflight_obj.mode,
            "phase": "preflight",
            "status": "preflight",
            "started_at": now,
            "updated_at": now,
            "byte_offset": 0,
            "line_number": 0,
            "batch_index": 0,
            "chunk_index": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
            "ignored": 0,
            "saved_cells": 0,
            "receipts": [],
            "eof": False,
            "preflight": preflight,
            "workspace": {},
        }

        if store.composer_enabled and not bool(self.orch.kv_store.get("mem_cell:composer:started", False)):
            state["phase"] = "blocked"
            state["status"] = "blocked"
            state["blocked_reason"] = "memory_composer_not_running"
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            payload = self._slearn_status_payload(state, summary="SLEARN blocked: memory composer is not running")
            await self._publish_slearn_diagnostic("slearn/blocked", payload, kind="slearn_blocked")
            await self._publish_learning_result("learning/blocked", payload)
            return state

        workspace = await asyncio.to_thread(self._slearn_cleaner.prepare, job_id=job_id, source_path=path)
        state["workspace"] = workspace
        state["phase"] = "ingesting"
        state["status"] = "ingesting"
        await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
        self._sync_slearn_runtime_state(state, now=now)
        payload = self._slearn_status_payload(
            state,
            summary=(
                f"preflight {path.name}: {preflight_obj.line_count} lines, "
                f"{preflight_obj.file_size_bytes} bytes -> {preflight_obj.mode.upper()} ({preflight_obj.reason})"
            ),
        )
        await asyncio.to_thread(self._append_slearn_audit, {"event": "preflight", **payload})
        await self._publish_slearn_diagnostic("slearn/preflight", payload, kind="slearn_preflight")
        self._wake_event.set()
        return state

    async def _migrate_legacy_slearn_state(self, path: Path, old_state: Dict[str, Any], slearn_dir: Path, *, now: float) -> Dict[str, Any]:
        preflight_obj = await asyncio.to_thread(
            scan_slearn_file,
            path,
            bucket_min_bytes=int(self.orch.kv_store.get("slearn:bucket_min_bytes", SLEARN_BUCKET_MIN_BYTES) or SLEARN_BUCKET_MIN_BYTES),
            bucket_min_lines=int(self.orch.kv_store.get("slearn:bucket_min_lines", SLEARN_BUCKET_MIN_LINES) or SLEARN_BUCKET_MIN_LINES),
        )
        legacy_chunk_lines = int(self.orch.kv_store.get("slearn:chunk_lines", 80) or 80)
        legacy_chunk_index = int(old_state.get("chunk_index", self.orch.kv_store.get("slearn:chunk_index", 0)) or 0)
        line_number = max(0, legacy_chunk_index * max(1, legacy_chunk_lines))
        byte_offset = await asyncio.to_thread(byte_offset_for_line, path, line_number)
        job_id = stable_job_id(path)
        workspace = await asyncio.to_thread(self._slearn_cleaner.prepare, job_id=job_id, source_path=path)
        state = {
            "schema": "slearn.job.v2",
            "active_file": str(path),
            "job_id": job_id,
            "mode": preflight_obj.mode,
            "phase": "ingesting",
            "status": "ingesting",
            "started_at": float(old_state.get("started_at", now) or now),
            "updated_at": now,
            "byte_offset": byte_offset,
            "line_number": line_number,
            "batch_index": legacy_chunk_index,
            "chunk_index": legacy_chunk_index,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
            "ignored": 0,
            "saved_cells": 0,
            "receipts": [],
            "eof": False,
            "preflight": preflight_obj.to_dict(),
            "workspace": workspace,
            "migrated_from_chunk_state": True,
        }
        await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
        self._sync_slearn_runtime_state(state, now=now)
        await self._publish_slearn_diagnostic(
            "slearn/preflight",
            self._slearn_status_payload(state, summary=f"migrated legacy SLEARN state at line {line_number}; continuing in {preflight_obj.mode.upper()} mode"),
            kind="slearn_preflight",
        )
        return state

    async def _finalize_slearn_job(
        self,
        path: Path,
        ready_dir: Path,
        slearn_dir: Path,
        state: Dict[str, Any],
        *,
        now: float,
    ) -> bool:
        job_id = str(state.get("job_id", "") or stable_job_id(path))
        store = self._slearn_store_for_job(job_id)

        # Sweep any deferred rows before declaring the memory desk clean.
        if store.dirty_count("learned"):
            await asyncio.to_thread(store.flush_tier, "learned")
            state["receipts"] = list(state.get("receipts", []) or []) + store.take_staged_paths("learned")

        outstanding = self._prune_slearn_receipts(
            list(state.get("receipts", []) or []),
            job_id=job_id,
            started_at=float(state.get("started_at", 0.0) or 0.0),
        )
        state["receipts"] = outstanding
        if outstanding:
            state["phase"] = "waiting_commit"
            state["status"] = "waiting_commit"
            state["eof"] = True
            await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
            self._sync_slearn_runtime_state(state, now=now)
            await self._publish_slearn_diagnostic(
                "slearn/status",
                self._slearn_status_payload(state, summary=f"ingest finished; waiting for {len(outstanding)} memory composer commit(s)"),
                kind="slearn_waiting_commit",
            )
            return False

        workspace = dict(state.get("workspace", {}) or {})
        baseline = workspace.get("baseline") if isinstance(workspace.get("baseline"), dict) else {}
        cleanup = await asyncio.to_thread(self._slearn_cleaner.finish, job_id=job_id, baseline=baseline)
        state["workspace"] = cleanup

        target = ready_dir / path.name
        if target.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            target = ready_dir / f"{path.stem}-{stamp}{path.suffix}"
        try:
            await asyncio.to_thread(shutil.move, str(path), str(target))
        except Exception as exc:
            await self._fail_slearn_job(path, slearn_dir, state, exc, now=now)
            return False

        accepted = int(state.get("accepted", 0) or 0)
        # Applied means composer-confirmed for this file, not merely queued.
        self.orch.kv_store["slearn:rules_applied_total"] = int(self.orch.kv_store.get("slearn:rules_applied_total", 0) or 0) + accepted
        duration_s = max(0.0, time.time() - float(state.get("started_at", now) or now))
        result = self._slearn_status_payload(
            state,
            summary=f"{path.name} completed; durable commits confirmed and workspace restored",
        )
        result.update({
            "status": "completed",
            "completed_file": path.name,
            "ready_path": str(target),
            "completed_batches": int(state.get("batch_index", 0) or 0),
            "completed_chunks": int(state.get("batch_index", 0) or 0),
            "duration_s": duration_s,
            "workspace": cleanup,
            "workspace_clean": bool(cleanup.get("clean", False)),
            "baseline_restored": bool(cleanup.get("baseline_restored", False)),
            "audit_path": str(self._slearn_audit_path()),
        })
        self._apply_slearn_status(result=result, active_file="", chunk_index=0, now=time.time())
        result["files_completed_count"] = int(self.orch.kv_store.get("slearn:files_completed_count", 0) or 0)
        result["rules_emitted_total"] = int(self.orch.kv_store.get("slearn:rules_emitted_total", 0) or 0)
        result["rules_applied_total"] = int(self.orch.kv_store.get("slearn:rules_applied_total", 0) or 0)
        result["rules_staged_total"] = int(self.orch.kv_store.get("slearn:rules_staged_total", 0) or 0)
        self.orch.kv_store["slearn:status"] = "completed"
        self.orch.kv_store["slearn:mode"] = str(state.get("mode", "") or "")
        self.orch.kv_store["slearn:workspace"] = cleanup
        await asyncio.to_thread(self._append_slearn_audit, {"event": "file_completed", **result})
        await asyncio.to_thread(
            self._save_slearn_state_file,
            slearn_dir,
            {
                "schema": "slearn.job.v2",
                "active_file": "",
                "chunk_index": 0,
                "last_result": result,
                "completed_files": list(self.orch.kv_store.get("slearn:completed_files", []) or []),
                "files_completed_count": result["files_completed_count"],
                "rules_emitted_total": result["rules_emitted_total"],
                "rules_applied_total": result["rules_applied_total"],
                "audit_path": result["audit_path"],
            },
        )
        await self._publish_slearn_diagnostic("slearn/completed", result, kind="slearn_completed")
        await self._publish_learning_result("learning/completed", result)
        await self.orch.push_event(
            "ui/status",
            {"text": f"/slearn: completed {path.name}; {accepted} rule(s) committed; workspace clean={bool(cleanup.get('clean', False))}", "style": "system", "channel": "default"},
            source="read_sidecar",
            meta={"control": True, "kind": "slearn_status", "store_in_memory": False, "reinforcement_eligible": False, "self_output_track": False, "cognitive_visible": False},
        )

        # Post-job memory hygiene: release the bulk store/index and duplicate set
        # so a 50k-line lexical file does not linger in the process after commit.
        self._slearn_store = None
        self._slearn_store_job_id = ""
        self._slearn_seen_rule_hashes.clear()
        self._slearn_seen_job_id = ""
        return True

    async def _fail_slearn_job(self, path: Path, slearn_dir: Path, state: Dict[str, Any], exc: BaseException, *, now: float) -> None:
        state["phase"] = "failed"
        state["status"] = "failed"
        state["error"] = repr(exc)
        state["updated_at"] = now
        await asyncio.to_thread(self._save_slearn_state_file, slearn_dir, state)
        self._sync_slearn_runtime_state(state, now=now)
        payload = self._slearn_status_payload(state, summary=f"SLEARN failed at {path.name}: {exc!r}")
        payload["recoverable"] = True
        await asyncio.to_thread(self._append_slearn_audit, {"event": "failed", **payload})
        await self._publish_slearn_diagnostic("slearn/failed", payload, kind="slearn_failed")
        await self._publish_learning_result("learning/failed", payload)

    def _slearn_store_for_job(self, job_id: str) -> MemCellStore:
        resolved = str(job_id or "slearn-job")
        if self._slearn_store is None or self._slearn_store_job_id != resolved:
            self._slearn_store = MemCellStore(str(self.memdir), writer_id=f"slearn-sidecar-{resolved}")
            self._slearn_store_job_id = resolved
        return self._slearn_store

    def _receipt_pending_paths(self, receipt: str) -> List[Path]:
        """Return live pending/processing paths for one composer receipt."""
        path = Path(str(receipt or ""))
        if not str(path):
            return []
        out: List[Path] = []
        try:
            if path.exists():
                out.append(path)
        except OSError:
            # Unknown filesystem state is treated as pending by the caller.
            return [path]

        processing_dir = self.memdir / "mem_cell" / "_processing" / "learned"
        try:
            out.extend(processing_dir.glob(f"{path.name}.*.processing"))
        except OSError:
            if not out:
                return [path]
        return out

    def _receipt_is_pending(self, receipt: str) -> bool:
        return bool(self._receipt_pending_paths(receipt))

    def _receipt_metadata(self, receipt: str) -> Dict[str, Any]:
        """Read the first composer envelope without consuming the receipt."""
        for candidate in self._receipt_pending_paths(receipt):
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        envelope = json.loads(raw)
                        if not isinstance(envelope, dict):
                            return {}
                        return {
                            "writer_id": str(envelope.get("writer_id", "") or ""),
                            "queued_at": float(envelope.get("queued_at", 0.0) or 0.0),
                            "op_id": str(envelope.get("op_id", "") or ""),
                        }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return {}

    def _reconcile_slearn_receipts(
        self,
        receipts: List[str],
        *,
        job_id: str = "",
        started_at: float = 0.0,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Keep only pending receipts owned by the active SLEARN attempt.

        Composer files are intentionally *not* deleted here. A detached receipt may
        still contain valid work from an interrupted prior job, and the memory
        composer remains free to commit it. This method only prevents the current
        SLEARN file from waiting on another job/attempt's receipt.
        """
        out: List[str] = []
        detached: List[Dict[str, Any]] = []
        seen: set[str] = set()
        expected_writer = f"slearn-sidecar-{job_id}" if job_id else ""
        attempt_floor = max(0.0, float(started_at or 0.0))

        for raw in receipts or []:
            receipt = str(raw or "").strip()
            if not receipt or receipt in seen:
                continue
            seen.add(receipt)
            if not self._receipt_is_pending(receipt):
                continue

            meta = self._receipt_metadata(receipt)
            writer_id = str(meta.get("writer_id", "") or "")
            queued_at = float(meta.get("queued_at", 0.0) or 0.0)

            reason = ""
            if expected_writer and writer_id and writer_id != expected_writer:
                reason = "foreign_job"
            elif attempt_floor > 0.0 and queued_at > 0.0 and queued_at < (attempt_floor - 0.001):
                # stable_job_id intentionally survives a process restart for the
                # same source file, so writer_id alone cannot distinguish a fresh
                # attempt from old receipts. queued_at vs persisted started_at can.
                reason = "older_attempt"

            if reason:
                detached.append({
                    "receipt": receipt,
                    "reason": reason,
                    "writer_id": writer_id,
                    "queued_at": queued_at,
                })
                continue
            out.append(receipt)
        return out, detached

    def _prune_slearn_receipts(
        self,
        receipts: List[str],
        *,
        job_id: str = "",
        started_at: float = 0.0,
    ) -> List[str]:
        out, _ = self._reconcile_slearn_receipts(receipts, job_id=job_id, started_at=started_at)
        return out

    def _sync_slearn_runtime_state(self, state: Dict[str, Any], *, now: float) -> None:
        self.orch.kv_store["slearn:active_file"] = str(state.get("active_file", "") or "")
        self.orch.kv_store["slearn:chunk_index"] = int(state.get("batch_index", state.get("chunk_index", 0)) or 0)
        self.orch.kv_store["slearn:last_activity_ts"] = now
        self.orch.kv_store["slearn:mode"] = str(state.get("mode", "") or "")
        self.orch.kv_store["slearn:status"] = str(state.get("status", state.get("phase", "")) or "")
        self.orch.kv_store["slearn:preflight"] = dict(state.get("preflight", {}) or {})
        self.orch.kv_store["slearn:workspace"] = dict(state.get("workspace", {}) or {})
        receipt_paths = [str(p) for p in list(state.get("receipts", []) or []) if str(p or "").strip()]
        self.orch.kv_store["slearn:receipt_paths"] = receipt_paths
        self.orch.kv_store["slearn:outstanding_batches"] = len(receipt_paths)
        self.orch.kv_store["slearn:eof"] = bool(state.get("eof", False))
        self.orch.kv_store["slearn:phase"] = str(state.get("phase", "") or "")
        self.orch.kv_store["slearn:last_result"] = self._slearn_status_payload(state)

    def _slearn_status_payload(self, state: Dict[str, Any], *, summary: str = "") -> Dict[str, Any]:
        preflight = dict(state.get("preflight", {}) or {})
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "job_id": str(state.get("job_id", "") or ""),
            "file": str(state.get("active_file", "") or ""),
            "active_file": str(state.get("active_file", "") or ""),
            "mode": str(state.get("mode", "") or ""),
            "status": str(state.get("status", state.get("phase", "")) or ""),
            "phase": str(state.get("phase", "") or ""),
            "batch_index": int(state.get("batch_index", 0) or 0),
            "chunk_index": int(state.get("batch_index", state.get("chunk_index", 0)) or 0),
            "processed": int(state.get("line_number", 0) or 0),
            "total": int(preflight.get("line_count", 0) or 0),
            "total_lines": int(preflight.get("line_count", 0) or 0),
            "progress_pct": float(state.get("progress_pct", 0.0) or 0.0),
            "accepted": int(state.get("accepted", 0) or 0),
            "duplicates": int(state.get("duplicates", 0) or 0),
            "rejected": int(state.get("rejected", 0) or 0),
            "ignored": int(state.get("ignored", 0) or 0),
            "saved_cells": int(state.get("saved_cells", 0) or 0),
            "outstanding_batches": len(list(state.get("receipts", []) or [])),
            "receipt_recovery": dict(state.get("receipt_recovery", {}) or {}),
            "preflight": preflight,
            "workspace": dict(state.get("workspace", {}) or {}),
            "warnings": list(state.get("warnings", []) or []),
            "summary": summary or str(state.get("summary", "") or ""),
        }
        return payload

    async def _publish_slearn_diagnostic(self, topic: str, payload: Dict[str, Any], *, kind: str) -> None:
        await self.orch.push_event(
            topic,
            dict(payload),
            source="read_sidecar",
            meta={
                "source": "read_sidecar",
                "channel": "internal",
                "control": True,
                "kind": kind,
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "self_output_track": False,
                "cognitive_visible": False,
            },
        )

    async def _publish_learning_result(self, topic: str, payload: Dict[str, Any]) -> None:
        compact = {
            "job_id": str(payload.get("job_id", "") or ""),
            "file": Path(str(payload.get("file") or payload.get("completed_file") or "")).name,
            "mode": str(payload.get("mode", "") or ""),
            "status": topic.split("/", 1)[1] if "/" in topic else str(payload.get("status", "") or ""),
            "accepted": int(payload.get("accepted", 0) or 0),
            "duplicates": int(payload.get("duplicates", 0) or 0),
            "rejected": int(payload.get("rejected", 0) or 0),
            "duration_s": float(payload.get("duration_s", 0.0) or 0.0),
            "workspace_clean": bool(payload.get("workspace_clean", (payload.get("workspace") or {}).get("clean", False)) if isinstance(payload.get("workspace") or {}, dict) else payload.get("workspace_clean", False)),
            "baseline_restored": bool(payload.get("baseline_restored", (payload.get("workspace") or {}).get("baseline_restored", False)) if isinstance(payload.get("workspace") or {}, dict) else payload.get("baseline_restored", False)),
            "reason": str(payload.get("blocked_reason", payload.get("error", "")) or ""),
        }
        await self.orch.push_event(
            topic,
            compact,
            source="read_sidecar",
            meta={
                "source": "read_sidecar",
                "channel": "internal",
                "kind": topic.replace("/", "_"),
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "self_output_track": False,
                "cognitive_visible": True,
            },
        )

    def _resolve_slearn_dir(self) -> Path:
        raw = str(self.orch.kv_store.get("slearn:dir", "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        default = self.memdir / "slearn_dir"
        self.orch.kv_store["slearn:dir"] = str(default)
        return default

    def _resolve_read_dir(self) -> Path:
        raw = str(self.orch.kv_store.get("read:dir", "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        # Canonical read intake now matches the explicit reading writer:
        #   <memdir>/reading/queue
        # Legacy <memdir>/read_dir is still scanned by _list_candidates so old
        # habits do not silently fail.
        default = self.memdir / "reading" / "queue"
        self.orch.kv_store["read:dir"] = str(default)
        self.orch.kv_store["read:legacy_dir"] = str(self._legacy_read_dir())
        return default

    def _legacy_read_dir(self) -> Path:
        return self.memdir / "read_dir"

    def _resolve_read_ready_dir(self, read_dir: Path) -> Path:
        # If read_dir is a queue folder, put completed documents beside it:
        #   reading/queue -> reading/ready
        # Otherwise preserve the older read_dir/ready behavior.
        if read_dir.name.lower() == "queue":
            return read_dir.parent / "ready"
        return read_dir / "ready"

    async def _publish_read_status(self, result: Dict[str, Any], *, text: str) -> None:
        await self.orch.push_event(
            "read/status",
            dict(result),
            source="read_sidecar",
            meta={
                "source": "read_sidecar",
                "channel": "internal",
                "control": True,
                "store_in_memory": False,
            },
        )
        await self.orch.push_event(
            "ui/status",
            {"text": text, "style": "system", "channel": "default"},
            source="read_sidecar",
            meta={
                "control": True,
                "kind": "read_status",
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "self_output_track": False,
                "cognitive_visible": False,
            },
        )

    def _apply_slearn_status(self, *, result: Dict[str, Any], active_file: str, chunk_index: int, now: float) -> None:
        self.orch.kv_store["slearn:last_result"] = result
        self.orch.kv_store["slearn:active_file"] = active_file
        self.orch.kv_store["slearn:chunk_index"] = chunk_index
        self.orch.kv_store["slearn:last_activity_ts"] = now

        emitted = int(result.get("emitted", 0) or 0) if isinstance(result, dict) else 0
        if emitted > 0:
            self.orch.kv_store["slearn:rules_emitted_total"] = int(self.orch.kv_store.get("slearn:rules_emitted_total", 0) or 0) + emitted

        if isinstance(result, dict) and result.get("completed_file"):
            completed = list(self.orch.kv_store.get("slearn:completed_files", []) or [])
            completed.append({
                "file": result.get("completed_file"),
                "ready_path": result.get("ready_path", ""),
                "ts": result.get("ts", now),
                "chunks": result.get("completed_chunks", chunk_index),
            })
            self.orch.kv_store["slearn:completed_files"] = completed[-25:]
            self.orch.kv_store["slearn:files_completed_count"] = int(self.orch.kv_store.get("slearn:files_completed_count", 0) or 0) + 1

    def _apply_status(self, *, result: Dict[str, Any], active_file: str, active_kind: str, chunk_index: int, now: float) -> None:
        self.orch.kv_store["read:last_result"] = result
        self.orch.kv_store["read:active_file"] = active_file
        self.orch.kv_store["read:active_kind"] = active_kind
        self.orch.kv_store["read:chunk_index"] = chunk_index
        self.orch.kv_store["read:last_activity_ts"] = now

    def _slearn_state_path(self, slearn_dir: Path) -> Path:
        return slearn_dir / "_slearn_state.json"

    def _load_slearn_state_file(self, slearn_dir: Path) -> Dict[str, Any]:
        path = self._slearn_state_path(slearn_dir)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_slearn_state_file(self, slearn_dir: Path, data: Dict[str, Any]) -> None:
        path = self._slearn_state_path(slearn_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _slearn_audit_path(self) -> Path:
        return self.memdir / "slearn" / "slearn_audit.jsonl"

    def _append_slearn_audit(self, entry: Dict[str, Any]) -> None:
        path = self._slearn_audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

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

    def _list_slearn_candidates(self, slearn_dir: Path) -> List[Path]:
        out: List[Path] = []
        for path in sorted(slearn_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("_slearn_state"):
                continue
            if path.suffix.lower() not in (".txt", ".md", ".slearn"):
                continue
            out.append(path)
        return out

    def _extract_slearn_rules(self, text: str) -> List[Tuple[int, str]]:
        rules: List[Tuple[int, str]] = []
        # /slearn accepts a broader CAPS grammar than /r did originally:
        #   IF USER says moin THEN CLASSIFY social_greeting AND REPLY good morning
        #   IF POWER is low THEN CLASSIFY need_power, energy_deficit
        #   IF OBJECT detected THEN CLASSIFY base_object
        #
        # Safety rule: the grammar rails must remain visibly uppercase. Domain
        # words such as "says" or "is" may remain natural language because they
        # are condition text, not control delimiters.
        for idx, raw in enumerate(str(text or "").splitlines()):
            line = strip_slearn_inline_comment(raw).strip()
            if not line or line.startswith("//"):
                continue
            if not line.startswith("IF ") or " THEN " not in line:
                continue
            # Exact grammar validation happens in SyntaxLearningNeuron.  This
            # prefilter only identifies candidate rule lines.
            rules.append((idx, line))
        return rules

    def _list_candidates(self, read_dir: Path) -> List[Path]:
        out: List[Path] = []
        roots: List[Path] = [read_dir]

        # Back-compat and human-forgiveness: scan the canonical queue plus the
        # old direct read_dir if it exists. This avoids the "/read is on but
        # nothing happens" failure when files are dropped into the older folder.
        legacy = self._legacy_read_dir()
        if legacy != read_dir:
            roots.append(legacy)

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_file():
                    continue
                if path.name.startswith("_read_state"):
                    continue
                if path.suffix.lower() not in (".txt", ".pdf", ".md"):
                    continue
                out.append(path)

        # Stable order, but prefer canonical queue over legacy when timestamps tie.
        out.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)))
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
        if suffix in (".txt", ".md", ".slearn"):
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
                topic="percept/reading",
                role="assistant",
                transport_source="reading",
                source=source_name,
                meta={
                    "channel": "reading",
                    "ingest_mode": "sensory",
                    "social_pressure": False,
                    "route": "eyes",
                    "source_kind": "document",
                    "source_name": source_name,
                },
                tier="now",
            )
            rows = (
                [result.get("utterance")]
                + list(result.get("tokens", []))
                + list(result.get("patterns", []))
                + list(result.get("word_roles", []))
                + list(result.get("thought_templates", []))
                + list(result.get("clause_frames", []))
                + list(result.get("general_patterns", []))
                + list(result.get("linkers", []))
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                kind = str(row.get("kind", "") or "")
                if kind == "utterance_anchor":
                    row["activation"] = min(float(row.get("activation", 0.14) or 0.14), 0.14)
                    row["promotion"] = min(float(row.get("promotion", 0.0) or 0.0), 0.0)
                    row["trust"] = min(float(row.get("trust", 0.22) or 0.22), 0.22)
                elif kind in {"general_pattern", "thought_template", "clause_frame", "pattern_linker"}:
                    row["activation"] = min(float(row.get("activation", 0.28) or 0.28), 0.28)
                    row["promotion"] = min(float(row.get("promotion", 0.02) or 0.02), 0.02)
                    row["trust"] = min(float(row.get("trust", 0.42) or 0.42), 0.42)
                else:
                    row["activation"] = min(float(row.get("activation", 0.18) or 0.18), 0.18)
                    row["promotion"] = min(float(row.get("promotion", 0.0) or 0.0), 0.0)
                    row["trust"] = min(float(row.get("trust", 0.30) or 0.30), 0.30)
                mem_cell_store.upsert_cell(row, tier=str(row.get("tier", "now") or "now"))
            count += 1
        return count
