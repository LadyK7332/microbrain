from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from microbrain.memory.mem_cell_store import MemCellStore, TIERS


class MemCellComposer:
    """Single-writer composer for mem-cell shard files.

    Normal MB organs should stage pending row updates under
    mem_cell/_pending/<tier>/.  This composer is the only desk goblin that
    merges those staged rows and rewrites the canonical tier shard.
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        lock_timeout_s: float = 2.0,
        stale_lock_after_s: float = 180.0,
        max_files_per_tier: int = 512,
        archive_applied: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.mem_cell_dir = self.base_dir / "mem_cell"
        self.pending_root = self.mem_cell_dir / "_pending"
        self.processing_root = self.mem_cell_dir / "_processing"
        self.applied_root = self.mem_cell_dir / "_applied"
        self.lock_path = self.mem_cell_dir / "_composer.lock"
        self.lock_timeout_s = float(lock_timeout_s)
        self.stale_lock_after_s = float(stale_lock_after_s)
        self.max_files_per_tier = max(1, int(max_files_per_tier))
        self.archive_applied = bool(archive_applied)

    @contextmanager
    def _composer_lock(self) -> Iterator[None]:
        self.mem_cell_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.1, self.lock_timeout_s)
        owner = f"pid={os.getpid()} ts={time.time():.6f}\n"
        fd: int | None = None
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, owner.encode("utf-8", errors="replace"))
                break
            except FileExistsError:
                try:
                    age_s = time.time() - self.lock_path.stat().st_mtime
                    if age_s > self.stale_lock_after_s:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for mem-cell composer lock: {self.lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _recover_processing_files(self, tier: str) -> None:
        tier = self._coerce_tier(tier)
        processing_dir = self.processing_root / tier
        if not processing_dir.exists():
            return
        pending_dir = self.pending_root / tier
        pending_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(processing_dir.glob("*.processing")):
            target_name = path.name.split(".processing", 1)[0]
            target = pending_dir / target_name
            n = 1
            while target.exists():
                target = pending_dir / f"{target_name}.{n}"
                n += 1
            try:
                path.replace(target)
            except OSError:
                continue

    def pending_files(self, tier: str) -> List[Path]:
        tier = self._coerce_tier(tier)
        self._recover_processing_files(tier)
        pending_dir = self.pending_root / tier
        if not pending_dir.exists():
            return []
        return sorted(
            [p for p in pending_dir.glob("*.jsonl") if p.is_file() and not p.name.startswith(".")],
            key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, p.name),
        )[: self.max_files_per_tier]

    def pending_count(self) -> Dict[str, int]:
        return {tier: len(self.pending_files(tier)) for tier in TIERS}

    def compose_once(self) -> Dict[str, Any]:
        """Drain pending files once and rewrite canonical tier shards.

        Returns a compact status dict suitable for UI/status KV.  Raises only
        for lock acquisition failures; malformed pending lines are skipped and
        reported in the status.
        """
        status: Dict[str, Any] = {
            "started_at": time.time(),
            "tiers": {},
            "files_processed": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "bad_lines": 0,
        }
        with self._composer_lock():
            for tier in TIERS:
                tier_status = self._compose_tier(tier)
                if tier_status["files_processed"] or tier_status["rows_applied"] or tier_status["bad_lines"]:
                    status["tiers"][tier] = tier_status
                    status["files_processed"] += int(tier_status["files_processed"])
                    status["rows_applied"] += int(tier_status["rows_applied"])
                    status["reinforcements_applied"] += int(tier_status.get("reinforcements_applied", 0))
                    status["bad_lines"] += int(tier_status["bad_lines"])
        status["finished_at"] = time.time()
        status["elapsed_s"] = round(status["finished_at"] - status["started_at"], 4)
        return status

    def _compose_tier(self, tier: str) -> Dict[str, Any]:
        tier = self._coerce_tier(tier)
        files = self.pending_files(tier)
        status = {
            "files_processed": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "bad_lines": 0,
            "pending_remaining": 0,
        }
        if not files:
            status["pending_remaining"] = 0
            return status

        # Direct mode store is the only canonical writer.
        store = MemCellStore(self.base_dir, composer_enabled=False, writer_id="memory_composer")
        operations: List[Dict[str, Any]] = []
        processing_dir = self.processing_root / tier
        processing_dir.mkdir(parents=True, exist_ok=True)
        moved_files: List[Path] = []

        for path in files:
            processing = processing_dir / f"{path.name}.{os.getpid()}.processing"
            try:
                path.replace(processing)
            except FileNotFoundError:
                continue
            except PermissionError:
                # Another process may still be finishing the file; leave it for
                # the next composer cycle instead of fighting Windows.
                continue
            moved_files.append(processing)
            status["files_processed"] += 1
            for envelope in self._read_pending_file(processing):
                if envelope is None:
                    status["bad_lines"] += 1
                    continue
                op = str(envelope.get("op", "upsert") or "upsert") if isinstance(envelope, dict) else ""
                if op == "reinforce":
                    update = envelope.get("update") if isinstance(envelope, dict) else None
                    if not isinstance(update, dict) or not str(update.get("cell_id", "") or "").strip():
                        status["bad_lines"] += 1
                        continue
                    operations.append({"op": "reinforce", "update": dict(update)})
                    continue

                row = envelope.get("row") if isinstance(envelope, dict) else None
                if not isinstance(row, dict):
                    status["bad_lines"] += 1
                    continue
                operations.append({"op": "upsert", "row": dict(row), "touch": bool(envelope.get("touch", True))})

        if operations:
            touched_tiers: set[str] = set()
            for operation in operations:
                if operation.get("op") == "reinforce":
                    update = dict(operation.get("update", {}) or {})
                    update_tier = self._coerce_tier(str(update.get("tier", tier) or tier))
                    if store.apply_reinforcement(update, tier=update_tier, flush=False) is not None:
                        status["reinforcements_applied"] += 1
                        touched_tiers.add(update_tier)
                    continue

                row = dict(operation.get("row", {}) or {})
                row_tier = self._coerce_tier(str(row.get("tier", tier) or tier))
                store.upsert_cell(row, tier=row_tier, touch=bool(operation.get("touch", True)), flush=False)
                status["rows_applied"] += 1
                touched_tiers.add(row_tier)

            # One canonical write per touched tier, not one per writer.
            for touched_tier in sorted(touched_tiers):
                store.flush_tier(touched_tier)

        for processing in moved_files:
            try:
                if self.archive_applied:
                    applied_dir = self.applied_root / tier
                    applied_dir.mkdir(parents=True, exist_ok=True)
                    target = applied_dir / f"{processing.name}.done"
                    n = 1
                    while target.exists():
                        target = applied_dir / f"{processing.name}.{n}.done"
                        n += 1
                    processing.replace(target)
                else:
                    processing.unlink(missing_ok=True)
            except OSError:
                pass

        status["pending_remaining"] = len(self.pending_files(tier))
        return status

    def _read_pending_file(self, path: Path) -> Iterable[Dict[str, Any] | None]:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield obj if isinstance(obj, dict) else None
                    except Exception:
                        yield None
        except Exception:
            yield None

    @staticmethod
    def _coerce_tier(tier: str) -> str:
        tier = str(tier or "now").strip().lower()
        return tier if tier in TIERS else "now"
