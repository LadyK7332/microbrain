from __future__ import annotations

import json
import os
import threading
import time
import uuid
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
        self.owner_id = f"composer-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
        self._telemetry_lock = threading.RLock()
        self._telemetry: Dict[str, Any] = {
            "phase": "idle",
            "tier": "",
            "file": "",
            "detail": "",
            "phase_started_ts": time.time(),
            "phase_updated_ts": time.time(),
            "files_selected": 0,
            "files_moved": 0,
            "operations_loaded": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "flush_tier": "",
            "lock_owner": "",
            "lock_recovery_reason": "",
            "lock_recoveries": 0,
            "skipped_lock_no_work": False,
        }

    def _set_phase(self, phase: str, *, tier: str = "", path: Path | str | None = None, detail: str = "", **extra: Any) -> None:
        """Publish thread-safe fine-grained composer progress.

        The sidecar runs compose_once() in a worker thread.  This snapshot is
        intentionally small so the UI can tell whether the composer is waiting
        on the lock, scanning a tier, reading a pending file, applying rows, or
        flushing a shard without touching the same storage path itself.
        """
        now = time.time()
        name = ""
        if path is not None:
            try:
                name = Path(path).name
            except Exception:
                name = str(path)
        with self._telemetry_lock:
            prior_phase = str(self._telemetry.get("phase", "") or "")
            prior_tier = str(self._telemetry.get("tier", "") or "")
            prior_file = str(self._telemetry.get("file", "") or "")
            if phase != prior_phase or tier != prior_tier or name != prior_file:
                self._telemetry["phase_started_ts"] = now
            self._telemetry.update({
                "phase": str(phase or ""),
                "tier": str(tier or ""),
                "file": name,
                "detail": str(detail or ""),
                "phase_updated_ts": now,
            })
            for key, value in extra.items():
                self._telemetry[str(key)] = value

    def telemetry_snapshot(self, *, now: float | None = None) -> Dict[str, Any]:
        now_ts = float(now if now is not None else time.time())
        with self._telemetry_lock:
            snap = dict(self._telemetry)
        started = float(snap.get("phase_started_ts", 0.0) or 0.0)
        updated = float(snap.get("phase_updated_ts", 0.0) or 0.0)
        snap["phase_age_s"] = round(max(0.0, now_ts - started), 3) if started else 0.0
        snap["phase_pulse_age_s"] = round(max(0.0, now_ts - updated), 3) if updated else 0.0
        return snap

    @staticmethod
    def _parse_lock_owner(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in str(text or "").replace("\n", " ").split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                out[key] = value
        return out

    def _read_lock_owner(self) -> tuple[str, Dict[str, str]]:
        try:
            text = self.lock_path.read_text(encoding="utf-8", errors="replace")[:2048]
        except OSError:
            return "", {}
        return text, self._parse_lock_owner(text)

    @staticmethod
    def _owner_pid_is_alive(pid: int) -> bool | None:
        """Best-effort live PID check without psutil.

        Return True when the process appears alive, False when it is known dead,
        and None when the platform refuses to say.  On Windows this uses
        OpenProcess/GetExitCodeProcess so signal 0 never risks terminating the
        owner process.
        """
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                kernel32 = ctypes.windll.kernel32
                kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenProcess.restype = wintypes.HANDLE
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
                if not handle:
                    return False
                try:
                    code = wintypes.DWORD()
                    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                        return None
                    return int(code.value) == STILL_ACTIVE
                finally:
                    kernel32.CloseHandle(handle)
            except Exception:
                return None
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return None

    def _lock_recovery_reason(self) -> tuple[str, str]:
        """Return (reason, owner_text) when an existing lock can be broken."""
        owner_text, owner = self._read_lock_owner()
        owner_pid = 0
        try:
            owner_pid = int(str(owner.get("pid", "0") or "0"))
        except Exception:
            owner_pid = 0
        try:
            age_s = max(0.0, time.time() - self.lock_path.stat().st_mtime)
        except OSError:
            return "vanished", owner_text

        alive = self._owner_pid_is_alive(owner_pid) if owner_pid else None
        if alive is False:
            return "dead_owner_pid", owner_text
        if age_s > self.stale_lock_after_s:
            return "stale_lock", owner_text
        return "", owner_text

    def _break_recoverable_lock(self) -> bool:
        reason, owner_text = self._lock_recovery_reason()
        if not reason:
            with self._telemetry_lock:
                self._telemetry["lock_owner"] = owner_text[:240]
                self._telemetry["lock_recovery_reason"] = ""
            return False
        self._set_phase(
            "lock_recover",
            detail=f"{reason}: {self.lock_path}",
            lock_owner=owner_text[:240],
            lock_recovery_reason=reason,
            lock_recoveries=int(self._telemetry.get("lock_recoveries", 0) or 0) + 1,
        )
        try:
            self.lock_path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _tier_may_have_work(self, tier: str) -> bool:
        """Cheap no-lock preflight used to avoid fighting a lock for zero work."""
        tier = self._coerce_tier(tier)
        pending_dir = self.pending_root / tier
        processing_dir = self.processing_root / tier
        try:
            if pending_dir.exists():
                with os.scandir(pending_dir) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file() and entry.name.endswith(".jsonl") and not entry.name.startswith("."):
                                return True
                        except OSError:
                            continue
        except OSError:
            return True
        try:
            if processing_dir.exists():
                with os.scandir(processing_dir) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file() and entry.name.endswith(".processing"):
                                return True
                        except OSError:
                            continue
        except OSError:
            return True
        return False

    def _processing_candidates_for_receipt(self, tier: str, receipt: Path) -> List[Path]:
        """Return possible processing paths for one original pending receipt."""
        tier = self._coerce_tier(tier)
        processing_dir = self.processing_root / tier
        if not processing_dir.exists():
            return []
        prefix = f"{receipt.name}."
        out: List[Path] = []
        try:
            with os.scandir(processing_dir) as entries:
                for entry in entries:
                    name = entry.name
                    if not name.startswith(prefix) or not name.endswith(".processing"):
                        continue
                    try:
                        if entry.is_file():
                            out.append(processing_dir / name)
                    except OSError:
                        continue
        except OSError:
            return []
        return sorted(out, key=lambda p: p.name)

    def receipt_paths(self, receipts: Sequence[str | Path], *, tier: str = "learned") -> List[Path]:
        """Resolve exact pending/processing receipt files without scanning the tier backlog.

        SLEARN keeps exact composer receipt paths in its job state.  During a
        waiting-commit flush, those paths are the only files the job is waiting
        on.  A large learned backlog from older jobs must not force the composer
        to walk or process unrelated pending files before acknowledging the
        current SLEARN receipts.
        """
        tier = self._coerce_tier(tier)
        out: List[Path] = []
        seen: set[str] = set()
        for raw in receipts or []:
            receipt = Path(str(raw or ""))
            if not str(receipt):
                continue
            candidates: List[Path] = []
            try:
                if receipt.exists():
                    candidates.append(receipt)
            except OSError:
                candidates.append(receipt)
            candidates.extend(self._processing_candidates_for_receipt(tier, receipt))
            for candidate in candidates:
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
        return out

    @contextmanager
    def _composer_lock(self) -> Iterator[None]:
        self.mem_cell_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.1, self.lock_timeout_s)
        owner = f"pid={os.getpid()} thread={threading.get_ident()} owner_id={self.owner_id} ts={time.time():.6f}\n"
        fd: int | None = None
        self._set_phase("lock_wait", detail=str(self.lock_path), lock_owner="", lock_recovery_reason="")
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, owner.encode("utf-8", errors="replace"))
                self._set_phase("lock_acquired", detail=str(self.lock_path), lock_owner=owner.strip())
                break
            except FileExistsError:
                if self._break_recoverable_lock():
                    continue
                if time.monotonic() >= deadline:
                    reason, owner_text = self._lock_recovery_reason()
                    detail = f"timed out waiting for mem-cell composer lock: {self.lock_path}"
                    if owner_text:
                        detail += f" | owner {owner_text.strip()[:180]}"
                    if reason:
                        detail += f" | recovery pending {reason}"
                    raise TimeoutError(detail)
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
            self._set_phase("lock_released")

    def _recover_processing_files(self, tier: str) -> None:
        tier = self._coerce_tier(tier)
        self._set_phase("recover_processing", tier=tier)
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
        self._set_phase("scan_pending", tier=tier)
        pending_dir = self.pending_root / tier
        if not pending_dir.exists():
            return []

        # Avoid a full stat/sort walk on network-backed or very large pending
        # dirs.  Composer only needs a bounded batch; deterministic filename order
        # is useful but not worth freezing MB on the wrong tier.
        out: List[Path] = []
        try:
            with os.scandir(pending_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file() or entry.name.startswith(".") or not entry.name.endswith(".jsonl"):
                            continue
                    except OSError:
                        continue
                    out.append(pending_dir / entry.name)
                    if len(out) >= self.max_files_per_tier:
                        break
        except OSError:
            return []
        return sorted(out, key=lambda p: p.name)

    def pending_count(self) -> Dict[str, int]:
        return {tier: len(self.pending_files(tier)) for tier in TIERS}

    def compose_once(self, *, tiers: Sequence[str] | None = None) -> Dict[str, Any]:
        """Drain pending files once and rewrite canonical tier shards.

        Returns a compact status dict suitable for UI/status KV.  Raises only
        for lock acquisition failures; malformed pending lines are skipped and
        reported in the status.
        """
        selected_tiers = list(TIERS) if tiers is None else [str(t) for t in tiers if str(t) in TIERS]
        # Preserve order while removing duplicates.
        selected_tiers = list(dict.fromkeys(selected_tiers))
        self._set_phase("compose_start", detail=",".join(selected_tiers), tiers_requested=list(selected_tiers), skipped_lock_no_work=False)
        status: Dict[str, Any] = {
            "started_at": time.time(),
            "tiers": {},
            "tiers_requested": list(selected_tiers),
            "tiers_actionable": [],
            "files_processed": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "bad_lines": 0,
            "skipped_lock_no_work": False,
        }
        actionable_tiers = [tier for tier in selected_tiers if self._tier_may_have_work(tier)]
        status["tiers_actionable"] = list(actionable_tiers)
        if not actionable_tiers:
            status["finished_at"] = time.time()
            status["elapsed_s"] = round(status["finished_at"] - status["started_at"], 4)
            status["skipped_lock_no_work"] = True
            self._set_phase(
                "idle",
                detail="no eligible pending work",
                tiers_requested=list(selected_tiers),
                tiers_actionable=[],
                skipped_lock_no_work=True,
                files_processed=0,
                rows_applied=0,
                reinforcements_applied=0,
            )
            return status
        try:
            with self._composer_lock():
                for tier in actionable_tiers:
                    self._set_phase("tier_start", tier=tier)
                    tier_status = self._compose_tier(tier)
                    if tier_status["files_processed"] or tier_status["rows_applied"] or tier_status["bad_lines"]:
                        status["tiers"][tier] = tier_status
                        status["files_processed"] += int(tier_status["files_processed"])
                        status["rows_applied"] += int(tier_status["rows_applied"])
                        status["reinforcements_applied"] += int(tier_status.get("reinforcements_applied", 0))
                        status["bad_lines"] += int(tier_status["bad_lines"])
            status["finished_at"] = time.time()
            status["elapsed_s"] = round(status["finished_at"] - status["started_at"], 4)
            self._set_phase(
                "idle",
                detail="compose complete",
                files_processed=int(status.get("files_processed", 0) or 0),
                rows_applied=int(status.get("rows_applied", 0) or 0),
                reinforcements_applied=int(status.get("reinforcements_applied", 0) or 0),
            )
            return status
        except Exception as exc:
            self._set_phase("error", detail=f"{type(exc).__name__}: {exc}")
            raise

    def compose_receipts(self, receipts: Sequence[str | Path], *, tier: str = "learned") -> Dict[str, Any]:
        """Commit exactly the supplied SLEARN receipt files.

        This is the focused commit path for bulk SLEARN.  It does not enumerate
        mem_cell/_pending/<tier>/ and therefore cannot get trapped behind a huge
        stale learned backlog while the active job waits on a small receipt list.
        """
        tier = self._coerce_tier(tier)
        files = self.receipt_paths(receipts, tier=tier)
        self._set_phase(
            "compose_receipts_start",
            tier=tier,
            detail=f"{len(files)}/{len(list(receipts or []))} exact receipt(s)",
            tiers_requested=[tier],
            tiers_actionable=[tier] if files else [],
            receipt_focused=True,
            receipts_requested=len(list(receipts or [])),
            receipts_found=len(files),
            skipped_lock_no_work=False,
        )
        status: Dict[str, Any] = {
            "started_at": time.time(),
            "tiers": {},
            "tiers_requested": [tier],
            "tiers_actionable": [tier] if files else [],
            "files_processed": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "bad_lines": 0,
            "skipped_lock_no_work": False,
            "receipt_focused": True,
            "receipts_requested": len(list(receipts or [])),
            "receipts_found": len(files),
        }
        if not files:
            status["finished_at"] = time.time()
            status["elapsed_s"] = round(status["finished_at"] - status["started_at"], 4)
            status["skipped_lock_no_work"] = True
            self._set_phase(
                "idle",
                tier=tier,
                detail="no pending SLEARN receipts",
                receipt_focused=True,
                receipts_found=0,
                skipped_lock_no_work=True,
            )
            return status
        try:
            with self._composer_lock():
                self._set_phase("tier_start", tier=tier, detail="exact SLEARN receipts", receipt_focused=True)
                tier_status = self._compose_tier(tier, files=files, receipt_focused=True)
                if tier_status["files_processed"] or tier_status["rows_applied"] or tier_status["bad_lines"]:
                    status["tiers"][tier] = tier_status
                    status["files_processed"] += int(tier_status["files_processed"])
                    status["rows_applied"] += int(tier_status["rows_applied"])
                    status["reinforcements_applied"] += int(tier_status.get("reinforcements_applied", 0))
                    status["bad_lines"] += int(tier_status["bad_lines"])
            status["finished_at"] = time.time()
            status["elapsed_s"] = round(status["finished_at"] - status["started_at"], 4)
            self._set_phase(
                "idle",
                detail="exact receipt compose complete",
                receipt_focused=True,
                files_processed=int(status.get("files_processed", 0) or 0),
                rows_applied=int(status.get("rows_applied", 0) or 0),
                reinforcements_applied=int(status.get("reinforcements_applied", 0) or 0),
            )
            return status
        except Exception as exc:
            self._set_phase("error", detail=f"{type(exc).__name__}: {exc}", receipt_focused=True)
            raise

    def _compose_tier(self, tier: str, *, files: Sequence[Path] | None = None, receipt_focused: bool = False) -> Dict[str, Any]:
        tier = self._coerce_tier(tier)
        files = list(files) if files is not None else self.pending_files(tier)
        self._set_phase("files_selected", tier=tier, detail=f"{len(files)} pending file(s)", files_selected=len(files), receipt_focused=receipt_focused)
        status = {
            "files_processed": 0,
            "rows_applied": 0,
            "reinforcements_applied": 0,
            "bad_lines": 0,
            "pending_remaining": 0,
        }
        if not files:
            status["pending_remaining"] = 0
            self._set_phase("tier_idle", tier=tier, detail="no pending files", files_selected=0)
            return status

        # Direct mode store is the only canonical writer.
        store = MemCellStore(self.base_dir, composer_enabled=False, writer_id="memory_composer")
        operations: List[Dict[str, Any]] = []
        processing_dir = self.processing_root / tier
        processing_dir.mkdir(parents=True, exist_ok=True)
        moved_files: List[Path] = []

        for path in files:
            processing: Path
            if str(path.name).endswith(".processing") and path.parent == processing_dir:
                # Recovery path: this exact receipt had already been moved into
                # _processing before a crash/restart.  Consume it in place.
                self._set_phase("resume_processing", tier=tier, path=path, detail="resuming exact receipt", files_moved=len(moved_files), receipt_focused=receipt_focused)
                processing = path
                try:
                    if not processing.exists():
                        continue
                except OSError:
                    continue
            else:
                self._set_phase("move_pending", tier=tier, path=path, detail="pending -> processing", files_moved=len(moved_files), receipt_focused=receipt_focused)
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
            self._set_phase("read_pending", tier=tier, path=processing, detail="loading staged envelopes", files_moved=len(moved_files), receipt_focused=receipt_focused)
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

        self._set_phase("operations_loaded", tier=tier, detail=f"{len(operations)} operation(s)", operations_loaded=len(operations))
        if operations:
            touched_tiers: set[str] = set()
            for idx, operation in enumerate(operations, start=1):
                if idx == 1 or idx % 500 == 0 or idx == len(operations):
                    self._set_phase(
                        "apply_operations",
                        tier=tier,
                        detail=f"{idx}/{len(operations)} operation(s)",
                        operations_loaded=len(operations),
                        operations_applied=idx,
                        rows_applied=int(status.get("rows_applied", 0) or 0),
                        reinforcements_applied=int(status.get("reinforcements_applied", 0) or 0),
                    )
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
                self._set_phase("flush_tier", tier=tier, detail=f"flushing {touched_tier}", flush_tier=touched_tier)
                store.flush_tier(touched_tier)

        for processing in moved_files:
            self._set_phase("cleanup_processing", tier=tier, path=processing, detail="removing applied pending file")
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

        if receipt_focused:
            status["pending_remaining"] = len(self.receipt_paths(files, tier=tier))
        else:
            status["pending_remaining"] = len(self.pending_files(tier))
        self._set_phase(
            "tier_done",
            tier=tier,
            detail=f"remaining {status['pending_remaining']}",
            files_processed=int(status.get("files_processed", 0) or 0),
            rows_applied=int(status.get("rows_applied", 0) or 0),
            reinforcements_applied=int(status.get("reinforcements_applied", 0) or 0),
            receipt_focused=receipt_focused,
        )
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
