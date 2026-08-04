from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Files at or above either threshold use the bucket path.  The decision is
# intentionally cheap and deterministic: file bytes + physical line count.
SLEARN_BUCKET_MIN_BYTES = 2 * 1024 * 1024
SLEARN_BUCKET_MIN_LINES = 5_000

# Normal jobs remain deliberately small and conversationally polite.  Bucket
# jobs use larger streaming slices but still report often enough for Window 2.
SLEARN_NORMAL_BATCH_LINES = 80
SLEARN_BUCKET_BATCH_LINES = 1_000

# Do not let bulk ingestion outrun the single-writer memory composer forever.
SLEARN_MAX_INFLIGHT_BATCHES = 64

# Bucket ingest deliberately coalesces pending learned writes before the single
# writer composer rewrites the canonical learned shard.  This prevents a 50k
# lexical file from forcing the ever-growing shard to be rewritten every 8k.
SLEARN_COMPOSER_FLUSH_BATCHES = 64

# Workspace hygiene only owns files below memdir/slearn/workspace.  Unknown
# residue is quarantined rather than deleted.
SLEARN_STALE_TMP_AGE_S = 300.0

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

SLEARN_PREFLIGHT_SCHEMA = "slearn.preflight.v1"
SLEARN_WORKSPACE_SCHEMA = "slearn.workspace.v1"


@dataclass(frozen=True)
class SlearnPreflight:
    file: str
    file_size_bytes: int
    line_count: int
    avg_line_bytes: float
    mode: str
    reason: str
    bucket_min_bytes: int
    bucket_min_lines: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SLEARN_PREFLIGHT_SCHEMA,
            "file": self.file,
            "file_size_bytes": self.file_size_bytes,
            "line_count": self.line_count,
            "total_lines": self.line_count,
            "avg_line_bytes": self.avg_line_bytes,
            "mode": self.mode,
            "selected_mode": self.mode,
            "reason": self.reason,
            "bucket_min_bytes": self.bucket_min_bytes,
            "bucket_min_lines": self.bucket_min_lines,
        }


@dataclass(frozen=True)
class SlearnLineBatch:
    lines: Tuple[Tuple[int, str], ...]
    start_line: int
    end_line: int
    byte_offset: int
    eof: bool


def scan_slearn_file(
    path: Path,
    *,
    bucket_min_bytes: int = SLEARN_BUCKET_MIN_BYTES,
    bucket_min_lines: int = SLEARN_BUCKET_MIN_LINES,
) -> SlearnPreflight:
    """Cheap size/line preflight without loading the file into memory."""

    stat = path.stat()
    size = int(stat.st_size)
    line_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if size > 0 and last_byte != b"\n":
        line_count += 1

    byte_hit = size >= max(1, int(bucket_min_bytes))
    line_hit = line_count >= max(1, int(bucket_min_lines))
    mode = "bucket" if (byte_hit or line_hit) else "normal"
    if byte_hit and line_hit:
        reason = "byte_and_line_threshold"
    elif byte_hit:
        reason = "byte_threshold"
    elif line_hit:
        reason = "line_threshold"
    else:
        reason = "below_bulk_thresholds"

    return SlearnPreflight(
        file=str(path),
        file_size_bytes=size,
        line_count=line_count,
        avg_line_bytes=(float(size) / float(line_count)) if line_count else 0.0,
        mode=mode,
        reason=reason,
        bucket_min_bytes=int(bucket_min_bytes),
        bucket_min_lines=int(bucket_min_lines),
    )


def read_line_batch(
    path: Path,
    *,
    byte_offset: int,
    line_number: int,
    max_lines: int,
) -> SlearnLineBatch:
    """Read the next physical-line bucket using a resumable byte cursor.

    This is intentionally binary underneath so ``tell()`` remains a stable byte
    offset on Windows too.  Decoding is per-line and bad bytes are ignored.
    """

    rows: List[Tuple[int, str]] = []
    start_line = max(0, int(line_number)) + 1
    current_line = max(0, int(line_number))
    offset = max(0, int(byte_offset))
    eof = False
    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(max(1, int(max_lines))):
            raw = handle.readline()
            if not raw:
                eof = True
                break
            current_line += 1
            rows.append((current_line, raw.decode("utf-8", errors="ignore").rstrip("\r\n")))
        offset = int(handle.tell())
        if not eof:
            probe = handle.read(1)
            eof = not bool(probe)

    return SlearnLineBatch(
        lines=tuple(rows),
        start_line=start_line,
        end_line=current_line,
        byte_offset=offset,
        eof=eof,
    )


def byte_offset_for_line(path: Path, line_number: int) -> int:
    """Return the byte offset immediately after ``line_number`` physical lines."""
    target = max(0, int(line_number))
    if target <= 0:
        return 0
    seen = 0
    with path.open("rb") as handle:
        while seen < target:
            raw = handle.readline()
            if not raw:
                break
            seen += 1
        return int(handle.tell())


def stable_job_id(path: Path) -> str:
    try:
        stat = path.stat()
        source = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except Exception:
        source = str(path)
    digest = hashlib.blake2b(source.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    return f"job-{digest}"


class SlearnWorkspaceCleaner:
    """Own only SLEARN's scratch floor; never prune durable MB memory."""

    def __init__(self, memdir: Path) -> None:
        self.root = Path(memdir) / "slearn" / "workspace"
        self.quarantine = self.root / "quarantine"

    def snapshot(self) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(parents=True, exist_ok=True)
        job_dirs = [p for p in self.root.glob("job-*") if p.is_dir()]
        tmp_files = [p for p in self.root.rglob("*.tmp") if self.quarantine not in p.parents]
        scratch_files = [
            p for p in self.root.rglob("*")
            if p.is_file() and self.quarantine not in p.parents
        ]
        return {
            "schema": SLEARN_WORKSPACE_SCHEMA,
            "job_dirs": len(job_dirs),
            "tmp_files": len(tmp_files),
            "scratch_files": len(scratch_files),
            "ts": time.time(),
        }

    def prepare(self, *, job_id: str, source_path: Path) -> Dict[str, Any]:
        before = self.snapshot()
        actions: List[str] = []
        now = time.time()

        # Temp files are job-local scratch, not durable learning.  Only stale
        # temps are deleted; fresh/unknown files are left alone.
        for tmp in list(self.root.rglob("*.tmp")):
            if self.quarantine in tmp.parents:
                continue
            try:
                if (now - tmp.stat().st_mtime) >= SLEARN_STALE_TMP_AGE_S:
                    tmp.unlink(missing_ok=True)
                    actions.append(f"removed_stale_tmp:{tmp.name}")
            except OSError:
                continue

        # Any other unfinished job is unknown state.  Move it out of the active
        # work floor instead of deleting it.
        for other in list(self.root.glob("job-*")):
            if not other.is_dir() or other.name == job_id:
                continue
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            target = self.quarantine / f"{other.name}-{stamp}"
            n = 1
            while target.exists():
                target = self.quarantine / f"{other.name}-{stamp}-{n}"
                n += 1
            try:
                shutil.move(str(other), str(target))
                actions.append(f"quarantined:{other.name}")
            except OSError:
                continue

        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "schema": SLEARN_WORKSPACE_SCHEMA,
            "job_id": job_id,
            "source_path": str(source_path),
            "status": "running",
            "prepared_at": now,
        }
        self._write_json_atomic(job_dir / "job.json", marker)
        after = self.snapshot()
        return {
            "schema": SLEARN_WORKSPACE_SCHEMA,
            "baseline": before,
            "after_preclean": after,
            "actions": actions,
            "clean": after.get("tmp_files", 0) == 0,
            "baseline_restored": False,
            "job_dir": str(job_dir),
        }

    def update_job(self, job_id: str, payload: Mapping[str, Any]) -> None:
        job_dir = self.root / job_id
        if not job_dir.exists():
            return
        marker = dict(payload)
        marker.setdefault("schema", SLEARN_WORKSPACE_SCHEMA)
        marker.setdefault("job_id", job_id)
        marker["updated_at"] = time.time()
        self._write_json_atomic(job_dir / "job.json", marker)

    def finish(self, *, job_id: str, baseline: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        actions: List[str] = []
        job_dir = self.root / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                actions.append(f"removed_job_workspace:{job_id}")
            except OSError:
                pass

        # No current job owns a .tmp once durable commit has been confirmed.
        for tmp in list(self.root.rglob("*.tmp")):
            if self.quarantine in tmp.parents:
                continue
            try:
                tmp.unlink(missing_ok=True)
                actions.append(f"removed_tmp:{tmp.name}")
            except OSError:
                continue

        after = self.snapshot()
        base = dict(baseline or {})
        restored = (
            int(after.get("tmp_files", 0) or 0) <= int(base.get("tmp_files", 0) or 0)
            and int(after.get("job_dirs", 0) or 0) <= int(base.get("job_dirs", 0) or 0)
        )
        return {
            "schema": SLEARN_WORKSPACE_SCHEMA,
            "clean": int(after.get("tmp_files", 0) or 0) == 0,
            "baseline_restored": restored,
            "baseline": base,
            "after_cleanup": after,
            "actions": actions,
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
