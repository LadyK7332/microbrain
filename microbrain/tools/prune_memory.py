"""
Pruning utilities for MicroBrain memory folders.

Phase 1: Episodes only (low-risk, high-win).
- Archives old episode JSONL files to .gz under episodes/archive/
- Maintains a lightweight episodes/index.json metadata file
- Keeps "correction-heavy" episodes (no/stop/don't/bad) even if old
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

_CORRECTION_TOKENS = {"no", "stop", "don't", "dont", "bad"}
_EP_RE = re.compile(r"episode-(\d{8})-(\d{6})-pid(\d+)\.jsonl(\.gz)?$", re.IGNORECASE)


@dataclass
class PruneReport:
    memdir: str
    keep_days: int
    candidates: int = 0
    kept: int = 0
    archived: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def _now_ts() -> float:
    return time.time()


def _cutoff_ts(keep_days: int) -> float:
    return _now_ts() - (keep_days * 86400.0)


def _safe_read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _is_old(path: Path, cutoff: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff
    except Exception:
        return False


def _episode_has_correction(path: Path) -> bool:
    """
    Stream-scan the JSONL episode for a correction token in percept/text,
    or a curiosity/adjust event (which implies feedback).
    """
    try:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                topic = str(obj.get("topic", ""))
                if topic == "curiosity/adjust":
                    return True
                if topic == "percept/text":
                    payload = obj.get("payload", {}) or {}
                    txt = str(payload.get("text", "")).strip().lower()
                    if not txt:
                        continue
                    # token-level match; keep it simple
                    toks = [t for t in re.split(r"\s+", txt) if t]
                    if any(t in _CORRECTION_TOKENS for t in toks):
                        return True
        return False
    except Exception:
        return False


def _gzip_archive(src: Path, dst: Path) -> Tuple[int, int]:
    """
    Compress src JSONL -> dst .gz. Returns (bytes_before, bytes_after).
    """
    before = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    after = dst.stat().st_size
    return before, after


def prune_episodes(
    memdir: str,
    keep_days: int = 14,
    apply: bool = False,
    keep_corrections: bool = True,
) -> PruneReport:
    """
    Phase 1: Episodes only.
    """
    report = PruneReport(memdir=str(memdir), keep_days=int(keep_days))
    base = Path(memdir)
    eps_dir = base / "episodes"
    archive_dir = eps_dir / "archive"
    index_path = eps_dir / "index.json"

    cutoff = _cutoff_ts(keep_days)

    if not eps_dir.exists():
        report.notes.append("episodes_dir_missing")
        return report

    index = _safe_read_json(index_path, default={"episodes": []})
    episodes_meta: list[dict[str, Any]] = list(index.get("episodes", []))

    # Build a lookup for existing entries
    by_name: Dict[str, Dict[str, Any]] = {e.get("name"): e for e in episodes_meta if isinstance(e, dict)}

    # Candidates: raw .jsonl (not gz) that are old
    episode_files = sorted([p for p in eps_dir.glob("episode-*.jsonl") if p.is_file()])
    report.candidates = len(episode_files)

    for ep in episode_files:
        name = ep.name
        entry = by_name.get(name, {"name": name})
        entry.setdefault("path", str(ep))
        entry["mtime"] = ep.stat().st_mtime
        entry["size"] = ep.stat().st_size
        entry["status"] = entry.get("status", "raw")

        # Not old? keep
        if not _is_old(ep, cutoff):
            entry["status"] = "raw"
            report.kept += 1
            by_name[name] = entry
            continue

        # Old: decide whether to keep raw or archive
        keep_raw = False
        if keep_corrections:
            keep_raw = _episode_has_correction(ep)

        if keep_raw:
            entry["status"] = "raw_kept_correction"
            report.kept += 1
            by_name[name] = entry
            continue

        # Archive target name
        gz_name = name + ".gz"
        dst = archive_dir / gz_name

        if apply:
            b_before, b_after = _gzip_archive(ep, dst)
            report.bytes_before += b_before
            report.bytes_after += b_after
            # remove original after successful gzip
            ep.unlink(missing_ok=True)
            entry["status"] = "archived"
            entry["archived_path"] = str(dst)
            entry["archived_size"] = b_after
            report.archived += 1
        else:
            # Dry run estimate: assume 35% of original size (rough)
            b_before = ep.stat().st_size
            report.bytes_before += b_before
            report.bytes_after += int(b_before * 0.35)
            entry["status"] = "would_archive"
            entry["archived_path"] = str(dst)
            report.archived += 1

        by_name[name] = entry

    # Write index back (apply and dry-run both update index for transparency)
    new_list = sorted(by_name.values(), key=lambda e: e.get("mtime", 0), reverse=True)
    index["episodes"] = new_list

    # Only write index if apply OR index exists already (avoid creating eps_dir in dry-run)
    try:
        if apply or index_path.exists():
            _write_json(index_path, index)
    except Exception:
        # Don't fail pruning because index couldn't write
        report.notes.append("index_write_failed")

    return report


def format_report(report: PruneReport) -> str:
    saved = max(0, report.bytes_before - report.bytes_after)
    def _fmt(n: int) -> str:
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024:
                return f"{n:.0f}{unit}" if unit=="B" else f"{n/1.0:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}PB"
    lines = [
        f"PruneReport(memdir={report.memdir})",
        f" keep_days={report.keep_days}",
        f" candidates={report.candidates}",
        f" kept={report.kept}",
        f" archived={report.archived}",
        f" bytes_before≈{_fmt(int(report.bytes_before))}",
        f" bytes_after≈{_fmt(int(report.bytes_after))}",
        f" est_saved≈{_fmt(int(saved))}",
    ]
    if report.notes:
        lines.append(" notes=" + ",".join(report.notes))
    return "\n".join(lines)
