"""Audit MicroBrain modules for the canonical top-of-file tuning layout.

This tool is read-only. It reports files that do not yet contain both the
Behavioral tuning and Required static constants headings. It does not attempt to
classify or rewrite literals automatically because tunable behavior and required
static structure need human architectural judgment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Directories commonly containing independently tunable modules.
DEFAULT_SCAN_DIRS = (
    "microbrain/neurons",
    "microbrain/sidecars",
    "microbrain/patterns",
    "microbrain/memory",
)

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

BEHAVIORAL_HEADING = "# Behavioral tuning"
STATIC_HEADING = "# Required static constants"
PYTHON_SUFFIX = ".py"
IGNORED_FILENAMES = {"__init__.py"}


def iter_python_files(repo_root: Path, scan_dirs: Iterable[str]) -> Iterable[Path]:
    for relative_dir in scan_dirs:
        base = repo_root / relative_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob(f"*{PYTHON_SUFFIX}")):
            if path.name in IGNORED_FILENAMES or "__pycache__" in path.parts:
                continue
            yield path


def audit(repo_root: Path, scan_dirs: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in iter_python_files(repo_root, scan_dirs):
        text = path.read_text(encoding="utf-8", errors="replace")
        has_behavioral = BEHAVIORAL_HEADING in text
        has_static = STATIC_HEADING in text
        rows.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "behavioral_tuning": has_behavioral,
                "required_static_constants": has_static,
                "conforms": has_behavioral and has_static,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        dest="scan_dirs",
        help="Relative directory to scan. May be supplied more than once.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    scan_dirs = tuple(args.scan_dirs or DEFAULT_SCAN_DIRS)
    rows = audit(repo_root, scan_dirs)
    missing = [row for row in rows if not row["conforms"]]

    print(f"Scanned: {len(rows)} Python modules")
    print(f"Conforming: {len(rows) - len(missing)}")
    print(f"Needs review: {len(missing)}")
    for row in missing:
        missing_parts = []
        if not row["behavioral_tuning"]:
            missing_parts.append("Behavioral tuning")
        if not row["required_static_constants"]:
            missing_parts.append("Required static constants")
        print(f"- {row['path']}: missing {', '.join(missing_parts)}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
