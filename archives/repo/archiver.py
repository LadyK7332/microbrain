#!/usr/bin/env python
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

# --- CONFIG -------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

# Any path segment matching one of these will be skipped completely
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "env",
    "archives",
    "__pycache__",
    "models",          # adjust/remove if your models dir is named differently
    ".mypy_cache",
    ".pytest_cache",
}

# ------------------------------------------------------------------------


def is_excluded(path: Path) -> bool:
    """Return True if any part of the path is in EXCLUDE_DIRS."""
    return any(part in EXCLUDE_DIRS for part in path.parts)


def find_python_files(root: Path) -> list[Path]:
    """Find all .py files under root, excluding EXCLUDE_DIRS."""
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if not is_excluded(p):
            files.append(p)
    return files


def git_add_files(files: list[Path]) -> None:
    """Run `git add` on the given files (relative to repo root)."""
    if not files:
        print("No .py files to add to git.")
        return

    rel_paths = [str(p.relative_to(REPO_ROOT)) for p in files]
    print(f"Adding {len(rel_paths)} .py files to git…")

    # If there are a LOT of files, we might want to batch, but for most repos
    # this single call is fine.
    try:
        subprocess.run(
            ["git", "add", *rel_paths],
            cwd=REPO_ROOT,
            check=True,
        )
        print("git add completed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] git add failed with exit code {e.returncode}")


def build_zip(root: Path) -> Path:
    """Create a timestamped zip of the repo (excluding EXCLUDE_DIRS)."""
    archives_dir = root / "archives"
    archives_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_name = f"microbrain-src-{ts}.zip"
    zip_path = archives_dir / zip_name

    print(f"Creating archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_dir():
                # Skip any directory whose name is in EXCLUDE_DIRS
                if path.name in EXCLUDE_DIRS:
                    # Skip entire subtree
                    continue
                # We only write files, not directories
                continue

            if is_excluded(path):
                continue

            arcname = path.relative_to(root)
            zf.write(path, arcname.as_posix())

    print("Archive complete.")
    return zip_path


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")

    # 1) Find and add .py files to git (excluding envs, etc.)
    py_files = find_python_files(REPO_ROOT)
    print(f"Found {len(py_files)} .py files (after exclusions).")
    git_add_files(py_files)

    # 2) Build zip archive
    build_zip(REPO_ROOT)


if __name__ == "__main__":
    main()
