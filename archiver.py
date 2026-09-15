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
    ".venv_broken",
    ".vs",
    ".vscode",
    "env",
    "archives",
    "__pycache__",
    "models",          # adjust/remove if your models dir is named differently
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# ------------------------------------------------------------------------


def is_excluded(path: Path) -> bool:
    """Return True if any part of the path is in EXCLUDE_DIRS."""
    return any(part in EXCLUDE_DIRS or part.startswith(".venv_broken_") for part in path.parts)


def find_python_files(root: Path) -> list[Path]:
    """Find all .py files under root, excluding EXCLUDE_DIRS."""
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if not is_excluded(p):
            files.append(p)
    return files


def git_add_files_batched(rel_paths: list[str], batch_size: int = 100) -> None:
    """Fallback for older Git versions that cannot read pathspecs from stdin."""
    for start in range(0, len(rel_paths), batch_size):
        batch = rel_paths[start : start + batch_size]
        subprocess.run(
            ["git", "add", "--", *batch],
            cwd=REPO_ROOT,
            check=True,
        )


def filter_git_ignored(rel_paths: list[str]) -> list[str]:
    """Drop paths Git already ignores so `git add` does not fail on them."""
    if not rel_paths:
        return rel_paths

    proc = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        cwd=REPO_ROOT,
        input="\0".join(rel_paths) + "\0",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        print("[WARN] git check-ignore failed; continuing without ignored-path filtering.")
        return rel_paths

    ignored = {p for p in proc.stdout.split("\0") if p}
    if ignored:
        print(f"Skipping {len(ignored)} git-ignored .py file(s).")
    return [p for p in rel_paths if p not in ignored]


def git_add_files(files: list[Path]) -> None:
    """Run `git add` on the given files (relative to repo root)."""
    if not files:
        print("No .py files to add to git.")
        return

    rel_paths = [str(p.relative_to(REPO_ROOT)) for p in files]
    rel_paths = filter_git_ignored(rel_paths)
    if not rel_paths:
        print("No non-ignored .py files to add to git.")
        return

    print(f"Adding {len(rel_paths)} .py files to git…")

    try:
        subprocess.run(
            ["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"],
            cwd=REPO_ROOT,
            input="\0".join(rel_paths) + "\0",
            text=True,
            check=True,
        )
        print("git add completed.")
    except subprocess.CalledProcessError as e:
        print(
            "[WARN] git add via stdin failed "
            f"(exit code {e.returncode}); retrying in batches."
        )
        git_add_files_batched(rel_paths)
        print("git add completed.")


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
