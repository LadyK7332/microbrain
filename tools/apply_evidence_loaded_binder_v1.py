from __future__ import annotations

import shutil
from pathlib import Path

PATCH_NAME = "Evidence Loaded Binder v1"
FILES = [
    "microbrain/evidence/evidence_loaded_binder.py",
    "microbrain/neurons/evidence_loaded_binder_neuron.py",
    "microbrain/neurons/evidence_loader_neuron.py",
    "tests/test_evidence_loaded_binder.py",
    "tests/test_evidence_loaded_binder_neuron.py",
    "tests/test_evidence_loader_context.py",
    "docs/evidence_loaded_binder_v1_20260821.md",
    "PATCH_MANIFEST_EVIDENCE_LOADED_BINDER_V1.md",
]


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_file(patch_root: Path, repo_root: Path, rel: str) -> None:
    src = patch_root / rel
    dst = repo_root / rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def main() -> None:
    repo_root = _repo_root_from_script()
    for rel in FILES:
        _copy_file(repo_root, repo_root, rel)
        print(f"installed {rel}")
    print(f"{PATCH_NAME} installed.")


if __name__ == "__main__":
    main()
