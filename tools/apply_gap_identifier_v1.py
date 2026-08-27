from __future__ import annotations

import shutil
from pathlib import Path

PATCH_FILES = [
    "microbrain/cognition/__init__.py",
    "microbrain/cognition/gap_identifier.py",
    "microbrain/neurons/gap_identifier_neuron.py",
    "tests/test_gap_identifier.py",
    "tests/test_gap_identifier_neuron.py",
    "docs/gap_identifier_v1_20260824.md",
    "PATCH_MANIFEST_GAP_IDENTIFIER_V1.md",
]


def main() -> None:
    patch_root = Path(__file__).resolve().parents[1]
    repo_root = Path.cwd()
    for rel in PATCH_FILES:
        src = patch_root / rel
        dst = repo_root / rel
        if not src.exists():
            raise FileNotFoundError(f"patch file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {rel}")
    print("Gap Identifier v1 applied.")


if __name__ == "__main__":
    main()
