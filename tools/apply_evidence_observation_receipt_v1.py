from __future__ import annotations

import shutil
from pathlib import Path

PATCH_FILES = [
    "microbrain/evidence/evidence_observation_receipt.py",
    "microbrain/neurons/evidence_observation_receipt_neuron.py",
    "tests/test_evidence_observation_receipt.py",
    "tests/test_evidence_observation_receipt_neuron.py",
    "docs/evidence_observation_receipt_v1_20260823.md",
    "PATCH_MANIFEST_EVIDENCE_OBSERVATION_RECEIPT_V1.md",
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
    print("Evidence Observation Receipt v1 applied.")


if __name__ == "__main__":
    main()
