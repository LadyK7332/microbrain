from __future__ import annotations

"""Apply Conversation / Recollection Review Sidecar v1.

Run from repo root:
    python tools\apply_conversation_recollection_review_v1.py

The patch is additive.  It writes new helper/neuron/test/doc files and does not
modify existing source files.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "microbrain/conversation_recollection_review.py",
    "microbrain/neurons/conversation_recollection_review_neuron.py",
    "tests/test_conversation_recollection_review.py",
    "docs/conversation_recollection_review_v1_20260818.md",
    "PATCH_MANIFEST_CONVERSATION_RECOLLECTION_REVIEW_V1.md",
]


def main() -> None:
    copied = []
    for rel in FILES:
        src = PATCH_ROOT / rel
        dst = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Patch source missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied.append(rel)

    print("Conversation / Recollection Review Sidecar v1 applied:")
    for rel in copied:
        print(f"  - {rel}")
    print("\nValidate with:")
    print(r"  python -m py_compile microbrain\conversation_recollection_review.py microbrain\neurons\conversation_recollection_review_neuron.py")
    print(r"  python -m pytest -q tests\test_conversation_recollection_review.py")


if __name__ == "__main__":
    main()
