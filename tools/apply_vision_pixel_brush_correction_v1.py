from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILES = [
    "microbrain/vision_pixel_ownership.py",
    "microbrain/vision_mask_correction.py",
    "microbrain/neurons/vision_pixel_ownership_neuron.py",
    "microbrain/neurons/vision_mask_correction_neuron.py",
    "tests/test_vision_pixel_ownership.py",
    "tests/test_vision_mask_correction.py",
    "tests/test_vision_mask_correction_neuron.py",
    "docs/vision_pixel_ownership_v1_20260815.md",
    "docs/vision_pixel_brush_correction_v1_20260824.md",
    "PATCH_MANIFEST_VISION_PIXEL_BRUSH_CORRECTION_V1.md",
]


def copy_file(src_root: Path, repo_root: Path, rel: str) -> None:
    src = src_root / rel
    dst = repo_root / rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Vision Pixel Ownership + Brush Correction v1 patch.")
    parser.add_argument("--repo-root", default=".", help="Repository root to patch. Default: current directory.")
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    repo_root = Path(args.repo_root).resolve()
    for rel in FILES:
        copy_file(src_root, repo_root, rel)
        print(f"copied {rel}")
    print("Vision Pixel Ownership + Brush Correction v1 applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
