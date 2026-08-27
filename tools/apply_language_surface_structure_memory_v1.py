from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "microbrain/language/surface_structure_memory.py",
    "microbrain/neurons/language_surface_structure_neuron.py",
    "docs/language_surface_structure_memory_v1_20260827.md",
    "tests/test_language_surface_structure_memory.py",
    "tests/test_language_surface_structure_neuron.py",
    "PATCH_MANIFEST_LANGUAGE_SURFACE_STRUCTURE_MEMORY_V1.md",
]


def main() -> None:
    # The zip is intended to be expanded over the repository root.  This tool is
    # mostly a sanity helper; it leaves files in place and verifies they exist.
    missing = []
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
    if missing:
        raise SystemExit("Language Surface Structure Memory v1 appears incompletely expanded. Missing:\n" + "\n".join(missing))
    print("Language Surface Structure Memory v1 files are present.")
    print("Next: run py_compile and pytest commands from the manifest.")


if __name__ == "__main__":
    main()
