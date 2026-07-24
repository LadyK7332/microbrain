"""Inspect normalized MicroBrain module configuration blocks without importing them."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Only these project areas are treated as behavior-bearing modules for the
# dashboard catalogue.  Add new organ families here when they become real.
CATALOG_RELATIVE_DIRS = (
    "microbrain/neurons",
    "microbrain/memory",
    "microbrain/patterns",
    "microbrain/sidecars",
)

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

BEHAVIORAL_HEADING = "# Behavioral tuning"
STATIC_HEADING = "# Required static constants"


@dataclass(frozen=True, slots=True)
class ConfigEntry:
    module: str
    category: str  # tune | law
    name: str
    value: str
    line: int


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def _render_value(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        value = ast.literal_eval(node)
        return repr(value)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expression>"


def _section_markers(lines: list[str]) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == BEHAVIORAL_HEADING:
            markers.append((index, "tune"))
        elif stripped == STATIC_HEADING:
            markers.append((index, "law"))
    return markers


def _category_for_line(line: int, markers: list[tuple[int, str]]) -> str | None:
    category: str | None = None
    for marker_line, marker_category in markers:
        if marker_line > line:
            break
        category = marker_category
    return category


def scan_file(path: Path, *, root: Path) -> list[ConfigEntry]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return []

    markers = _section_markers(source.splitlines())
    if not markers:
        return []

    entries: list[ConfigEntry] = []
    module = path.relative_to(root).as_posix()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = _assignment_name(node)
        if not name or not name.isupper():
            continue
        category = _category_for_line(getattr(node, "lineno", 0), markers)
        if category not in {"tune", "law"}:
            continue
        entries.append(
            ConfigEntry(
                module=module,
                category=category,
                name=name,
                value=_render_value(_assignment_value(node)),
                line=int(getattr(node, "lineno", 0)),
            )
        )

    entries.sort(key=lambda item: (item.module, item.line, item.name))
    return entries


def iter_module_files(root: Path) -> Iterable[Path]:
    for relative in CATALOG_RELATIVE_DIRS:
        base = root / relative
        if not base.exists():
            continue
        yield from sorted(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)


def scan_repo(root: str | Path) -> list[ConfigEntry]:
    repo_root = Path(root).resolve()
    entries: list[ConfigEntry] = []
    for path in iter_module_files(repo_root):
        entries.extend(scan_file(path, root=repo_root))
    return entries
