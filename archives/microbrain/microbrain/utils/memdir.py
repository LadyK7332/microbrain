from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Any


def resolve_memdir_cli(arg_memdir: str | None) -> Path:
    """Resolve memdir for CLI/config contexts.

    Priority:
      1) CLI arg
      2) MB_MEMDIR env
      3) ./memory under current working directory
    """
    if arg_memdir:
        return Path(arg_memdir)
    env_memdir = os.getenv("MB_MEMDIR")
    if env_memdir:
        return Path(env_memdir)
    return Path.cwd() / "memory"


async def resolve_memdir_ctx(ctx: Any, fallback: str | None = None) -> Path:
    """Resolve memdir from an orchestrator ctx (async).

    Priority:
      1) ctx KV 'memory:store' (if present and has .memdir)
      2) MB_MEMDIR env
      3) fallback (if provided)
      4) ./memory under current working directory
    """
    memdir: Optional[str] = None
    try:
        mem_store = await ctx.get_kv("memory:store", None)
    except Exception:
        mem_store = None

    if mem_store is not None:
        memdir = getattr(mem_store, "memdir", None)

    if memdir:
        return Path(memdir)

    env_memdir = os.getenv("MB_MEMDIR")
    if env_memdir:
        return Path(env_memdir)

    if fallback:
        return Path(fallback)

    return Path.cwd() / "memory"


def ensure_child_dirs(base: Path, child_dirs: Iterable[str], logger: Any | None = None) -> None:
    """Ensure base directory and each child directory exists."""
    try:
        base.mkdir(parents=True, exist_ok=True)
        for d in child_dirs:
            (base / d).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if logger is not None:
            try:
                logger.exception("Failed to ensure memdir layout at %s: %s", base, e)
            except Exception:
                pass
        raise
