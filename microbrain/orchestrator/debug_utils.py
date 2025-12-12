from __future__ import annotations

from typing import Any

# microbrain/orchestrator/debug_utils.py

_DEBUG_ENABLED: bool = False

def set_debug_enabled(enabled: bool) -> None:
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(enabled)

def is_debug_enabled() -> bool:
    return _DEBUG_ENABLED
