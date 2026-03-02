from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


UNIT_CONFIG_FILENAME = "unit_config.json"


def default_unit_config() -> Dict[str, Any]:
    # Defaults for a fresh "blank" unit.
    # Keep these conservative; you can override once and persist.
    return {
        "tts_out": {
            "enabled": True,
            "voice": "Zira",     # substring match; safe default if installed
            "rate": 155,
            "volume": 0.9,
        },
        "pdna": {
            "profile_name": "microbrain_default",
        },
    }


def load_or_create_unit_config(memdir: Path) -> Dict[str, Any]:
    memdir = Path(memdir)
    path = memdir / UNIT_CONFIG_FILENAME

    if not path.exists():
        cfg = default_unit_config()
        save_unit_config(memdir, cfg)
        return cfg

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cfg = json.loads(raw) if raw.strip() else {}
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception:
        cfg = {}

    # merge in any missing defaults (non-destructive)
    merged = default_unit_config()
    _deep_update_missing(merged, cfg)
    if merged != cfg:
        save_unit_config(memdir, merged)
    return merged


def save_unit_config(memdir: Path, cfg: Dict[str, Any]) -> None:
    memdir = Path(memdir)
    memdir.mkdir(parents=True, exist_ok=True)
    path = memdir / UNIT_CONFIG_FILENAME
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_update_missing(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            continue
        if isinstance(dst.get(k), dict) and isinstance(v, dict):
            _deep_update_missing(dst[k], v)


def get_tts_out(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("tts_out")
    return t if isinstance(t, dict) else {}


def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """
    Sets cfg["a"]["b"]["c"] by dotted path "a.b.c".
    """
    cur = cfg
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value
