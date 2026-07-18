from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PDNA_RUNTIME_KV_KEYS = (
    "pdna:affect_model",
    "pdna:reinforcement_model",
    "pdna:drive_thresholds",
    "pdna:ddna_mutators",
    "pdna:wans",
    "drive:ddna_modulators",
)


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def profile_extra(pdna: Any) -> Mapping[str, Any]:
    """Return forwards-compatible top-level profile sections.

    PDNAProfile v1 only had fixed dataclass fields. v2+ may carry DDNA,
    affect_model, reinforcement_model, WANS, and other profile organs. This
    helper lets runtime organs read those sections without caring whether the
    profile is a dataclass instance or a raw dict snapshot.
    """
    if isinstance(pdna, Mapping):
        return pdna
    extra = getattr(pdna, "extra", None)
    return extra if isinstance(extra, Mapping) else {}


def get_profile_section(pdna: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(pdna, Mapping):
        return pdna.get(name, default)
    if hasattr(pdna, name):
        value = getattr(pdna, name)
        if value is not None:
            return value
    extra = profile_extra(pdna)
    return extra.get(name, default)


def get_path(data: Any, path: str, default: Any | None = None) -> Any:
    cur = data
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def profile_path(pdna: Any, section: str, path: str, default: Any | None = None) -> Any:
    return get_path(get_profile_section(pdna, section, {}), path, default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def trait_value(pdna: Any, trait: str, default: float = 0.5) -> float:
    if isinstance(pdna, Mapping):
        return clamp(safe_float(pdna.get(trait), default))
    return clamp(safe_float(getattr(pdna, trait, default), default))


def ddna_trait_mutator(pdna: Any, trait: str, key: str, default: float = 1.0) -> float:
    mutators = get_profile_section(pdna, "ddna_mutators", {})
    if not isinstance(mutators, Mapping):
        return float(default)
    raw = mutators.get(trait, {})
    if not isinstance(raw, Mapping):
        return float(default)
    return safe_float(raw.get(key), default)


def profile_sections_for_kv(pdna: Any) -> dict[str, Any]:
    sections = {}
    for name in (
        "profile_schema_version",
        "profile_kind",
        "ddna",
        "affect_model",
        "reinforcement_model",
        "drive_thresholds",
        "ddna_mutators",
        "wans",
        "learning_policy",
        "safety_homeostasis_spine",
        "compatibility",
    ):
        value = get_profile_section(pdna, name, None)
        if value is not None:
            sections[name] = value
    return sections


async def publish_profile_sections(ctx: Any, pdna: Any) -> None:
    sections = profile_sections_for_kv(pdna)
    await ctx.set_kv("pdna:sections", sections)
    for key, value in sections.items():
        await ctx.set_kv(f"pdna:{key}", value)
