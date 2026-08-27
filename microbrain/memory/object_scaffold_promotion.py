from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from microbrain.evidence.object_modality_scaffold import (
    OBJECT_MODALITY_SCAFFOLD_FIELD,
    OBJECT_MODALITY_SCAFFOLD_SCHEMA,
    ObjectModalityScaffoldStore,
    attach_scaffold_to_memcell,
    normalize_modalities,
)

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Scaffolds are for durable-ish object memories, not every hot token or one-off
# utterance.  These gates intentionally mirror the memory lifecycle idea:
# when an object has enough use/promotion to be worth keeping, preseed its
# modality ledgers before evidence refs start piling up inline.
SCAFFOLD_DURABLE_TIERS = ("short", "long", "learned")
SCAFFOLD_NOW_MIN_ENCOUNTERS = 3
SCAFFOLD_NOW_MIN_PROMOTION = 0.36
SCAFFOLD_DEFAULT_SOURCE = "memcell_scaffold_promotion"

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

OBJECT_ID_PREFIXES = (
    "object:",
    "person:",
    "place:",
    "entity:",
    "visual:",
    "auditory:",
    "vobj:",
    "touch:",
)

OBJECT_KIND_HINTS = (
    "object",
    "person",
    "place",
    "entity",
    "visual",
    "auditory",
    "tactile",
    "touch",
    "material",
    "tool",
)

NON_OBJECT_KIND_HINTS = (
    "utterance",
    "token",
    "word",
    "pattern",
    "phrase",
    "clause",
    "template",
    "linker",
    "learning_frame",
    "language",
    "conversation_turn",
    "trainer_correction",
)

MODALITY_ALIASES = {
    "visual": "vision",
    "vision_delta": "vision",
    "visual_delta": "vision",
    "image": "vision",
    "auditory": "audio",
    "sound": "audio",
    "voice": "audio",
    "tactile": "touch",
    "haptic": "touch",
    "text": "language",
    "linguistic": "language",
}


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _norm_modality(value: Any) -> str:
    raw = _lower_text(value)
    raw = MODALITY_ALIASES.get(raw, raw)
    mods = normalize_modalities([raw]) if raw else []
    return mods[0] if mods else ""


def has_object_modality_scaffold(cell: Mapping[str, Any]) -> bool:
    scaffold = cell.get(OBJECT_MODALITY_SCAFFOLD_FIELD)
    return isinstance(scaffold, Mapping) and str(scaffold.get("schema", "") or "") == OBJECT_MODALITY_SCAFFOLD_SCHEMA


def extract_modalities_for_object(cell: Mapping[str, Any]) -> list[str]:
    """Extract compact modality names already hinted by a memcell.

    The returned list is a shelf request, not proof that each modality has data.
    Empty/missing signals are handled by the scaffold store defaults.
    """
    found: list[str] = []

    def add(raw: Any) -> None:
        mod = _norm_modality(raw)
        if mod and mod not in found:
            found.append(mod)

    # Common compact fields.
    for raw in _listish(cell.get("modalities_present")):
        add(raw)
    for raw in _listish(cell.get("modalities")):
        if isinstance(raw, Mapping):
            continue
        add(raw)

    modalities = cell.get("modalities")
    if isinstance(modalities, Mapping):
        for key, value in modalities.items():
            if value not in (None, {}, [], ""):
                add(key)

    # Older sensory tags / senses containers.
    senses_present = cell.get("senses_present")
    if isinstance(senses_present, Mapping):
        for key, present in senses_present.items():
            if bool(present):
                add(key)

    for field in ("sense_tags", "senses"):
        value = cell.get(field)
        if isinstance(value, Mapping):
            for key, payload in value.items():
                if payload not in (None, {}, [], ""):
                    add(key)

    # Classifier hints such as sensor.camera_0 or visual_proto_object.
    for cls in _listish(cell.get("classifiers")):
        text = _lower_text(cls)
        if text.startswith("sensor."):
            add(text.split(".", 1)[1])
        elif "visual" in text or "image" in text:
            add("vision")
        elif "audio" in text or "voice" in text or "sound" in text:
            add("audio")
        elif "touch" in text or "tactile" in text or "haptic" in text:
            add("touch")

    if str(cell.get("text", "") or "").strip():
        add("language")

    return found


def looks_like_object_memcell(cell: Mapping[str, Any]) -> bool:
    """Return True for cells that deserve object modality shelves."""
    if not isinstance(cell, Mapping):
        return False

    cell_id = _lower_text(cell.get("id"))
    kind = _lower_text(cell.get("kind"))
    schema = _lower_text(cell.get("schema"))
    meta = cell.get("meta", {}) if isinstance(cell.get("meta", {}), Mapping) else {}
    meta_kind = _lower_text(meta.get("kind"))
    classifiers = [_lower_text(c) for c in _listish(cell.get("classifiers"))]

    joined = " ".join(x for x in [cell_id, kind, schema, meta_kind, *classifiers] if x)
    if not joined:
        return False

    if any(hint in joined for hint in NON_OBJECT_KIND_HINTS):
        # Avoid giving every utterance/token/pattern its own evidence shelves.
        # Those are language structures, not durable object identity cards.
        return False

    if any(cell_id.startswith(prefix) for prefix in OBJECT_ID_PREFIXES):
        return True
    if any(hint in joined for hint in OBJECT_KIND_HINTS):
        return True
    if extract_modalities_for_object(cell) and any(k in joined for k in ("memory_frame", "base.object", "scene.object")):
        return True
    return False


def tier_allows_scaffold(cell: Mapping[str, Any], tier: str) -> bool:
    tier = _lower_text(tier or cell.get("tier") or "now")
    if tier in SCAFFOLD_DURABLE_TIERS:
        return True
    if tier != "now":
        return False
    meta = cell.get("meta", {}) if isinstance(cell.get("meta", {}), Mapping) else {}
    if bool(meta.get("promoted_from_hot")):
        return True
    try:
        encounters = int(cell.get("encounter_count", 0) or 0)
    except Exception:
        encounters = 0
    try:
        promotion = float(cell.get("promotion", 0.0) or 0.0)
    except Exception:
        promotion = 0.0
    return encounters >= SCAFFOLD_NOW_MIN_ENCOUNTERS or promotion >= SCAFFOLD_NOW_MIN_PROMOTION


def should_attach_object_modality_scaffold(cell: Mapping[str, Any], *, tier: str = "now") -> bool:
    if not isinstance(cell, Mapping):
        return False
    if has_object_modality_scaffold(cell):
        return False
    if not tier_allows_scaffold(cell, tier):
        return False
    return looks_like_object_memcell(cell)


def maybe_attach_object_modality_scaffold(
    cell: Mapping[str, Any],
    *,
    base_dir: str | Path,
    tier: str = "now",
    source: str = SCAFFOLD_DEFAULT_SOURCE,
    create_files: bool = True,
    modalities: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return a memcell with a preseeded object modality scaffold if eligible.

    This is deliberately side-effect-light for normal cells.  It creates files
    only for durable-ish object cells and keeps mutable ref counters in the
    scaffold index files, not the memcell row.
    """
    row = dict(cell or {})
    if not should_attach_object_modality_scaffold(row, tier=tier):
        return row

    object_id = str(row.get("id", "") or "").strip()
    if not object_id:
        return row
    kind = str(row.get("kind", row.get("meta", {}).get("kind", "object") if isinstance(row.get("meta", {}), Mapping) else "object") or "object")
    mods = list(modalities or extract_modalities_for_object(row) or []) or None
    store = ObjectModalityScaffoldStore(base_dir)
    scaffold = store.ensure_scaffold(
        object_id,
        object_kind=kind,
        modalities=mods,
        source=str(source or SCAFFOLD_DEFAULT_SOURCE),
        create_files=create_files,
    )
    out = attach_scaffold_to_memcell(row, scaffold)
    meta = dict(out.get("meta", {}) or {}) if isinstance(out.get("meta", {}), Mapping) else {}
    meta["evidence_scaffold_tier"] = _lower_text(tier or row.get("tier") or "")
    meta["evidence_scaffold_source"] = str(source or SCAFFOLD_DEFAULT_SOURCE)
    out["meta"] = meta
    return out
