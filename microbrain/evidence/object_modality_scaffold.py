from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

DEFAULT_OBJECT_MODALITIES = ("vision", "audio", "touch", "language", "proprio", "internal")
DEFAULT_REF_BLOCK_SIZE = 64
MAX_INLINE_SCAFFOLD_MODALITIES = 12

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

OBJECT_MODALITY_SCAFFOLD_SCHEMA = "object.modality_scaffold.v1"
OBJECT_MODALITY_INDEX_SCHEMA = "object.modality_ref_index.v1"
OBJECT_MODALITY_LEDGER_SCHEMA = "object.modality_ref_ledger.v1"
OBJECT_MODALITY_REF_SCHEMA = "object.modality_ref.v1"
OBJECT_MODALITY_SCAFFOLD_FIELD = "evidence_scaffold"
_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_.=-]+")


def _stable_digest(data: Any, *, size: int = 10) -> str:
    try:
        raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        raw = repr(data)
    return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=size).hexdigest()


def safe_ref_name(value: Any, *, prefix: str = "object") -> str:
    """Return a filesystem-safe, mostly readable ref stem.

    The readable portion helps debugging, while the digest prevents collisions
    between similarly named object IDs.
    """
    text = str(value or "").strip() or prefix
    readable = _SAFE_CHARS_RE.sub("_", text.replace(":", "_"))
    readable = readable.strip("._-")[:72] or prefix
    return f"{readable}_{_stable_digest(text, size=5)}"


def normalize_modalities(modalities: Optional[Iterable[str]] = None) -> list[str]:
    out: list[str] = []
    for raw in modalities or DEFAULT_OBJECT_MODALITIES:
        mod = str(raw or "").strip().lower()
        if not mod:
            continue
        mod = _SAFE_CHARS_RE.sub("_", mod).strip("_")
        if mod and mod not in out:
            out.append(mod)
    return out


def _atomic_write_json(path: Path, packet: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(packet), f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _append_jsonl(path: Path, packet: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(packet), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data) if isinstance(data, Mapping) else dict(default or {})
    except FileNotFoundError:
        return dict(default or {})
    except Exception:
        return dict(default or {})


class ObjectModalityScaffoldStore:
    """Preseed compact modality ledgers for durable object/memcell frames.

    This is intentionally a helper, not a running organ.  When an object is used
    enough to become a durable memory entry, callers can create a small scaffold
    packet and attach that packet to the memcell.  Future modality evidence gets
    appended to per-modality ledgers without growing the memcell's inline lists.

    Important design constraint:
      The memcell keeps stable namespaces and file refs.  Mutable counters live
      in the index files, not in the memcell row, so adding evidence does not
      force a rewrite of the durable object cell.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.scaffold_root = self.base_dir / "evidence" / "object_modality_scaffolds"

    def ensure_scaffold(
        self,
        object_id: str,
        *,
        object_kind: str = "object",
        modalities: Optional[Iterable[str]] = None,
        source: str = "",
        create_files: bool = True,
        ref_block_size: int = DEFAULT_REF_BLOCK_SIZE,
    ) -> Dict[str, Any]:
        object_id = str(object_id or "").strip()
        if not object_id:
            raise ValueError("object_id is required")
        object_key = safe_ref_name(object_id, prefix="object")
        mods = normalize_modalities(modalities)
        now = time.time()
        root_rel = f"evidence/object_modality_scaffolds/{object_key}"
        root_abs = self.scaffold_root / object_key
        root_abs.mkdir(parents=True, exist_ok=True)

        modality_packets: Dict[str, Dict[str, Any]] = {}
        for mod in mods:
            ledger_rel = f"{root_rel}/{mod}.refs.jsonl"
            index_rel = f"{root_rel}/{mod}.idx.json"
            ledger_abs = self.base_dir / ledger_rel
            index_abs = self.base_dir / index_rel
            namespace = f"{object_key}:{mod}"
            packet = {
                "modality": mod,
                "ledger_ref": ledger_rel,
                "index_ref": index_rel,
                "ref_namespace": namespace,
                "seeded_ref_range": [1, int(ref_block_size)],
                "next_ref_source": "index_file_not_memcell",
                "schema": OBJECT_MODALITY_INDEX_SCHEMA,
            }
            modality_packets[mod] = packet
            if create_files:
                ledger_abs.parent.mkdir(parents=True, exist_ok=True)
                ledger_abs.touch(exist_ok=True)
                existing = _read_json(index_abs)
                if not existing:
                    index_packet = {
                        "schema": OBJECT_MODALITY_INDEX_SCHEMA,
                        "object_id": object_id,
                        "object_key": object_key,
                        "object_kind": str(object_kind or "object"),
                        "modality": mod,
                        "ledger_ref": ledger_rel,
                        "index_ref": index_rel,
                        "ref_namespace": namespace,
                        "created_at": now,
                        "updated_at": now,
                        "entry_count": 0,
                        "next_ref_number": 1,
                        "seeded_ref_range": [1, int(ref_block_size)],
                        "source": str(source or ""),
                    }
                    _atomic_write_json(index_abs, index_packet)

        scaffold = {
            "schema": OBJECT_MODALITY_SCAFFOLD_SCHEMA,
            "object_id": object_id,
            "object_key": object_key,
            "object_kind": str(object_kind or "object"),
            "root_ref": root_rel,
            "created_at": now,
            "source": str(source or ""),
            "modalities": modality_packets,
            "policy": {
                "memcell_keeps": "stable namespaces and ledger/index refs",
                "mutable_counters_live_in": "index files",
                "raw_data_policy": "never_inline_raw_modalities",
            },
        }
        return compact_scaffold_packet(scaffold)

    def reserve_ref(
        self,
        scaffold: Mapping[str, Any],
        modality: str,
        *,
        artifact_ref: str = "",
        fossil_ref: str = "",
        summary: str = "",
        claims_supported: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
        source: str = "",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append one evidence reference to a modality ledger and update its index."""
        if not isinstance(scaffold, Mapping):
            raise ValueError("scaffold packet is required")
        mod = normalize_modalities([modality])[0]
        modalities = scaffold.get("modalities", {}) if isinstance(scaffold.get("modalities", {}), Mapping) else {}
        mod_packet = modalities.get(mod)
        if not isinstance(mod_packet, Mapping):
            raise KeyError(f"scaffold does not contain modality: {mod}")

        index_rel = str(mod_packet.get("index_ref", "") or "")
        ledger_rel = str(mod_packet.get("ledger_ref", "") or "")
        if not index_rel or not ledger_rel:
            raise ValueError(f"modality scaffold is missing ledger/index refs: {mod}")
        index_abs = self.base_dir / index_rel
        ledger_abs = self.base_dir / ledger_rel
        index = _read_json(index_abs)
        if not index:
            # Recreate a minimal index if a ledger was copied without it.
            now = time.time()
            index = {
                "schema": OBJECT_MODALITY_INDEX_SCHEMA,
                "object_id": str(scaffold.get("object_id", "") or ""),
                "object_key": str(scaffold.get("object_key", "") or ""),
                "object_kind": str(scaffold.get("object_kind", "object") or "object"),
                "modality": mod,
                "ledger_ref": ledger_rel,
                "index_ref": index_rel,
                "ref_namespace": str(mod_packet.get("ref_namespace", "") or f"{scaffold.get('object_key','object')}:{mod}"),
                "created_at": now,
                "entry_count": 0,
                "next_ref_number": 1,
                "seeded_ref_range": list(mod_packet.get("seeded_ref_range", [1, DEFAULT_REF_BLOCK_SIZE]) or [1, DEFAULT_REF_BLOCK_SIZE]),
            }

        next_num = max(1, int(index.get("next_ref_number", 1) or 1))
        namespace = str(index.get("ref_namespace", "") or mod_packet.get("ref_namespace", "") or f"{scaffold.get('object_key','object')}:{mod}")
        ref_id = f"{namespace}:{next_num:06d}"
        now = time.time()
        entry: Dict[str, Any] = {
            "schema": OBJECT_MODALITY_REF_SCHEMA,
            "ref_id": ref_id,
            "object_id": str(scaffold.get("object_id", "") or ""),
            "object_key": str(scaffold.get("object_key", "") or ""),
            "modality": mod,
            "artifact_ref": str(artifact_ref or ""),
            "fossil_ref": str(fossil_ref or ""),
            "summary": str(summary or ""),
            "claims_supported": [str(c) for c in (claims_supported or []) if str(c or "").strip()],
            "confidence": confidence,
            "source": str(source or ""),
            "created_at": now,
        }
        if extra:
            entry["extra"] = dict(extra)
        _append_jsonl(ledger_abs, entry)

        index["schema"] = OBJECT_MODALITY_INDEX_SCHEMA
        index["updated_at"] = now
        index["entry_count"] = max(0, int(index.get("entry_count", 0) or 0) + 1)
        index["next_ref_number"] = next_num + 1
        index["last_ref_id"] = ref_id
        index["last_summary"] = str(summary or "")[:240]
        if claims_supported:
            known = list(index.get("claim_hints", []) or []) if isinstance(index.get("claim_hints", []), list) else []
            for claim in claims_supported:
                c = str(claim or "").strip()
                if c and c not in known:
                    known.append(c)
            index["claim_hints"] = known[-64:]
        _atomic_write_json(index_abs, index)
        return entry

    def read_index(self, scaffold: Mapping[str, Any], modality: str) -> Dict[str, Any]:
        mod = normalize_modalities([modality])[0]
        modalities = scaffold.get("modalities", {}) if isinstance(scaffold.get("modalities", {}), Mapping) else {}
        mod_packet = modalities.get(mod)
        if not isinstance(mod_packet, Mapping):
            return {}
        index_ref = str(mod_packet.get("index_ref", "") or "")
        if not index_ref:
            return {}
        return _read_json(self.base_dir / index_ref)

    def iter_ledger_entries(self, scaffold: Mapping[str, Any], modality: str, *, limit: int = 0) -> Iterable[Dict[str, Any]]:
        mod = normalize_modalities([modality])[0]
        modalities = scaffold.get("modalities", {}) if isinstance(scaffold.get("modalities", {}), Mapping) else {}
        mod_packet = modalities.get(mod)
        if not isinstance(mod_packet, Mapping):
            return []
        ledger_ref = str(mod_packet.get("ledger_ref", "") or "")
        if not ledger_ref:
            return []
        path = self.base_dir / ledger_ref
        try:
            with path.open("r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, Mapping):
                        yield dict(row)
                        count += 1
                        if limit and count >= limit:
                            break
        except FileNotFoundError:
            return []


def compact_scaffold_packet(scaffold: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the small packet that belongs inside a memcell/frame.

    It intentionally excludes mutable counts and long ref lists.  The packet is
    a lodestone-like pointer set: modality -> ledger/index namespace.
    """
    modalities = scaffold.get("modalities", {}) if isinstance(scaffold.get("modalities", {}), Mapping) else {}
    clean_modalities: Dict[str, Dict[str, Any]] = {}
    for mod, raw in modalities.items():
        if not isinstance(raw, Mapping):
            continue
        clean_modalities[str(mod)] = {
            "schema": str(raw.get("schema", OBJECT_MODALITY_INDEX_SCHEMA) or OBJECT_MODALITY_INDEX_SCHEMA),
            "modality": str(raw.get("modality", mod) or mod),
            "ledger_ref": str(raw.get("ledger_ref", "") or ""),
            "index_ref": str(raw.get("index_ref", "") or ""),
            "ref_namespace": str(raw.get("ref_namespace", "") or ""),
            "seeded_ref_range": list(raw.get("seeded_ref_range", []) or [])[:2],
            "next_ref_source": "index_file_not_memcell",
        }
    if len(clean_modalities) > MAX_INLINE_SCAFFOLD_MODALITIES:
        # This should be rare.  If a caller really creates many modality classes,
        # preserve compactness by keeping only refs to the first group and a hint.
        items = list(clean_modalities.items())[:MAX_INLINE_SCAFFOLD_MODALITIES]
        clean_modalities = dict(items)
    return {
        "schema": OBJECT_MODALITY_SCAFFOLD_SCHEMA,
        "object_id": str(scaffold.get("object_id", "") or ""),
        "object_key": str(scaffold.get("object_key", "") or ""),
        "object_kind": str(scaffold.get("object_kind", "object") or "object"),
        "root_ref": str(scaffold.get("root_ref", "") or ""),
        "created_at": float(scaffold.get("created_at", time.time()) or time.time()),
        "source": str(scaffold.get("source", "") or ""),
        "modalities": clean_modalities,
        "policy": {
            "memcell_keeps": "stable namespaces and ledger/index refs",
            "mutable_counters_live_in": "index files",
            "raw_data_policy": "never_inline_raw_modalities",
        },
    }


def attach_scaffold_to_memcell(
    cell: Mapping[str, Any],
    scaffold: Mapping[str, Any],
    *,
    field: str = OBJECT_MODALITY_SCAFFOLD_FIELD,
) -> Dict[str, Any]:
    """Return a copy of a memcell with a compact modality scaffold attached."""
    if not isinstance(cell, Mapping):
        raise ValueError("cell must be a mapping")
    out = dict(cell)
    out[field] = compact_scaffold_packet(scaffold)
    meta = dict(out.get("meta", {}) or {}) if isinstance(out.get("meta", {}), Mapping) else {}
    meta["evidence_scaffolded"] = True
    meta["evidence_scaffold_schema"] = OBJECT_MODALITY_SCAFFOLD_SCHEMA
    out["meta"] = meta
    classifiers = list(out.get("classifiers", []) or []) if isinstance(out.get("classifiers", []), list) else []
    if "evidence_scaffolded" not in classifiers:
        classifiers.append("evidence_scaffolded")
    out["classifiers"] = classifiers
    return out
