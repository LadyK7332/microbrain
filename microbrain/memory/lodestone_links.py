"""Append-only lodestone link ledgers for dense memcell link buckets.

Memcells should remain small index cards.  If a cell becomes a hub, its link
bucket spills into an append-only ledger and the cell keeps one lodestone
pointer instead of a giant inline edge list.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from microbrain.evidence.evidence_card import clean_ref, clean_token, safe_json, stable_digest

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

LODESTONE_ROOT_NAME = "mem_cell_links"
DEFAULT_LINK_BUCKET = "links_explicit"
DEFAULT_LINK_TYPE = "related"
MAX_INLINE_MEMCELL_LINKS = 48
INLINE_LODESTONE_SAMPLE_SIZE = 8
LODESTONE_QUERY_SAMPLE_LIMIT = 24
LODESTONE_WALK_LIMIT = 24
LODESTONE_SCATTER_LIMIT = 8
LINK_LEDGER_FSYNC = True

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

MEMCELL_LINK_PACK_SCHEMA = "memcell.link_pack.v1"
MEMCELL_LINK_LODESTONE_SCHEMA = "memcell.link_lodestone.v1"
MEMCELL_LINK_LEDGER_ENTRY_SCHEMA = "memcell.link_ledger_entry.v1"
MEMCELL_LINK_LEDGER_INDEX_SCHEMA = "memcell.link_ledger_index.v1"
MEMCELL_LINK_RETRIEVAL_SCHEMA = "memcell.link_retrieval.v1"
RETRIEVAL_DIRECTED = "directed"
RETRIEVAL_WALK = "walk"
RETRIEVAL_SCATTER = "scatter"
RETRIEVAL_MODES = (RETRIEVAL_DIRECTED, RETRIEVAL_WALK, RETRIEVAL_SCATTER)


def resolve_inline_link_limit(value: Any = None, *, default: int = MAX_INLINE_MEMCELL_LINKS) -> int:
    """Resolve the memcell inline-link limit from an argument or environment."""

    if value is None:
        value = os.getenv("MB_MEMCELL_LINK_INLINE_MAX", "")
    try:
        if value not in (None, ""):
            return max(0, int(value))
    except Exception:
        pass
    return max(0, int(default))


def normalize_link_entry(
    value: Any,
    *,
    bucket: str = DEFAULT_LINK_BUCKET,
    link_type: str = DEFAULT_LINK_TYPE,
    source: str = "",
    reason: str = "",
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Return a compact link entry that can live inline or in a ledger."""

    stamp = float(timestamp or time.time())
    if isinstance(value, Mapping):
        ref = clean_ref(
            value.get("ref")
            or value.get("cell_id")
            or value.get("target")
            or value.get("id")
            or value.get("artifact_ref")
            or value.get("path")
            or ""
        )
        entry = {
            "ref": ref,
            "bucket": clean_token(value.get("bucket") or bucket, fallback=DEFAULT_LINK_BUCKET),
            "link_type": clean_token(value.get("link_type") or value.get("kind") or link_type, fallback=DEFAULT_LINK_TYPE),
            "weight": _float_or(value.get("weight", value.get("score", 0.0)), 0.0),
            "confidence": _float_or(value.get("confidence", value.get("trust", 0.0)), 0.0),
            "source": str(value.get("source", source) or ""),
            "reason": str(value.get("reason", reason) or ""),
            "ts": _float_or(value.get("ts", value.get("timestamp", stamp)), stamp),
            "meta": safe_json(value.get("meta", {})) if isinstance(value.get("meta", {}), Mapping) else {},
        }
        for key in ("role", "slot", "modality", "claim", "evidence_ref"):
            if value.get(key) not in (None, "", [], {}):
                entry[key] = safe_json(value.get(key))
    else:
        ref = clean_ref(value)
        entry = {
            "ref": ref,
            "bucket": clean_token(bucket, fallback=DEFAULT_LINK_BUCKET),
            "link_type": clean_token(link_type, fallback=DEFAULT_LINK_TYPE),
            "weight": 0.0,
            "confidence": 0.0,
            "source": str(source or ""),
            "reason": str(reason or ""),
            "ts": stamp,
            "meta": {},
        }

    if not entry.get("ref"):
        return {}
    entry["weight"] = _clamp_signed(entry.get("weight", 0.0))
    entry["confidence"] = _clamp01(entry.get("confidence", 0.0))
    return {k: v for k, v in entry.items() if v not in (None, "", [], {})}


def normalize_link_entries(
    values: Iterable[Any] | None,
    *,
    bucket: str = DEFAULT_LINK_BUCKET,
    link_type: str = DEFAULT_LINK_TYPE,
    source: str = "",
    reason: str = "",
    timestamp: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values or []:
        entry = normalize_link_entry(
            value,
            bucket=bucket,
            link_type=link_type,
            source=source,
            reason=reason,
            timestamp=timestamp,
        )
        if not entry:
            continue
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


class MemCellLinkLodestoneStore:
    """Store overflow links for dense memcells as append-only ledgers."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / LODESTONE_ROOT_NAME
        self.root.mkdir(parents=True, exist_ok=True)

    def cell_dir(self, cell_id: str) -> Path:
        safe_cell = clean_token(cell_id, fallback="cell")[:96]
        path = self.root / safe_cell
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ledger_path(self, cell_id: str, bucket: str = DEFAULT_LINK_BUCKET) -> Path:
        safe_bucket = clean_token(bucket, fallback=DEFAULT_LINK_BUCKET)[:64]
        return self.cell_dir(cell_id) / f"{safe_bucket}.jsonl"

    def index_path(self, cell_id: str, bucket: str = DEFAULT_LINK_BUCKET) -> Path:
        safe_bucket = clean_token(bucket, fallback=DEFAULT_LINK_BUCKET)[:64]
        return self.cell_dir(cell_id) / f"{safe_bucket}.idx.json"

    def to_ref(self, path: str | Path) -> str:
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.base_dir.resolve()).as_posix()
        except Exception:
            return clean_ref(candidate.as_posix())

    def resolve_ref(self, ref: str | Path) -> Path:
        clean = clean_ref(ref)
        path = Path(clean)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def append_links(
        self,
        *,
        cell_id: str,
        bucket: str = DEFAULT_LINK_BUCKET,
        links: Iterable[Any],
        link_type: str = DEFAULT_LINK_TYPE,
        source: str = "",
        reason: str = "",
        timestamp: float | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Append links to the ledger and return the written ledger entries."""

        clean_cell = clean_ref(cell_id)
        clean_bucket = clean_token(bucket, fallback=DEFAULT_LINK_BUCKET)
        stamp = float(timestamp or time.time())
        entries = normalize_link_entries(
            links,
            bucket=clean_bucket,
            link_type=link_type,
            source=source,
            reason=reason,
            timestamp=stamp,
        )
        if not clean_cell or not entries:
            return []

        ledger = self.ledger_path(clean_cell, clean_bucket)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        meta_payload = safe_json(dict(meta or {}))
        written: list[dict[str, Any]] = []
        with ledger.open("a", encoding="utf-8") as handle:
            for entry in entries:
                row = {
                    "schema": MEMCELL_LINK_LEDGER_ENTRY_SCHEMA,
                    "cell_id": clean_cell,
                    "bucket": clean_bucket,
                    "entry_id": stable_digest({"cell_id": clean_cell, "bucket": clean_bucket, "entry": entry, "ts": stamp})[:20],
                    "queued_at": stamp,
                    "entry": entry,
                }
                if meta_payload:
                    row["meta"] = meta_payload
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written.append(row)
            handle.flush()
            if LINK_LEDGER_FSYNC:
                os.fsync(handle.fileno())
        self.write_index(
            cell_id=clean_cell,
            bucket=clean_bucket,
            degree_estimate=self.estimate_degree(clean_cell, clean_bucket),
            inline_sample=[row.get("entry", {}) for row in written[:INLINE_LODESTONE_SAMPLE_SIZE]],
            source=source,
            reason=reason,
            timestamp=stamp,
        )
        return written

    def write_index(
        self,
        *,
        cell_id: str,
        bucket: str = DEFAULT_LINK_BUCKET,
        degree_estimate: int = 0,
        inline_sample: Sequence[Mapping[str, Any]] | None = None,
        source: str = "",
        reason: str = "",
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Write a small index card; the full edge crowd stays in the ledger."""

        stamp = float(timestamp or time.time())
        clean_cell = clean_ref(cell_id)
        clean_bucket = clean_token(bucket, fallback=DEFAULT_LINK_BUCKET)
        ledger = self.ledger_path(clean_cell, clean_bucket)
        index = self.index_path(clean_cell, clean_bucket)
        sample = [dict(item) for item in list(inline_sample or [])[:INLINE_LODESTONE_SAMPLE_SIZE] if isinstance(item, Mapping)]
        payload = {
            "schema": MEMCELL_LINK_LEDGER_INDEX_SCHEMA,
            "cell_id": clean_cell,
            "bucket": clean_bucket,
            "overflowed": True,
            "hub": True,
            "degree_estimate": max(0, int(degree_estimate)),
            "ledger_ref": self.to_ref(ledger),
            "index_ref": self.to_ref(index),
            "inline_sample": sample,
            "retrieval_modes": list(RETRIEVAL_MODES),
            "query_weight_hint": "broad_traversal_not_specific_answer",
            "updated_at": stamp,
            "source": str(source or ""),
            "reason": str(reason or ""),
        }
        if not index.exists():
            payload["created_at"] = stamp
        else:
            old = self.read_index(clean_cell, clean_bucket)
            payload["created_at"] = _float_or(old.get("created_at"), stamp) if isinstance(old, Mapping) else stamp
        self._write_json(index, payload)
        return payload

    def read_index(self, cell_id: str, bucket: str = DEFAULT_LINK_BUCKET) -> dict[str, Any]:
        path = self.index_path(cell_id, bucket)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def build_pointer(
        self,
        *,
        cell_id: str,
        bucket: str = DEFAULT_LINK_BUCKET,
        degree_estimate: int = 0,
        inline_sample: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        clean_cell = clean_ref(cell_id)
        clean_bucket = clean_token(bucket, fallback=DEFAULT_LINK_BUCKET)
        index = self.index_path(clean_cell, clean_bucket)
        ledger = self.ledger_path(clean_cell, clean_bucket)
        sample = [dict(item) for item in list(inline_sample or [])[:INLINE_LODESTONE_SAMPLE_SIZE] if isinstance(item, Mapping)]
        return {
            "schema": MEMCELL_LINK_LODESTONE_SCHEMA,
            "overflowed": True,
            "hub": True,
            "cell_id": clean_cell,
            "bucket": clean_bucket,
            "degree_estimate": max(0, int(degree_estimate)),
            "inline_sample": sample,
            "ledger_ref": self.to_ref(ledger),
            "index_ref": self.to_ref(index),
            "retrieval_modes": list(RETRIEVAL_MODES),
            "query_weight_hint": "broad_traversal_not_specific_answer",
        }

    def pack_links(
        self,
        *,
        cell_id: str,
        bucket: str = DEFAULT_LINK_BUCKET,
        links: Iterable[Any],
        max_inline_links: int | None = None,
        link_type: str = DEFAULT_LINK_TYPE,
        source: str = "",
        reason: str = "",
        timestamp: float | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keep small link buckets inline; spill dense buckets to a lodestone."""

        limit = resolve_inline_link_limit(max_inline_links)
        entries = normalize_link_entries(
            links,
            bucket=bucket,
            link_type=link_type,
            source=source,
            reason=reason,
            timestamp=timestamp,
        )
        pack: dict[str, Any] = {
            "schema": MEMCELL_LINK_PACK_SCHEMA,
            "cell_id": clean_ref(cell_id),
            "bucket": clean_token(bucket, fallback=DEFAULT_LINK_BUCKET),
            "count": len(entries),
            "max_inline_links": limit,
            "links": [],
            "lodestone": {},
        }
        if len(entries) <= limit:
            pack["links"] = entries
            return pack

        written = self.append_links(
            cell_id=pack["cell_id"],
            bucket=pack["bucket"],
            links=entries,
            link_type=link_type,
            source=source,
            reason=reason,
            timestamp=timestamp,
            meta=meta,
        )
        sample = [row.get("entry", {}) for row in written[:INLINE_LODESTONE_SAMPLE_SIZE] if isinstance(row, Mapping)]
        pack["lodestone"] = self.build_pointer(
            cell_id=pack["cell_id"],
            bucket=pack["bucket"],
            degree_estimate=self.estimate_degree(pack["cell_id"], pack["bucket"]),
            inline_sample=sample,
        )
        return pack

    def shape_link_field(self, **kwargs: Any) -> list[dict[str, Any]] | dict[str, Any]:
        """Return exactly what a memcell link bucket should store.

        Small buckets return a plain list to stay compatible with existing
        memcells. Dense buckets return a lodestone pointer dict.
        """

        pack = self.pack_links(**kwargs)
        if pack.get("lodestone"):
            return dict(pack["lodestone"])
        return list(pack.get("links", []) or [])

    def read_ledger_entries(self, cell_id: str, bucket: str = DEFAULT_LINK_BUCKET, *, limit: int | None = None) -> list[dict[str, Any]]:
        path = self.ledger_path(cell_id, bucket)
        out: list[dict[str, Any]] = []
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= max(0, int(limit)):
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if isinstance(row, Mapping):
                    out.append(dict(row))
        return out

    def estimate_degree(self, cell_id: str, bucket: str = DEFAULT_LINK_BUCKET) -> int:
        path = self.ledger_path(cell_id, bucket)
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in handle:
                count += 1
        return count

    def retrieve_links(
        self,
        *,
        cell_id: str,
        bucket: str = DEFAULT_LINK_BUCKET,
        mode: str = RETRIEVAL_DIRECTED,
        query: str | Iterable[Any] = "",
        limit: int | None = None,
        seed: int | str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retrieve overflow links by directed search, walk, or scatter."""

        clean_mode = clean_token(mode, fallback=RETRIEVAL_DIRECTED)
        if clean_mode not in RETRIEVAL_MODES:
            clean_mode = RETRIEVAL_DIRECTED
        default_limit = {
            RETRIEVAL_DIRECTED: LODESTONE_QUERY_SAMPLE_LIMIT,
            RETRIEVAL_WALK: LODESTONE_WALK_LIMIT,
            RETRIEVAL_SCATTER: LODESTONE_SCATTER_LIMIT,
        }.get(clean_mode, LODESTONE_QUERY_SAMPLE_LIMIT)
        max_results = max(0, int(limit if limit is not None else default_limit))
        rows = self.read_ledger_entries(cell_id, bucket)
        entries = [dict(row.get("entry", {}) or {}) for row in rows if isinstance(row, Mapping) and isinstance(row.get("entry", {}), Mapping)]
        if clean_mode == RETRIEVAL_DIRECTED:
            query_tokens = _tokens(query)
            ranked = sorted(entries, key=lambda item: _directed_score(item, query_tokens), reverse=True)
            selected = [item for item in ranked if _directed_score(item, query_tokens) > 0.0]
            if not selected:
                selected = ranked
        elif clean_mode == RETRIEVAL_WALK:
            start = max(0, int(offset))
            ordered = sorted(entries, key=lambda item: (_float_or(item.get("confidence"), 0.0), _float_or(item.get("weight"), 0.0), _float_or(item.get("ts"), 0.0)), reverse=True)
            selected = ordered[start:]
        else:
            rng = random.Random(seed if seed is not None else time.time_ns())
            selected = list(entries)
            rng.shuffle(selected)

        selected = selected[:max_results] if max_results else []
        return {
            "schema": MEMCELL_LINK_RETRIEVAL_SCHEMA,
            "cell_id": clean_ref(cell_id),
            "bucket": clean_token(bucket, fallback=DEFAULT_LINK_BUCKET),
            "mode": clean_mode,
            "query": " ".join(_tokens(query)),
            "count": len(selected),
            "available_count": len(entries),
            "results": selected,
        }

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        data = json.dumps(safe_json(dict(payload)), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            if LINK_LEDGER_FSYNC:
                os.fsync(handle.fileno())
        tmp.replace(path)


def select_lodestone_retrieval_mode(state: Mapping[str, Any] | None = None) -> str:
    """Map existing drive/hypothesis state to a lodestone retrieval mode."""

    s = dict(state or {})
    if bool(s.get("safety_uncertainty") or s.get("crisis_mode") or s.get("direct_question")):
        return RETRIEVAL_DIRECTED
    try:
        if float(s.get("hypothesis_response_demand", s.get("response_demand", 0.0)) or 0.0) >= 0.65:
            return RETRIEVAL_DIRECTED
    except Exception:
        pass
    if bool(s.get("trainer_correction") or s.get("contradiction") or s.get("verify_hypothesis") or s.get("hypothesis_verification")):
        return RETRIEVAL_WALK
    try:
        boredom = float(s.get("boredom", 0.0) or 0.0)
        curiosity = float(s.get("curiosity", s.get("inquiry", 0.0)) or 0.0)
        uncertainty = float(s.get("hypothesis_uncertainty", s.get("uncertainty", 0.0)) or 0.0)
        if boredom >= 0.55 and curiosity >= 0.35 and uncertainty <= 0.35:
            return RETRIEVAL_SCATTER
    except Exception:
        pass
    return RETRIEVAL_DIRECTED


def _directed_score(entry: Mapping[str, Any], query_tokens: Sequence[str]) -> float:
    haystack = json.dumps(safe_json(dict(entry)), ensure_ascii=False, sort_keys=True).lower()
    token_hits = sum(1 for token in query_tokens if token and token in haystack)
    token_score = token_hits / max(1, len(query_tokens)) if query_tokens else 0.0
    return token_score + (_float_or(entry.get("confidence"), 0.0) * 0.25) + (_float_or(entry.get("weight"), 0.0) * 0.15)


def _tokens(value: str | Iterable[Any]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("_", " ").replace("-", " ").split()
    else:
        raw = [str(item) for item in value or []]
    out: list[str] = []
    for item in raw:
        token = clean_token(item, fallback="")
        if token and token not in out:
            out.append(token)
    return out


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any) -> float:
    number = _float_or(value, 0.0)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _clamp_signed(value: Any) -> float:
    number = _float_or(value, 0.0)
    if number < -1.0:
        return -1.0
    if number > 1.0:
        return 1.0
    return number
