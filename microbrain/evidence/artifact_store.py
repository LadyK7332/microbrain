"""Portable evidence artifact store.

The store writes heavier modality evidence to files under memory/evidence/ and
returns compact evidence cards that object frames and memcells can reference.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from microbrain.evidence.evidence_card import (
    build_evidence_card,
    bytes_digest,
    clean_ref,
    clean_token,
    evidence_ref_card,
    safe_json,
    short_text,
)

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

ARTIFACT_ROOT_NAME = "evidence"
DEFAULT_TEXT_EXTENSION = ".txt"
DEFAULT_JSON_EXTENSION = ".json"
DEFAULT_JSONL_EXTENSION = ".jsonl"
ARTIFACT_FSYNC = True
MAX_PREFIX_CHARS = 42
MAX_INLINE_EVIDENCE_REFS = 12

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

ARTIFACT_STORE_SCHEMA = "evidence.artifact_store.v1"
EVIDENCE_REF_PACK_SCHEMA = "evidence.ref_pack.v1"
EVIDENCE_REF_INDEX_SCHEMA = "evidence.ref_index.v1"
EVIDENCE_MULTIMODAL_REF_PACK_SCHEMA = "evidence.multimodal_ref_pack.v1"
EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA = "evidence.multimodal_ref_index.v1"
MULTIMODAL_INDEX_MODALITY = "multimodal"


class EvidenceArtifactStore:
    """Write/read evidence artifacts relative to a memory directory."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / ARTIFACT_ROOT_NAME
        self.root.mkdir(parents=True, exist_ok=True)

    def modality_dir(self, modality: str, *, timestamp: float | None = None) -> Path:
        stamp = float(timestamp or time.time())
        day = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
        path = self.root / clean_token(modality, fallback="unknown") / day
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_ref(self, path: str | Path) -> str:
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.base_dir.resolve()).as_posix()
        except Exception:
            return clean_ref(candidate.as_posix())

    def resolve_ref(self, artifact_ref: str | Path) -> Path:
        ref = clean_ref(artifact_ref)
        path = Path(ref)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def write_jsonl_artifact(
        self,
        *,
        modality: str,
        records: Iterable[Any],
        prefix: str = "sample",
        summary: str = "",
        claims_supported: Iterable[Any] | None = None,
        confidence: Any = 0.0,
        timestamp: float | None = None,
        time_range: Sequence[Any] | None = None,
        fossil_ref: str = "",
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_records = [safe_json(record) for record in records or []]
        payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in clean_records)
        data = payload.encode("utf-8", errors="replace")
        stamp = float(timestamp or time.time())
        digest = bytes_digest(data)
        path = self._artifact_path(
            modality=modality,
            prefix=prefix,
            extension=DEFAULT_JSONL_EXTENSION,
            timestamp=stamp,
            digest=digest,
        )
        self._write_bytes(path, data)
        return self._card_for_path(
            modality=modality,
            path=path,
            data=data,
            summary=summary,
            claims_supported=claims_supported,
            confidence=confidence,
            created_at=stamp,
            time_range=time_range,
            sample_count=len(clean_records),
            fossil_ref=fossil_ref,
            source=source,
            tags=tags,
            meta=meta,
        )

    def write_json_artifact(
        self,
        *,
        modality: str,
        payload: Any,
        prefix: str = "snapshot",
        summary: str = "",
        claims_supported: Iterable[Any] | None = None,
        confidence: Any = 0.0,
        timestamp: float | None = None,
        time_range: Sequence[Any] | None = None,
        fossil_ref: str = "",
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_payload = safe_json(payload)
        text = json.dumps(clean_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        data = text.encode("utf-8", errors="replace")
        stamp = float(timestamp or time.time())
        digest = bytes_digest(data)
        path = self._artifact_path(
            modality=modality,
            prefix=prefix,
            extension=DEFAULT_JSON_EXTENSION,
            timestamp=stamp,
            digest=digest,
        )
        self._write_bytes(path, data)
        return self._card_for_path(
            modality=modality,
            path=path,
            data=data,
            summary=summary,
            claims_supported=claims_supported,
            confidence=confidence,
            created_at=stamp,
            time_range=time_range,
            sample_count=1,
            fossil_ref=fossil_ref,
            source=source,
            tags=tags,
            meta=meta,
        )

    def write_bytes_artifact(
        self,
        *,
        modality: str,
        data: bytes | bytearray,
        extension: str,
        prefix: str = "artifact",
        summary: str = "",
        claims_supported: Iterable[Any] | None = None,
        confidence: Any = 0.0,
        timestamp: float | None = None,
        time_range: Sequence[Any] | None = None,
        sample_count: int = 0,
        fossil_ref: str = "",
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = bytes(data or b"")
        stamp = float(timestamp or time.time())
        digest = bytes_digest(raw)
        suffix = str(extension or "").strip()
        if not suffix.startswith("."):
            suffix = "." + suffix
        path = self._artifact_path(
            modality=modality,
            prefix=prefix,
            extension=suffix,
            timestamp=stamp,
            digest=digest,
        )
        self._write_bytes(path, raw)
        return self._card_for_path(
            modality=modality,
            path=path,
            data=raw,
            summary=summary,
            claims_supported=claims_supported,
            confidence=confidence,
            created_at=stamp,
            time_range=time_range,
            sample_count=sample_count,
            fossil_ref=fossil_ref,
            source=source,
            tags=tags,
            meta=meta,
        )


    def write_ref_index(
        self,
        *,
        modality: str,
        refs: Iterable[Any],
        prefix: str = "ref_index",
        summary: str = "",
        confidence: Any = 1.0,
        timestamp: float | None = None,
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a large reference list and return a card for the index.

        Object frames should carry a few evidence refs inline.  If a frame would
        need to carry a crowd of refs, it should carry one index card instead.
        The index file becomes the small doorway to the bigger reference set.
        """

        entries = _clean_ref_entries(refs)
        stamp = float(timestamp or time.time())
        index_payload = {
            "schema": EVIDENCE_REF_INDEX_SCHEMA,
            "modality": clean_token(modality, fallback="unknown"),
            "created_at": stamp,
            "count": len(entries),
            "refs": entries,
        }
        card_meta = dict(meta or {})
        card_meta.update({
            "artifact_store_schema": ARTIFACT_STORE_SCHEMA,
            "ref_index_schema": EVIDENCE_REF_INDEX_SCHEMA,
            "ref_count": len(entries),
        })
        index_tags = list(tags or [])
        if "ref_index" not in [str(tag) for tag in index_tags]:
            index_tags.append("ref_index")
        return self.write_json_artifact(
            modality=modality,
            payload=index_payload,
            prefix=prefix,
            summary=summary or f"{len(entries)} evidence reference(s) indexed",
            claims_supported=["evidence.ref_index"],
            confidence=confidence,
            timestamp=stamp,
            time_range=[],
            fossil_ref="",
            source=source,
            tags=index_tags,
            meta=card_meta,
        )

    def pack_refs(
        self,
        *,
        modality: str,
        refs: Iterable[Any],
        max_inline_refs: int = MAX_INLINE_EVIDENCE_REFS,
        prefix: str = "ref_index",
        summary: str = "",
        timestamp: float | None = None,
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return inline refs when small, otherwise one compact index ref.

        This is the guardrail for frames and memcells: no giant refs arrays in
        the fast path.  Small ref sets stay inline; large ref sets become one
        artifact-backed index handle.
        """

        entries = _clean_ref_entries(refs)
        limit = max(0, int(max_inline_refs))
        pack: dict[str, Any] = {
            "schema": EVIDENCE_REF_PACK_SCHEMA,
            "modality": clean_token(modality, fallback="unknown"),
            "count": len(entries),
            "max_inline_refs": limit,
            "refs": [],
            "index_ref": "",
            "index_card": {},
        }
        if len(entries) <= limit:
            pack["refs"] = entries
            return pack

        index_card = self.write_ref_index(
            modality=modality,
            refs=entries,
            prefix=prefix,
            summary=summary or f"{len(entries)} evidence reference(s) indexed",
            timestamp=timestamp,
            source=source,
            tags=tags,
            meta=meta,
        )
        pack["index_ref"] = str(index_card.get("artifact_ref", "") or "")
        pack["index_card"] = self.compact_ref(index_card)
        return pack

    def write_multimodal_ref_index(
        self,
        *,
        refs_by_modality: Mapping[str, Iterable[Any]] | Iterable[Any],
        prefix: str = "multimodal_ref_index",
        summary: str = "",
        confidence: Any = 1.0,
        timestamp: float | None = None,
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a large multimodal reference map and return its index card.

        This is the same anti-ballooning rule as ``write_ref_index``, but it
        counts refs across every modality/object bucket together.  A person,
        object, or scene frame should never carry hundreds of vision/audio/touch
        refs inline just because each individual modality looked small.
        """

        grouped = _clean_multimodal_ref_groups(refs_by_modality)
        total = sum(len(entries) for entries in grouped.values())
        stamp = float(timestamp or time.time())
        index_payload = {
            "schema": EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA,
            "modality": MULTIMODAL_INDEX_MODALITY,
            "created_at": stamp,
            "count": total,
            "modalities": sorted(grouped.keys()),
            "refs_by_modality": grouped,
        }
        card_meta = dict(meta or {})
        card_meta.update({
            "artifact_store_schema": ARTIFACT_STORE_SCHEMA,
            "ref_index_schema": EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA,
            "ref_count": total,
            "modalities": sorted(grouped.keys()),
        })
        index_tags = list(tags or [])
        for tag in ("ref_index", "multimodal_ref_index"):
            if tag not in [str(item) for item in index_tags]:
                index_tags.append(tag)
        return self.write_json_artifact(
            modality=MULTIMODAL_INDEX_MODALITY,
            payload=index_payload,
            prefix=prefix,
            summary=summary or f"{total} multimodal evidence reference(s) indexed",
            claims_supported=["evidence.multimodal_ref_index"],
            confidence=confidence,
            timestamp=stamp,
            time_range=[],
            fossil_ref="",
            source=source,
            tags=index_tags,
            meta=card_meta,
        )

    def pack_multimodal_refs(
        self,
        *,
        refs_by_modality: Mapping[str, Iterable[Any]] | Iterable[Any],
        max_inline_refs: int = MAX_INLINE_EVIDENCE_REFS,
        prefix: str = "multimodal_ref_index",
        summary: str = "",
        timestamp: float | None = None,
        source: str = "",
        tags: Iterable[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return inline multimodal refs when small, otherwise one index card.

        The threshold is global across modalities.  This prevents frames from
        keeping twelve touch refs, twelve audio refs, twelve vision refs, and a
        crowd of object refs inline.  The fast path gets a compact grouped map
        only when the whole map is small.
        """

        grouped = _clean_multimodal_ref_groups(refs_by_modality)
        total = sum(len(entries) for entries in grouped.values())
        limit = max(0, int(max_inline_refs))
        pack: dict[str, Any] = {
            "schema": EVIDENCE_MULTIMODAL_REF_PACK_SCHEMA,
            "modality": MULTIMODAL_INDEX_MODALITY,
            "count": total,
            "max_inline_refs": limit,
            "modalities": sorted(grouped.keys()),
            "refs_by_modality": {},
            "index_ref": "",
            "index_card": {},
        }
        if total <= limit:
            pack["refs_by_modality"] = grouped
            return pack

        index_card = self.write_multimodal_ref_index(
            refs_by_modality=grouped,
            prefix=prefix,
            summary=summary or f"{total} multimodal evidence reference(s) indexed",
            timestamp=timestamp,
            source=source,
            tags=tags,
            meta=meta,
        )
        pack["index_ref"] = str(index_card.get("artifact_ref", "") or "")
        pack["index_card"] = self.compact_ref(index_card)
        return pack

    def read_multimodal_ref_index(self, artifact_ref_or_card: str | Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Read a multimodal reference index file back into grouped entries."""

        payload = self.read_json_artifact(artifact_ref_or_card)
        if not isinstance(payload, Mapping):
            return {}
        if str(payload.get("schema", "") or "") != EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA:
            return {}
        return _clean_multimodal_ref_groups(payload.get("refs_by_modality", {}))

    def read_ref_index(self, artifact_ref_or_card: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        """Read an evidence reference index file back into compact entries."""

        payload = self.read_json_artifact(artifact_ref_or_card)
        if not isinstance(payload, Mapping):
            return []
        if str(payload.get("schema", "") or "") != EVIDENCE_REF_INDEX_SCHEMA:
            return []
        return _clean_ref_entries(payload.get("refs", []))

    def read_jsonl_artifact(self, artifact_ref_or_card: str | Mapping[str, Any], *, limit: int | None = None) -> list[Any]:
        ref = _ref_from_value(artifact_ref_or_card)
        path = self.resolve_ref(ref)
        rows: list[Any] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= max(0, int(limit)):
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except Exception:
                    rows.append({"raw": text})
        return rows

    def read_json_artifact(self, artifact_ref_or_card: str | Mapping[str, Any]) -> Any:
        ref = _ref_from_value(artifact_ref_or_card)
        path = self.resolve_ref(ref)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

    def compact_ref(self, card: Mapping[str, Any]) -> dict[str, Any]:
        return evidence_ref_card(card)

    def _artifact_path(
        self,
        *,
        modality: str,
        prefix: str,
        extension: str,
        timestamp: float,
        digest: str,
    ) -> Path:
        clean_prefix = clean_token(prefix, fallback="artifact")[:MAX_PREFIX_CHARS]
        ms = int(float(timestamp) * 1000)
        filename = f"{clean_prefix}_{ms}_{digest[:12]}{extension}"
        return self.modality_dir(modality, timestamp=timestamp) / filename

    def _write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            if ARTIFACT_FSYNC:
                os.fsync(handle.fileno())
        tmp.replace(path)

    def _card_for_path(
        self,
        *,
        modality: str,
        path: Path,
        data: bytes,
        summary: str,
        claims_supported: Iterable[Any] | None,
        confidence: Any,
        created_at: float,
        time_range: Sequence[Any] | None,
        sample_count: int,
        fossil_ref: str,
        source: str,
        tags: Iterable[Any] | None,
        meta: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        card_meta = dict(meta or {})
        card_meta.setdefault("artifact_store_schema", ARTIFACT_STORE_SCHEMA)
        return build_evidence_card(
            modality=modality,
            artifact_ref=self.to_ref(path),
            summary=short_text(summary),
            claims_supported=claims_supported,
            confidence=confidence,
            created_at=created_at,
            time_range=time_range,
            sample_count=sample_count,
            byte_count=len(data),
            checksum=f"blake2b:{bytes_digest(data)}",
            fossil_ref=fossil_ref,
            source=source,
            tags=tags,
            meta=card_meta,
        )


def _clean_ref_entries(values: Iterable[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in values or []:
        entry = _clean_ref_entry(raw)
        if not entry:
            continue
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _clean_ref_entry(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        looks_like_card = (
            value.get("schema") == "evidence.ref.v1"
            or value.get("evidence_id")
            or any(key in value for key in ("modality", "summary", "claims_supported", "confidence", "fossil_ref"))
        )
        if looks_like_card:
            entry = evidence_ref_card(value)
        else:
            ref = clean_ref(value.get("artifact_ref") or value.get("ref") or value.get("path") or value.get("artifact") or "")
            entry = {"artifact_ref": ref} if ref else {}
        for key in ("ref", "path", "source_ref"):
            if key in value and not entry.get("artifact_ref"):
                entry["artifact_ref"] = clean_ref(value.get(key))
        if value.get("kind") and "kind" not in entry:
            entry["kind"] = str(value.get("kind") or "")
        return {k: v for k, v in entry.items() if v not in (None, "", [], {})}
    ref = clean_ref(value)
    return {"artifact_ref": ref} if ref else {}


def _clean_multimodal_ref_groups(values: Mapping[str, Iterable[Any]] | Iterable[Any] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if isinstance(values, Mapping):
        for raw_modality, raw_refs in values.items():
            modality = clean_token(raw_modality, fallback="unknown")
            for entry in _clean_ref_entries(_as_ref_iterable(raw_refs)):
                if not entry.get("modality") or entry.get("modality") == "unknown":
                    entry["modality"] = modality
                bucket = grouped.setdefault(modality, [])
                bucket.append(entry)
    else:
        for entry in _clean_ref_entries(_as_ref_iterable(values)):
            modality = clean_token(entry.get("modality", "unknown"), fallback="unknown")
            entry["modality"] = modality
            grouped.setdefault(modality, []).append(entry)

    clean_grouped: dict[str, list[dict[str, Any]]] = {}
    for modality, entries in grouped.items():
        deduped = _clean_ref_entries(entries)
        if deduped:
            clean_grouped[clean_token(modality, fallback="unknown")] = deduped
    return dict(sorted(clean_grouped.items()))


def _as_ref_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _ref_from_value(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        return clean_ref(value.get("artifact_ref", ""))
    return clean_ref(value)
