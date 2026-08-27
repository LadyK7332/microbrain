from __future__ import annotations

"""Evidence artifact loader for deliberation.

Fast object frames carry evidence refs.  This module opens the heavier artifact
only when a deliberation, hypothesis, review, or debugging organ explicitly asks.
"""

import hashlib
import json
import random
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from microbrain.evidence.artifact_store import EvidenceArtifactStore
from microbrain.evidence.evidence_card import clean_ref, short_text

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

MAX_DEFAULT_ITEMS = 8
MAX_RECORDS_SCANNED = 512
MAX_INLINE_TEXT_CHARS = 1800
MAX_ITEM_CHARS = 900
MAX_QUERY_TERMS = 12

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

EVIDENCE_LOADED_SCHEMA = "evidence.loaded.v1"
EVIDENCE_LOAD_ERROR_SCHEMA = "evidence.load_error.v1"
SUPPORTED_LOAD_MODES = frozenset({"summary", "head", "tail", "walk", "directed", "scatter"})
MODE_ALIASES = {
    "novel": "directed",
    "novel_search": "directed",
    "search": "directed",
    "query": "directed",
    "one_by_one": "walk",
    "sequential": "walk",
    "shotgun": "scatter",
    "random": "scatter",
    "sample": "scatter",
}
TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_load_mode(value: Any) -> str:
    mode = str(value or "summary").strip().lower().replace("-", "_").replace(" ", "_")
    mode = MODE_ALIASES.get(mode, mode)
    return mode if mode in SUPPORTED_LOAD_MODES else "summary"


def artifact_ref_from(value: Any) -> str:
    """Extract an artifact ref from a request, evidence ref, or card."""
    if isinstance(value, Mapping):
        for key in (
            "artifact_ref",
            "index_ref",
            "ledger_ref",
            "data_ref",
            "ref",
            "path",
            "source_ref",
        ):
            ref = clean_ref(value.get(key, ""))
            if ref:
                return ref
        evidence_ref = value.get("evidence_ref") or value.get("index_card") or value.get("card")
        if isinstance(evidence_ref, Mapping):
            return artifact_ref_from(evidence_ref)
    return clean_ref(value)


def safe_resolve_artifact_path(base_dir: str | Path, artifact_ref_or_card: Any) -> Path:
    """Resolve an artifact ref and keep reads inside the memory directory.

    Evidence cards should normally carry relative refs such as
    ``evidence/touch/2026-08-20/touch_delta_...jsonl``.  Absolute paths are
    accepted only when they resolve under the configured memory base dir.  Path
    traversal is rejected instead of silently normalized.
    """
    base = Path(base_dir).resolve()
    ref = artifact_ref_from(artifact_ref_or_card)
    if not ref:
        raise ValueError("missing artifact_ref")
    candidate = Path(ref)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if any(part == ".." for part in Path(ref).parts):
            raise ValueError("artifact_ref may not contain '..'")
        resolved = (base / ref).resolve()
    try:
        resolved.relative_to(base)
    except Exception as exc:
        raise ValueError("artifact_ref resolves outside memory base dir") from exc
    return resolved


def _tokens(text: Any) -> list[str]:
    toks: list[str] = []
    for tok in TOKEN_RE.findall(str(text or "").lower()):
        if len(tok) >= 2 and tok not in toks:
            toks.append(tok)
        if len(toks) >= MAX_QUERY_TERMS:
            break
    return toks


def _json_text(value: Any, *, limit: int = MAX_ITEM_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _compact_item(value: Any, *, index: int | None = None, score: float | None = None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        item: dict[str, Any] = {}
        for key in (
            "schema",
            "summary",
            "description",
            "text",
            "label",
            "source",
            "created_at",
            "timestamp",
            "time_range",
            "features",
            "claims_supported",
            "confidence",
            "payload_meta",
            "artifact_ref",
            "evidence_id",
            "modality",
            "kind",
        ):
            if key in value and value.get(key) not in (None, "", [], {}):
                item[key] = _bound_value(value.get(key))
        if "records" in value:
            records = value.get("records")
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
                item["records_summary"] = {"count": len(records), "sample": [_bound_value(x) for x in list(records)[:3]]}
        if not item:
            item["value"] = _json_text(value)
    else:
        item = {"value": _bound_value(value)}
    if index is not None:
        item["index"] = int(index)
    if score is not None:
        item["score"] = round(float(score), 4)
    return item


def _bound_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return short_text(value, limit=MAX_ITEM_CHARS)
    if depth >= 2:
        return _json_text(value, limit=360)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 16:
                out["_omitted_key_count"] = len(value) - 16
                break
            out[str(key)] = _bound_value(val, depth=depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seq = list(value)
        out = [_bound_value(x, depth=depth + 1) for x in seq[:8]]
        if len(seq) > 8:
            return {"schema": "evidence.loader.sequence_sample.v1", "count": len(seq), "sample": out}
        return out
    return short_text(str(value), MAX_ITEM_CHARS)


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[Any]:
    rows: list[Any] = []
    max_rows = MAX_RECORDS_SCANNED if limit is None else max(0, int(limit))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception:
                rows.append({"raw": short_text(text, limit=MAX_ITEM_CHARS)})
    return rows


def _read_jsonl_tail(path: Path, *, limit: int) -> list[Any]:
    tail: deque[Any] = deque(maxlen=max(0, int(limit)))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                tail.append(json.loads(text))
            except Exception:
                tail.append({"raw": short_text(text, limit=MAX_ITEM_CHARS)})
    return list(tail)


def _row_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in ("summary", "description", "text", "label", "schema", "source"):
            if value.get(key) not in (None, "", [], {}):
                parts.append(str(value.get(key)))
        if not parts:
            parts.append(_json_text(value, limit=MAX_ITEM_CHARS))
        return " ".join(parts).lower()
    return str(value or "").lower()


def _score_row(value: Any, query_terms: Sequence[str]) -> float:
    if not query_terms:
        return 0.0
    text = _row_text(value)
    score = 0.0
    for term in query_terms:
        if term in text:
            score += 1.0
    return score / max(1.0, float(len(query_terms)))


def _load_json(path: Path, *, mode: str, limit: int, query_terms: Sequence[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"error": f"json parse failed: {exc}"}

    items: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        if isinstance(payload.get("refs"), list):
            raw_items = list(payload.get("refs") or [])
            items = [_compact_item(x, index=i) for i, x in enumerate(raw_items[:limit])]
        elif isinstance(payload.get("refs_by_modality"), Mapping):
            flat: list[Any] = []
            for modality, refs in payload.get("refs_by_modality", {}).items():
                for ref in refs or []:
                    entry = dict(ref) if isinstance(ref, Mapping) else {"artifact_ref": ref}
                    entry.setdefault("modality", str(modality))
                    flat.append(entry)
            items = [_compact_item(x, index=i) for i, x in enumerate(flat[:limit])]
        else:
            items = [_compact_item(payload, index=0)]
    elif isinstance(payload, list):
        raw_items = payload
        if mode == "directed" and query_terms:
            scored = [(idx, item, _score_row(item, query_terms)) for idx, item in enumerate(raw_items)]
            scored = [row for row in scored if row[2] > 0]
            scored.sort(key=lambda row: row[2], reverse=True)
            items = [_compact_item(item, index=idx, score=score) for idx, item, score in scored[:limit]]
        else:
            items = [_compact_item(item, index=i) for i, item in enumerate(raw_items[:limit])]
    else:
        items = [_compact_item(payload, index=0)]

    return {
        "artifact_kind": "json",
        "item_count": len(items),
        "items": items,
        "payload_schema": str(payload.get("schema", "") or "") if isinstance(payload, Mapping) else "",
    }


def _load_jsonl(path: Path, *, mode: str, limit: int, offset: int, query_terms: Sequence[str], seed: Any = None) -> dict[str, Any]:
    limit = max(0, int(limit))
    offset = max(0, int(offset))

    if mode == "tail":
        rows = _read_jsonl_tail(path, limit=limit)
        indexed = list(enumerate(rows))
    else:
        rows = _read_jsonl(path, limit=MAX_RECORDS_SCANNED)
        indexed = list(enumerate(rows))

    if mode == "walk":
        selected = indexed[offset:offset + limit]
    elif mode == "scatter":
        rng_seed = seed if seed not in (None, "") else hashlib.blake2b(str(path).encode(), digest_size=8).hexdigest()
        rng = random.Random(str(rng_seed))
        selected = indexed[:]
        rng.shuffle(selected)
        selected = selected[:limit]
        selected.sort(key=lambda row: row[0])
    elif mode == "directed" and query_terms:
        scored = [(idx, row, _score_row(row, query_terms)) for idx, row in indexed]
        scored = [row for row in scored if row[2] > 0]
        scored.sort(key=lambda row: row[2], reverse=True)
        selected = [(idx, row, score) for idx, row, score in scored[:limit]]
        return {
            "artifact_kind": "jsonl",
            "item_count": len(selected),
            "items": [_compact_item(row, index=idx, score=score) for idx, row, score in selected],
            "scanned_count": len(indexed),
            "matched_terms": list(query_terms),
        }
    else:
        selected = indexed[:limit]

    return {
        "artifact_kind": "jsonl",
        "item_count": len(selected),
        "items": [_compact_item(row, index=idx) for idx, row in selected],
        "scanned_count": len(indexed),
        "matched_terms": list(query_terms),
    }


def _load_text(path: Path, *, limit: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "artifact_kind": "text",
        "item_count": 1 if text else 0,
        "items": [{"index": 0, "text": short_text(text, limit=min(MAX_INLINE_TEXT_CHARS, max(256, limit * 256)))}] if text else [],
        "byte_count": path.stat().st_size,
    }


def load_evidence_reference(base_dir: str | Path, request: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Load a small, bounded evidence sample for deliberation.

    The output is intentionally compact.  This function proves or inspects a ref;
    it does not turn the raw artifact back into normal memory.
    """
    req = dict(request) if isinstance(request, Mapping) else {"artifact_ref": request}
    mode = normalize_load_mode(req.get("mode") or req.get("retrieval_mode"))
    limit = max(1, min(64, int(req.get("limit", MAX_DEFAULT_ITEMS) or MAX_DEFAULT_ITEMS)))
    offset = max(0, int(req.get("offset", 0) or 0))
    query = str(req.get("query", "") or req.get("text", "") or "")
    query_terms = _tokens(query)

    try:
        path = safe_resolve_artifact_path(base_dir, req)
    except Exception as exc:
        return {
            "schema": EVIDENCE_LOAD_ERROR_SCHEMA,
            "ok": False,
            "error": str(exc),
            "mode": mode,
            "artifact_ref": artifact_ref_from(req),
        }

    ref = EvidenceArtifactStore(base_dir).to_ref(path)
    if not path.exists():
        return {
            "schema": EVIDENCE_LOAD_ERROR_SCHEMA,
            "ok": False,
            "error": "artifact not found",
            "mode": mode,
            "artifact_ref": ref,
        }
    if not path.is_file():
        return {
            "schema": EVIDENCE_LOAD_ERROR_SCHEMA,
            "ok": False,
            "error": "artifact_ref is not a file",
            "mode": mode,
            "artifact_ref": ref,
        }

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        loaded = _load_jsonl(path, mode=mode, limit=limit, offset=offset, query_terms=query_terms, seed=req.get("seed"))
    elif suffix == ".json":
        loaded = _load_json(path, mode=mode, limit=limit, query_terms=query_terms)
    elif suffix in {".txt", ".md", ".log", ".csv"}:
        loaded = _load_text(path, limit=limit)
    else:
        loaded = {
            "artifact_kind": "binary",
            "item_count": 0,
            "items": [],
            "byte_count": path.stat().st_size,
            "note": "binary artifact kept closed; use modality-specific reader",
        }

    return {
        "schema": EVIDENCE_LOADED_SCHEMA,
        "ok": True,
        "artifact_ref": ref,
        "mode": mode,
        "query": query,
        "limit": limit,
        "offset": offset,
        "byte_count": path.stat().st_size,
        **loaded,
    }
