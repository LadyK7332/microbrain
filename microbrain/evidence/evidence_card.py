"""Compact evidence cards for sensor artifacts.

Evidence cards are small, durable handles.  They summarize why an artifact
matters without embedding the artifact's raw samples into object frames or
memcells.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

EVIDENCE_DIGEST_SIZE = 16
DEFAULT_CONFIDENCE = 0.0
MAX_SUMMARY_CHARS = 280
MAX_SUPPORTED_CLAIMS = 32
MAX_TAGS = 32

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

EVIDENCE_CARD_SCHEMA = "evidence.card.v1"
EVIDENCE_REF_PREFIX = "evidence"
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")


def clamp01(value: Any, default: float = DEFAULT_CONFIDENCE) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def safe_json(value: Any) -> Any:
    """Return a JSON-compatible representation without throwing."""

    try:
        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, Mapping):
            return {str(k): safe_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [safe_json(v) for v in value]
        return repr(value)


def stable_digest(value: Any, *, digest_size: int = EVIDENCE_DIGEST_SIZE) -> str:
    try:
        raw = json.dumps(safe_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        raw = repr(value)
    return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=digest_size).hexdigest()


def bytes_digest(data: bytes, *, digest_size: int = EVIDENCE_DIGEST_SIZE) -> str:
    return hashlib.blake2b(bytes(data), digest_size=digest_size).hexdigest()


def clean_token(value: Any, *, fallback: str = "unknown") -> str:
    text = _SAFE_TOKEN_RE.sub("_", str(value or "").strip()).strip("_ .:/\\")
    return text.lower() or fallback


def clean_ref(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("/")


def short_text(value: Any, *, limit: int = MAX_SUMMARY_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def clean_list(values: Iterable[Any] | None, *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        text = " ".join(str(raw or "").split())
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        if len(out) >= limit:
            break
    return out


@dataclass(slots=True)
class EvidenceCard:
    """Small durable handle pointing at a heavier evidence artifact."""

    modality: str
    artifact_ref: str
    summary: str = ""
    claims_supported: list[str] = field(default_factory=list)
    confidence: float = DEFAULT_CONFIDENCE
    schema: str = EVIDENCE_CARD_SCHEMA
    evidence_id: str = ""
    created_at: float = 0.0
    time_range: list[float] = field(default_factory=list)
    sample_count: int = 0
    byte_count: int = 0
    checksum: str = ""
    fossil_ref: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        card = asdict(self)
        card["modality"] = clean_token(card.get("modality"), fallback="unknown")
        card["artifact_ref"] = clean_ref(card.get("artifact_ref"))
        card["summary"] = short_text(card.get("summary", ""))
        card["claims_supported"] = clean_list(card.get("claims_supported", []), limit=MAX_SUPPORTED_CLAIMS)
        card["confidence"] = clamp01(card.get("confidence", DEFAULT_CONFIDENCE))
        card["created_at"] = float(card.get("created_at") or time.time())
        card["time_range"] = _clean_time_range(card.get("time_range", []))
        card["sample_count"] = max(0, int(card.get("sample_count", 0) or 0))
        card["byte_count"] = max(0, int(card.get("byte_count", 0) or 0))
        card["checksum"] = str(card.get("checksum", "") or "")
        card["fossil_ref"] = str(card.get("fossil_ref", "") or "")
        card["source"] = str(card.get("source", "") or "")
        card["tags"] = clean_list(card.get("tags", []), limit=MAX_TAGS)
        card["meta"] = safe_json(card.get("meta", {}) or {})
        if not card.get("evidence_id"):
            card["evidence_id"] = make_evidence_id(card)
        return card


def _clean_time_range(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: list[float] = []
    for item in list(value)[:2]:
        try:
            out.append(float(item))
        except Exception:
            pass
    if len(out) == 2 and out[1] < out[0]:
        out = [out[1], out[0]]
    return out


def make_evidence_id(card: Mapping[str, Any]) -> str:
    basis = {
        "schema": EVIDENCE_CARD_SCHEMA,
        "modality": clean_token(card.get("modality"), fallback="unknown"),
        "artifact_ref": clean_ref(card.get("artifact_ref")),
        "checksum": str(card.get("checksum", "") or ""),
        "time_range": _clean_time_range(card.get("time_range", [])),
    }
    return f"evidence:{basis['modality']}:{stable_digest(basis, digest_size=10)}"


def build_evidence_card(
    *,
    modality: str,
    artifact_ref: str,
    summary: str = "",
    claims_supported: Iterable[Any] | None = None,
    confidence: Any = DEFAULT_CONFIDENCE,
    created_at: float | None = None,
    time_range: Sequence[Any] | None = None,
    sample_count: int = 0,
    byte_count: int = 0,
    checksum: str = "",
    fossil_ref: str = "",
    source: str = "",
    tags: Iterable[Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact evidence card from a persisted artifact reference."""

    return EvidenceCard(
        modality=modality,
        artifact_ref=artifact_ref,
        summary=summary,
        claims_supported=clean_list(claims_supported, limit=MAX_SUPPORTED_CLAIMS),
        confidence=clamp01(confidence),
        created_at=float(created_at or time.time()),
        time_range=list(time_range or []),
        sample_count=max(0, int(sample_count or 0)),
        byte_count=max(0, int(byte_count or 0)),
        checksum=str(checksum or ""),
        fossil_ref=str(fossil_ref or ""),
        source=str(source or ""),
        tags=clean_list(tags, limit=MAX_TAGS),
        meta=dict(meta or {}),
    ).to_dict()


def evidence_ref_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tiny subset object frames should normally embed."""

    return {
        "schema": "evidence.ref.v1",
        "evidence_id": str(card.get("evidence_id", "") or ""),
        "modality": clean_token(card.get("modality", "unknown"), fallback="unknown"),
        "artifact_ref": clean_ref(card.get("artifact_ref", "")),
        "summary": short_text(card.get("summary", "")),
        "claims_supported": clean_list(card.get("claims_supported", []), limit=MAX_SUPPORTED_CLAIMS),
        "confidence": clamp01(card.get("confidence", DEFAULT_CONFIDENCE)),
        "fossil_ref": str(card.get("fossil_ref", "") or ""),
    }
