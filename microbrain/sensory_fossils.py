from __future__ import annotations

"""
Sensory Fossil Organ primitives.

This module is intentionally modality-neutral at the protocol boundary:
vision, touch, audio, and later senses may keep different encoders, but all of
those encoders return the same EvidencePacket shape.  A fossil match is evidence,
not belief.
"""

import hashlib
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SUPPORTED_MODALITIES = frozenset({"vision", "touch", "audio", "proprioception", "temperature"})
UNCERTAIN_SUFFIX = "?"

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

DEFAULT_FOSSIL_MATCH_THRESHOLD = 0.58
DEFAULT_PACKET_CANDIDATE_THRESHOLD = 0.65
DEFAULT_MAX_MATCHES = 6
DEFAULT_COLOR_WEIGHT = 0.25
DEFAULT_FEATURE_WEIGHT = 0.75

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        x = default
    if math.isnan(x) or math.isinf(x):
        return default
    return max(0.0, min(1.0, x))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def normalize_candidate(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(UNCERTAIN_SUFFIX):
        text = text[:-1].strip()
    return text


def candidate_with_uncertainty(value: Any, confident: bool) -> str:
    base = normalize_candidate(value)
    if not base:
        return "unknown?"
    return base if confident else f"{base}?"


def normalize_modality(value: Any) -> str:
    modality = str(value or "").strip().lower()
    return modality if modality in SUPPORTED_MODALITIES else "unknown"


def normalize_tags(tags: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        tag = str(raw or "").strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def _stable_hash(parts: Iterable[Any], prefix: str = "id") -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return f"{prefix}_{h.hexdigest()[:16]}"


def parse_hex_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def canonical_hex_color(value: Any) -> str:
    rgb = parse_hex_color(value)
    if rgb is None:
        return ""
    return "#%02X%02X%02X" % rgb


def color_similarity(a: Any, b: Any) -> float:
    ca = parse_hex_color(a)
    cb = parse_hex_color(b)
    if ca is None or cb is None:
        return 0.0
    dist = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(ca, cb)))
    max_dist = math.sqrt(3.0 * (255.0**2))
    return clamp01(1.0 - (dist / max_dist))


def _flatten_values(value: Any, *, max_items: int = 256) -> list[float]:
    """Flatten numeric-ish feature data into a normalized vector.

    Accepted forms:
    - mapping of numeric values
    - list/tuple of numeric values
    - nested list/tuple, such as a grayscale thumbnail

    Large byte/pixel ranges are normalized to 0..1 by assuming 0..255.
    """
    raw: list[float] = []

    def walk(x: Any) -> None:
        if len(raw) >= max_items:
            return
        if isinstance(x, Mapping):
            for key in sorted(x.keys(), key=str):
                walk(x[key])
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                walk(item)
            return
        try:
            v = float(x)
        except Exception:
            return
        if math.isnan(v) or math.isinf(v):
            return
        if abs(v) > 1.0:
            v = v / 255.0
        raw.append(clamp01(v))

    walk(value)
    return raw


def feature_vector_from_payload(payload: Mapping[str, Any]) -> tuple[list[float], list[str]]:
    """Extract a comparable feature vector from a fossil/query payload."""
    if not isinstance(payload, Mapping):
        return [], []

    feature_names: list[str] = []
    vector: list[float] = []

    # A low-res grayscale fossil is the preferred vision-friendly signature.
    gray = payload.get("low_res_gray") or payload.get("gray") or payload.get("grayscale")
    gray_vec = _flatten_values(gray, max_items=256)
    if gray_vec:
        step = max(1, len(gray_vec) // 64)
        sampled = gray_vec[::step][:64]
        vector.extend(sampled)
        feature_names.extend([f"gray:{i}" for i in range(len(sampled))])

    features = payload.get("features") or payload.get("signature") or {}
    if isinstance(features, Mapping):
        for key in sorted(features.keys(), key=str):
            vals = _flatten_values(features[key], max_items=16)
            for idx, val in enumerate(vals):
                vector.append(val)
                feature_names.append(f"{key}:{idx}" if len(vals) > 1 else str(key))
    else:
        vals = _flatten_values(features, max_items=128)
        vector.extend(vals)
        feature_names.extend([f"feature:{i}" for i in range(len(vals))])

    return vector, feature_names


def vector_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    # Average absolute difference is stable for cheap visual/touch/audio fossils.
    diff = sum(abs(clamp01(a[i]) - clamp01(b[i])) for i in range(n)) / float(n)
    len_penalty = abs(len(a) - len(b)) / float(max(len(a), len(b)))
    return clamp01(1.0 - diff - (0.15 * len_penalty))


# ---------------------------------------------------------------------------
# Data protocol
# ---------------------------------------------------------------------------


@dataclass
class FossilRecord:
    fossil_id: str
    modality: str
    concept: str
    branch: str = ""
    source_ref: str = ""
    feature_vector: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    low_res_ref: str = ""
    dominant_color: str = ""
    required_color: str = ""
    confidence: float = 0.0
    objecthood_confidence: float = 0.0
    stability: float = 0.0
    evidence_count: int = 1
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate(self) -> str:
        return self.branch or self.concept or "unknown"

    @property
    def trailing_tags(self) -> list[str]:
        tags = list(self.tags)
        if self.concept:
            tags.append(f"#mem_cell:{self.concept}")
        if self.branch:
            tags.append(f"#branch:{self.branch}")
        if self.modality:
            tags.append(f"#fossil:{self.modality}:{self.fossil_id}")
        if self.dominant_color:
            tags.append(f"#color:{self.dominant_color}")
        if self.required_color:
            tags.append(f"#requires_color:{self.required_color}")
        return normalize_tags(tags)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = self.trailing_tags
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FossilRecord":
        payload = dict(data or {})
        payload["modality"] = normalize_modality(payload.get("modality"))
        payload["confidence"] = clamp01(payload.get("confidence"))
        payload["objecthood_confidence"] = clamp01(payload.get("objecthood_confidence"))
        payload["stability"] = clamp01(payload.get("stability"))
        payload["tags"] = normalize_tags(payload.get("tags", []))
        payload["dominant_color"] = canonical_hex_color(payload.get("dominant_color"))
        payload["required_color"] = canonical_hex_color(payload.get("required_color"))
        payload.setdefault("feature_vector", [])
        payload.setdefault("feature_names", [])
        payload.setdefault("meta", {})
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__.keys() if k in payload})


@dataclass
class FossilMatch:
    fossil: FossilRecord
    feature_similarity: float
    color_similarity: float
    score: float
    supports: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)

    def to_packet(
        self,
        *,
        source_ref: str,
        modality: str,
        importance: float = 0.0,
        timestamp: float | None = None,
        query_meta: Mapping[str, Any] | None = None,
    ) -> "EvidencePacket":
        confidence = clamp01((self.score * 0.60) + (self.fossil.confidence * 0.20) + (self.fossil.stability * 0.20))
        confident = confidence >= DEFAULT_PACKET_CANDIDATE_THRESHOLD
        packet_id = _stable_hash(
            [source_ref, modality, self.fossil.fossil_id, self.score, timestamp or time.time()],
            prefix="evpkt",
        )
        meta = dict(query_meta or {})
        meta.update(
            {
                "fossil_id": self.fossil.fossil_id,
                "concept": self.fossil.concept,
                "branch": self.fossil.branch,
                "feature_similarity": round(self.feature_similarity, 4),
                "color_similarity": round(self.color_similarity, 4),
            }
        )
        return EvidencePacket(
            packet_id=packet_id,
            modality=normalize_modality(modality),
            source_ref=str(source_ref or ""),
            candidate=candidate_with_uncertainty(self.fossil.candidate, confident=confident),
            similarity=round(self.score, 4),
            confidence=round(confidence, 4),
            importance=clamp01(importance),
            fossil_refs=[f"{self.fossil.modality}:fossil:{self.fossil.fossil_id}"],
            mem_cell_tags=self.fossil.trailing_tags,
            supports=list(self.supports),
            uncertainty=list(self.uncertainty),
            timestamp=timestamp or time.time(),
            meta=meta,
        )


@dataclass
class EvidencePacket:
    packet_id: str
    modality: str
    source_ref: str
    candidate: str
    similarity: float
    confidence: float
    importance: float = 0.0
    fossil_refs: list[str] = field(default_factory=list)
    mem_cell_tags: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_family(self) -> str:
        base = normalize_candidate(self.candidate)
        if "_" in base:
            return base.split("_")[-1] if base.startswith(("green_", "red_", "blue_")) else base.split("_")[0]
        return base

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modality"] = normalize_modality(data.get("modality"))
        data["candidate"] = str(data.get("candidate") or "unknown?")
        data["similarity"] = clamp01(data.get("similarity"))
        data["confidence"] = clamp01(data.get("confidence"))
        data["importance"] = clamp01(data.get("importance"))
        data["mem_cell_tags"] = normalize_tags(data.get("mem_cell_tags"))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidencePacket":
        payload = dict(data or {})
        payload.setdefault("packet_id", _stable_hash([payload], prefix="evpkt"))
        payload.setdefault("timestamp", time.time())
        payload.setdefault("fossil_refs", [])
        payload.setdefault("mem_cell_tags", [])
        payload.setdefault("supports", [])
        payload.setdefault("uncertainty", [])
        payload.setdefault("meta", {})
        payload["modality"] = normalize_modality(payload.get("modality"))
        payload["similarity"] = clamp01(payload.get("similarity"))
        payload["confidence"] = clamp01(payload.get("confidence"))
        payload["importance"] = clamp01(payload.get("importance"))
        payload["mem_cell_tags"] = normalize_tags(payload.get("mem_cell_tags"))
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__.keys() if k in payload})


# ---------------------------------------------------------------------------
# Fossil instances and store
# ---------------------------------------------------------------------------


class SensoryFossilInstance:
    """One modality-specific fossil index using the shared evidence protocol."""

    def __init__(self, modality: str):
        self.modality = normalize_modality(modality)
        self._fossils: dict[str, FossilRecord] = {}

    @property
    def fossils(self) -> tuple[FossilRecord, ...]:
        return tuple(self._fossils.values())

    def add_fossil(self, fossil: FossilRecord) -> FossilRecord:
        if fossil.modality != self.modality:
            fossil.modality = self.modality
        fossil.tags = fossil.trailing_tags
        self._fossils[fossil.fossil_id] = fossil
        return fossil

    def build_fossil(self, payload: Mapping[str, Any]) -> FossilRecord:
        vector, names = feature_vector_from_payload(payload)
        concept = normalize_candidate(payload.get("concept") or payload.get("label") or payload.get("candidate") or "unknown")
        branch = normalize_candidate(payload.get("branch") or payload.get("branch_concept") or "")
        dominant_color = canonical_hex_color(payload.get("dominant_color") or payload.get("color_hex") or payload.get("color"))
        required_color = canonical_hex_color(payload.get("required_color") or payload.get("color_requirement") or "")
        source_ref = str(payload.get("source_ref") or payload.get("track_id") or payload.get("source") or "")
        stable_parts = [self.modality, concept, branch, source_ref, vector[:32], dominant_color, required_color]
        fossil_id = str(payload.get("fossil_id") or _stable_hash(stable_parts, prefix=f"{self.modality}_fossil"))
        tags = normalize_tags(payload.get("tags", []))
        fossil = FossilRecord(
            fossil_id=fossil_id,
            modality=self.modality,
            concept=concept,
            branch=branch,
            source_ref=source_ref,
            feature_vector=vector,
            feature_names=names,
            low_res_ref=str(payload.get("low_res_ref") or payload.get("crop_ref") or ""),
            dominant_color=dominant_color,
            required_color=required_color,
            confidence=clamp01(payload.get("confidence", payload.get("identity_confidence", 0.0))),
            objecthood_confidence=clamp01(payload.get("objecthood_confidence", payload.get("object_confidence", 0.0))),
            stability=clamp01(payload.get("stability", payload.get("stable", 0.0))),
            evidence_count=max(1, int(_safe_float(payload.get("evidence_count", 1), 1.0))),
            tags=tags,
            meta=dict(payload.get("meta") or {}),
        )
        fossil.tags = fossil.trailing_tags
        return fossil

    def store_from_payload(self, payload: Mapping[str, Any]) -> FossilRecord:
        return self.add_fossil(self.build_fossil(payload))

    def query(
        self,
        payload: Mapping[str, Any],
        *,
        threshold: float = DEFAULT_FOSSIL_MATCH_THRESHOLD,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ) -> list[FossilMatch]:
        query_vector, _names = feature_vector_from_payload(payload)
        query_color = canonical_hex_color(payload.get("dominant_color") or payload.get("color_hex") or payload.get("color"))
        candidate_hint = normalize_candidate(payload.get("candidate") or payload.get("concept") or payload.get("label") or "")
        branch_hint = normalize_candidate(payload.get("branch") or payload.get("branch_concept") or "")
        matches: list[FossilMatch] = []

        for fossil in self._fossils.values():
            if candidate_hint and candidate_hint not in {fossil.concept, fossil.branch, fossil.candidate}:
                # Keep family branches findable. Querying button may return green_button.
                if not (fossil.concept == candidate_hint or fossil.candidate.endswith("_" + candidate_hint)):
                    continue
            if branch_hint and fossil.branch and branch_hint != fossil.branch:
                continue

            fsim = vector_similarity(query_vector, fossil.feature_vector)
            csim = 0.0
            supports: list[str] = []
            uncertainty: list[str] = []

            if fsim > 0.0:
                supports.append(f"{self.modality} feature fossil match {fsim:.2f}")
            else:
                uncertainty.append("feature signature missing or incomparable")

            color_required = bool(fossil.required_color)
            if query_color and (fossil.required_color or fossil.dominant_color):
                target_color = fossil.required_color or fossil.dominant_color
                csim = color_similarity(query_color, target_color)
                if csim >= 0.75:
                    supports.append(f"color branch {target_color} match {csim:.2f}")
                elif color_required:
                    uncertainty.append(f"required color {target_color} weak match {csim:.2f}")
            elif color_required:
                uncertainty.append(f"required color {fossil.required_color} not observed")

            feature_weight = DEFAULT_FEATURE_WEIGHT
            color_weight = DEFAULT_COLOR_WEIGHT if (query_color and (fossil.required_color or fossil.dominant_color)) else 0.0
            if color_required:
                color_weight = 0.40
                feature_weight = 0.60
            score = (fsim * feature_weight) + (csim * color_weight)
            if color_weight <= 0.0:
                score = fsim

            if score >= clamp01(threshold):
                matches.append(
                    FossilMatch(
                        fossil=fossil,
                        feature_similarity=round(fsim, 4),
                        color_similarity=round(csim, 4),
                        score=round(score, 4),
                        supports=supports,
                        uncertainty=uncertainty,
                    )
                )

        matches.sort(key=lambda m: (m.score, m.fossil.confidence, m.fossil.stability), reverse=True)
        return matches[: max(1, int(max_matches))]

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "fossils": [f.to_dict() for f in self._fossils.values()],
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "SensoryFossilInstance":
        inst = cls(str((data or {}).get("modality") or "unknown"))
        for row in list((data or {}).get("fossils", []) or []):
            if isinstance(row, Mapping):
                inst.add_fossil(FossilRecord.from_dict(row))
        return inst


class SensoryFossilStore:
    """Container for per-sense fossil instances."""

    def __init__(self):
        self._instances: dict[str, SensoryFossilInstance] = {}

    def instance(self, modality: str) -> SensoryFossilInstance:
        key = normalize_modality(modality)
        if key not in self._instances:
            self._instances[key] = SensoryFossilInstance(key)
        return self._instances[key]

    def store_from_payload(self, payload: Mapping[str, Any]) -> FossilRecord:
        modality = normalize_modality(payload.get("modality"))
        return self.instance(modality).store_from_payload(payload)

    def query_packets(
        self,
        payload: Mapping[str, Any],
        *,
        threshold: float = DEFAULT_FOSSIL_MATCH_THRESHOLD,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ) -> list[EvidencePacket]:
        modality = normalize_modality(payload.get("modality"))
        source_ref = str(payload.get("source_ref") or payload.get("track_id") or payload.get("source") or "")
        importance = clamp01(payload.get("importance", payload.get("salience", 0.0)))
        timestamp = _safe_float(payload.get("timestamp", time.time()), time.time())
        matches = self.instance(modality).query(payload, threshold=threshold, max_matches=max_matches)
        return [
            match.to_packet(
                source_ref=source_ref,
                modality=modality,
                importance=importance,
                timestamp=timestamp,
                query_meta={"query_payload_kind": str(payload.get("kind") or "")},
            )
            for match in matches
        ]

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "instances": {modality: inst.to_snapshot() for modality, inst in self._instances.items()},
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any] | None) -> "SensoryFossilStore":
        store = cls()
        if not isinstance(data, Mapping):
            return store
        instances = data.get("instances", {})
        if isinstance(instances, Mapping):
            for modality, snapshot in instances.items():
                if isinstance(snapshot, Mapping):
                    inst = SensoryFossilInstance.from_snapshot(snapshot)
                    store._instances[normalize_modality(modality)] = inst
        return store


__all__ = [
    "EvidencePacket",
    "FossilMatch",
    "FossilRecord",
    "SensoryFossilInstance",
    "SensoryFossilStore",
    "candidate_with_uncertainty",
    "clamp01",
    "color_similarity",
    "feature_vector_from_payload",
    "normalize_candidate",
    "normalize_tags",
    "vector_similarity",
]
