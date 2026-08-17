from __future__ import annotations

"""
Evidence Convergence Organ primitives.

The convergence layer merges evidence packets into workspace candidates.  It is
not allowed to turn a single fossil match into truth.  It may create accepted
working beliefs when confidence is high enough, and it may raise anomaly events
when recent high-confidence beliefs are contradicted by action feedback.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from microbrain.sensory_fossils import (
    EvidencePacket,
    candidate_with_uncertainty,
    clamp01,
    normalize_candidate,
    normalize_tags,
)

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

DEFAULT_CONVERGENCE_WINDOW_S = 2.5
DEFAULT_CANDIDATE_THRESHOLD = 0.62
DEFAULT_ACCEPTED_BELIEF_THRESHOLD = 0.82
DEFAULT_ANOMALY_THRESHOLD = 0.68
RECENT_ACTION_WINDOW_S = 600.0
BODY_RELEVANT_ACTIONS = frozenset({"drink", "taste", "eat", "ingest", "touch_hot", "grasp", "press"})
MAINTENANCE_RELEVANT_TAGS = frozenset({"#maintenance", "#ingestion", "#safety", "#body", "#fluid", "#heat"})


@dataclass
class WorkspaceCandidate:
    candidate_id: str
    candidate: str
    target_refs: list[str]
    confidence: float
    importance: float
    modalities: list[str]
    supports: list[str] = field(default_factory=list)
    missing_checks: list[str] = field(default_factory=list)
    fossil_refs: list[str] = field(default_factory=list)
    mem_cell_tags: list[str] = field(default_factory=list)
    evidence_packet_ids: list[str] = field(default_factory=list)
    recommended_next: str = "watch"
    accepted_working_belief: bool = False
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = clamp01(data.get("confidence"))
        data["importance"] = clamp01(data.get("importance"))
        data["mem_cell_tags"] = normalize_tags(data.get("mem_cell_tags"))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkspaceCandidate":
        payload = dict(data or {})
        payload.setdefault("candidate_id", _candidate_id(payload.get("target_refs", []), payload.get("candidate", "")))
        payload.setdefault("target_refs", [])
        payload.setdefault("modalities", [])
        payload.setdefault("supports", [])
        payload.setdefault("missing_checks", [])
        payload.setdefault("fossil_refs", [])
        payload.setdefault("mem_cell_tags", [])
        payload.setdefault("evidence_packet_ids", [])
        payload.setdefault("timestamp", time.time())
        payload.setdefault("meta", {})
        payload["confidence"] = clamp01(payload.get("confidence"))
        payload["importance"] = clamp01(payload.get("importance"))
        payload["mem_cell_tags"] = normalize_tags(payload.get("mem_cell_tags"))
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__.keys() if k in payload})


@dataclass
class WorkingBelief:
    subject_ref: str
    believed_as: str
    confidence: float
    evidence_packet_ids: list[str] = field(default_factory=list)
    fossil_refs: list[str] = field(default_factory=list)
    mem_cell_tags: list[str] = field(default_factory=list)
    expected_results: dict[str, Any] = field(default_factory=dict)
    accepted_at: float = field(default_factory=time.time)
    source: str = "evidence_convergence"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = clamp01(data.get("confidence"))
        data["mem_cell_tags"] = normalize_tags(data.get("mem_cell_tags"))
        return data

    @classmethod
    def from_candidate(cls, candidate: WorkspaceCandidate) -> "WorkingBelief":
        return cls(
            subject_ref=str(candidate.target_refs[0] if candidate.target_refs else ""),
            believed_as=normalize_candidate(candidate.candidate),
            confidence=candidate.confidence,
            evidence_packet_ids=list(candidate.evidence_packet_ids),
            fossil_refs=list(candidate.fossil_refs),
            mem_cell_tags=list(candidate.mem_cell_tags),
            expected_results=dict(candidate.meta.get("expected_results", {}) or {}),
            accepted_at=candidate.timestamp,
            meta={
                "candidate_id": candidate.candidate_id,
                "modalities": list(candidate.modalities),
                "supports": list(candidate.supports),
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkingBelief":
        payload = dict(data or {})
        payload.setdefault("evidence_packet_ids", [])
        payload.setdefault("fossil_refs", [])
        payload.setdefault("mem_cell_tags", [])
        payload.setdefault("expected_results", {})
        payload.setdefault("accepted_at", time.time())
        payload.setdefault("source", "evidence_convergence")
        payload.setdefault("meta", {})
        payload["confidence"] = clamp01(payload.get("confidence"))
        payload["mem_cell_tags"] = normalize_tags(payload.get("mem_cell_tags"))
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__.keys() if k in payload})


@dataclass
class AnomalyEvent:
    subject_ref: str
    believed_as: str
    observed_as: str
    severity: float
    confidence_at_belief: float
    age_s: float
    action: str = ""
    reason: str = "belief_contradicted"
    questions: list[str] = field(default_factory=list)
    evidence_packet_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = clamp01(data.get("severity"))
        data["confidence_at_belief"] = clamp01(data.get("confidence_at_belief"))
        return data


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def _candidate_id(target_refs: Sequence[str], candidate: str) -> str:
    base = "|".join(sorted(str(x) for x in target_refs if str(x))) + "|" + normalize_candidate(candidate)
    import hashlib

    return "wkcand_" + hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()[:16]


def _packet_group_key(packet: EvidencePacket) -> tuple[str, str]:
    return (str(packet.source_ref or ""), normalize_candidate(packet.candidate))


def _unique(items: Iterable[Any], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _score_group(packets: Sequence[EvidencePacket]) -> tuple[float, float, list[str]]:
    if not packets:
        return 0.0, 0.0, []
    weights: list[float] = []
    scores: list[float] = []
    importances: list[float] = []
    modalities = _unique(p.modality for p in packets)
    for packet in packets:
        evidence_strength = (clamp01(packet.confidence) * 0.55) + (clamp01(packet.similarity) * 0.35) + (clamp01(packet.importance) * 0.10)
        weight = 1.0 + clamp01(packet.importance)
        scores.append(evidence_strength)
        weights.append(weight)
        importances.append(clamp01(packet.importance))
    weighted = sum(s * w for s, w in zip(scores, weights)) / max(0.001, sum(weights))
    diversity_bonus = min(0.12, max(0, len(modalities) - 1) * 0.06)
    agreement_bonus = min(0.06, max(0, len(packets) - 1) * 0.02)
    confidence = clamp01(weighted + diversity_bonus + agreement_bonus)
    importance = clamp01(max(importances or [0.0]) + (0.05 * len(modalities)))
    return confidence, importance, modalities


def converge_evidence_packets(
    packets: Iterable[EvidencePacket | Mapping[str, Any]],
    *,
    now_ts: float | None = None,
    window_s: float = DEFAULT_CONVERGENCE_WINDOW_S,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
    accepted_threshold: float = DEFAULT_ACCEPTED_BELIEF_THRESHOLD,
) -> list[WorkspaceCandidate]:
    """Merge evidence packets into workspace candidates.

    Evidence packets older than `window_s` relative to `now_ts` are ignored.
    Returned candidates are still revisable.  If confidence crosses the accepted
    threshold, the candidate is only an accepted working belief, not permanent truth.
    """
    now = now_ts or time.time()
    normalized: list[EvidencePacket] = []
    for item in packets:
        packet = item if isinstance(item, EvidencePacket) else EvidencePacket.from_dict(item)
        if window_s > 0 and (now - packet.timestamp) > window_s:
            continue
        if not packet.source_ref or not normalize_candidate(packet.candidate):
            continue
        normalized.append(packet)

    groups: dict[tuple[str, str], list[EvidencePacket]] = {}
    for packet in normalized:
        groups.setdefault(_packet_group_key(packet), []).append(packet)

    candidates: list[WorkspaceCandidate] = []
    for (source_ref, candidate_name), group in groups.items():
        confidence, importance, modalities = _score_group(group)
        if confidence < clamp01(candidate_threshold):
            continue
        accepted = confidence >= clamp01(accepted_threshold)
        candidate_label = candidate_with_uncertainty(candidate_name, confident=accepted)
        supports = _unique((support for packet in group for support in packet.supports), limit=16)
        uncertainty = _unique((u for packet in group for u in packet.uncertainty), limit=16)
        fossil_refs = _unique((ref for packet in group for ref in packet.fossil_refs), limit=20)
        tags = normalize_tags(tag for packet in group for tag in packet.mem_cell_tags)
        packet_ids = _unique((packet.packet_id for packet in group), limit=24)
        if accepted:
            recommended = "accept_working_belief"
        elif importance >= 0.70 or any("required" in u for u in uncertainty):
            recommended = "focus_or_ask"
        else:
            recommended = "watch"
        candidates.append(
            WorkspaceCandidate(
                candidate_id=_candidate_id([source_ref], candidate_name),
                candidate=candidate_label,
                target_refs=[source_ref],
                confidence=round(confidence, 4),
                importance=round(importance, 4),
                modalities=modalities,
                supports=supports,
                missing_checks=uncertainty,
                fossil_refs=fossil_refs,
                mem_cell_tags=tags,
                evidence_packet_ids=packet_ids,
                recommended_next=recommended,
                accepted_working_belief=accepted,
                timestamp=now,
                meta={
                    "evidence_count": len(group),
                    "candidate_threshold": candidate_threshold,
                    "accepted_threshold": accepted_threshold,
                    "belief_rule": "working_belief_not_truth",
                },
            )
        )

    candidates.sort(key=lambda c: (c.confidence, c.importance, len(c.modalities)), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Working belief contradiction / anomaly
# ---------------------------------------------------------------------------


def contradiction_anomaly(
    belief: WorkingBelief | Mapping[str, Any],
    feedback: Mapping[str, Any],
    *,
    now_ts: float | None = None,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
) -> AnomalyEvent | None:
    """Return an anomaly when a recent high-confidence belief is contradicted.

    Example: MB accepted "Bang can" as a working belief, drank from it, and
    taste/smell/liquid feedback says the result does not match.  This should not
    silently relabel; it should raise a question/anomaly event.
    """
    wb = belief if isinstance(belief, WorkingBelief) else WorkingBelief.from_dict(belief)
    now = now_ts or time.time()
    matches_expected = feedback.get("matches_expected", None)
    contradicted = bool(feedback.get("contradiction", False)) or matches_expected is False
    if not contradicted:
        return None

    subject = str(feedback.get("subject_ref") or wb.subject_ref or "")
    if wb.subject_ref and subject and subject != wb.subject_ref:
        return None

    age_s = max(0.0, now - float(wb.accepted_at or now))
    recency = clamp01(1.0 - (age_s / RECENT_ACTION_WINDOW_S), default=0.0)
    action = str(feedback.get("action") or feedback.get("action_type") or "").strip().lower()
    action_boost = 0.15 if action in BODY_RELEVANT_ACTIONS else 0.0

    feedback_tags = set(normalize_tags(feedback.get("tags", [])))
    belief_tags = set(normalize_tags(wb.mem_cell_tags))
    maintenance_boost = 0.12 if (feedback_tags | belief_tags) & MAINTENANCE_RELEVANT_TAGS else 0.0
    severity = clamp01((wb.confidence * 0.62) + (recency * 0.23) + action_boost + maintenance_boost)
    if severity < clamp01(threshold):
        return None

    observed_as = str(feedback.get("observed_as") or feedback.get("result") or feedback.get("observed") or "unexpected_result")
    questions = [
        f"Did I misidentify {wb.subject_ref or 'the object'} as {wb.believed_as}?",
        f"Did {wb.subject_ref or 'the object'} change since the belief was accepted?",
        "Is one of the sensory channels wrong or incomplete?",
    ]
    if action in {"drink", "taste", "eat", "ingest"}:
        questions.extend(
            [
                "Should I stop ingestion until this is resolved?",
                "Does the container contents fail to match the container identity?",
            ]
        )
    elif action:
        questions.append(f"Did action '{action}' produce a result that contradicts the object identity?")

    return AnomalyEvent(
        subject_ref=wb.subject_ref,
        believed_as=wb.believed_as,
        observed_as=observed_as,
        severity=round(severity, 4),
        confidence_at_belief=round(wb.confidence, 4),
        age_s=round(age_s, 3),
        action=action,
        questions=_unique(questions, limit=8),
        evidence_packet_ids=list(wb.evidence_packet_ids),
        timestamp=now,
        meta={
            "recency": round(recency, 4),
            "action_boost": action_boost,
            "maintenance_boost": maintenance_boost,
            "threshold": threshold,
            "rule": "high_confidence_recent_action_contradiction_demands_investigation",
        },
    )


__all__ = [
    "AnomalyEvent",
    "WorkingBelief",
    "WorkspaceCandidate",
    "contradiction_anomaly",
    "converge_evidence_packets",
]
