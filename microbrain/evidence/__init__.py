"""Evidence artifacts and compact evidence cards for MicroBrain."""

from microbrain.evidence.artifact_store import (
    EvidenceArtifactStore,
    EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA,
    EVIDENCE_MULTIMODAL_REF_PACK_SCHEMA,
    EVIDENCE_REF_INDEX_SCHEMA,
    EVIDENCE_REF_PACK_SCHEMA,
    MULTIMODAL_INDEX_MODALITY,
    MAX_INLINE_EVIDENCE_REFS,
)
from microbrain.evidence.evidence_card import build_evidence_card, evidence_ref_card

__all__ = [
    "EvidenceArtifactStore",
    "EVIDENCE_MULTIMODAL_REF_INDEX_SCHEMA",
    "EVIDENCE_MULTIMODAL_REF_PACK_SCHEMA",
    "EVIDENCE_REF_INDEX_SCHEMA",
    "EVIDENCE_REF_PACK_SCHEMA",
    "MULTIMODAL_INDEX_MODALITY",
    "MAX_INLINE_EVIDENCE_REFS",
    "build_evidence_card",
    "evidence_ref_card",
]
