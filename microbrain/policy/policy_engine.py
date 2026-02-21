# microbrain/policy/policy_engine.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from microbrain.orchestrator.neuron_base import Event

logger = logging.getLogger("microbrain.policy")

DecisionStatus = Literal["allow", "veto", "needs_review"]


@dataclass(frozen=True)
class PolicyDecision:
    status: DecisionStatus
    rule_id: str = ""
    reason: str = ""
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class Rule:
    id: str
    summary: str
    priority: int
    kind: Literal["hard_veto", "soft_gate"]
    jurisdiction_required: bool = False
    when_uncertain: str = "require_human_review"


class PolicyEngine:
    """
    Policy-as-data evaluation engine.

    Important design choice:
    - This engine does NOT pretend to know law.
    - It only:
      (a) enforces explicit hard vetoes, and
      (b) brakes (needs_review) when a rule requires jurisdiction and it's missing,
          OR when upstream has flagged an action as "legal_check" / "requires_confirmation".
    """

    def __init__(self, policy_path: Optional[Path] = None) -> None:
        self.policy_path = policy_path or (Path(__file__).resolve().parent / "no_go.json")
        self.jurisdiction: Dict[str, Any] = {"country": "US", "state": None, "locality": None}
        self.hard_veto: List[Rule] = []
        self.soft_gates: List[Rule] = []

        self.reload()

    def reload(self) -> None:
        if not self.policy_path.exists():
            # No policy file yet: fail open (but log), so dev builds don’t brick.
            logger.warning("Policy file missing, running with empty policy: %s", self.policy_path)
            self.hard_veto = []
            self.soft_gates = []
            return

        try:
            obj = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load policy JSON (%s): %s", self.policy_path, exc)
            # fail open
            self.hard_veto = []
            self.soft_gates = []
            return

        self.jurisdiction = obj.get("jurisdiction", self.jurisdiction)

        self.hard_veto = [self._parse_rule(x, "hard_veto") for x in obj.get("hard_veto", [])]
        self.soft_gates = [self._parse_rule(x, "soft_gate") for x in obj.get("soft_gates", [])]

        # Highest priority first
        self.hard_veto.sort(key=lambda r: r.priority, reverse=True)
        self.soft_gates.sort(key=lambda r: r.priority, reverse=True)

    def _parse_rule(self, raw: Dict[str, Any], kind: Literal["hard_veto", "soft_gate"]) -> Rule:
        return Rule(
            id=str(raw.get("id", "")).strip(),
            summary=str(raw.get("summary", "")).strip(),
            priority=int(raw.get("priority", 0)),
            kind=kind,
            jurisdiction_required=bool(raw.get("jurisdiction_required", False)),
            when_uncertain=str(raw.get("when_uncertain", "require_human_review")),
        )

    def jurisdiction_complete(self) -> bool:
        # “Complete enough” for now: country + state known.
        # You can tighten later (e.g., require locality) depending on use-case.
        j = self.jurisdiction or {}
        return bool(j.get("country")) and bool(j.get("state"))

    def describe(self) -> Dict[str, Any]:
        return {
            "policy_path": str(self.policy_path),
            "jurisdiction": self.jurisdiction,
            "hard_veto": [r.id for r in self.hard_veto],
            "soft_gates": [r.id for r in self.soft_gates],
        }

    def should_check_event(self, event: Event) -> bool:
        """
        Only check things that *look like actions*, unless explicitly flagged.

        Current system mostly uses act/speech; we avoid gating speech unless
        upstream explicitly marks it.
        """
        meta = event.meta or {}
        if meta.get("policy:force_check", False):
            return True

        if event.topic == "act/speech":
            # Speech is allowed by default; only check if flagged.
            return bool(meta.get("policy:flags") or meta.get("policy:legal_check") or meta.get("policy:requires_confirmation"))

        return event.topic.startswith("act/")

    def evaluate_event(self, event: Event) -> PolicyDecision:
        """
        Evaluate an event against loaded policy.
        Fail-open unless explicitly flagged or clearly violates a hard veto.
        """
        if not self.should_check_event(event):
            return PolicyDecision(status="allow")

        meta = event.meta or {}

        # Explicit hard veto request from upstream (fast lane)
        explicit = meta.get("policy:hard_veto")
        if isinstance(explicit, str) and explicit.strip():
            rid = explicit.strip()
            return PolicyDecision(status="veto", rule_id=rid, reason="explicit hard veto", priority=999)

        flags: Set[str] = set()
        raw_flags = meta.get("policy:flags", [])
        if isinstance(raw_flags, list):
            flags |= {str(x).strip() for x in raw_flags if str(x).strip()}
        elif isinstance(raw_flags, str) and raw_flags.strip():
            flags.add(raw_flags.strip())

        # Hard veto by rule id match in flags
        for r in self.hard_veto:
            if r.id and r.id in flags:
                return PolicyDecision(status="veto", rule_id=r.id, reason=r.summary, priority=r.priority)

        # Soft gates: law compliance and confirmation
        # These are triggered by explicit meta flags so we don't brick normal operation.
        if meta.get("policy:legal_check", False) or meta.get("policy:requires_jurisdiction", False):
            # Find the law gate rule if present
            law_rule = next((x for x in self.soft_gates if x.id == "comply_with_applicable_law"), None)
            if law_rule and law_rule.jurisdiction_required and not self.jurisdiction_complete():
                return PolicyDecision(
                    status="needs_review",
                    rule_id=law_rule.id,
                    reason="jurisdiction missing; require human review",
                    priority=law_rule.priority,
                )

        if meta.get("policy:requires_confirmation", False):
            conf_rule = next((x for x in self.soft_gates if x.id == "requires_explicit_confirmation"), None)
            if conf_rule:
                return PolicyDecision(
                    status="needs_review",
                    rule_id=conf_rule.id,
                    reason="requires explicit confirmation",
                    priority=conf_rule.priority,
                )

        # Default: allow
        return PolicyDecision(status="allow")