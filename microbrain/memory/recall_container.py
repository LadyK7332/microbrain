from __future__ import annotations

"""Bounded recall routing for answer/probe style memory lookups.

Normal cognition should start from an activated concept and a routed memory
organ, not from a full-memory sweep.  This module keeps that contract small and
serializable so neurons, tools, and dashboard widgets can inspect it without
loading memory themselves.
"""

import hashlib
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

DEFAULT_CANDIDATE_LIMIT = 100
DEFAULT_ACTIVE_LIMIT = 15

ROUTE_TIERS: dict[str, tuple[str, ...]] = {
    "surface": ("hot", "now", "short", "learned"),
    "object": ("hot", "now", "short", "long", "learned"),
    "event": ("hot", "now", "short", "long"),
    "scene": ("hot", "now", "short", "long"),
    "pattern": ("learned", "long", "hot", "now"),
    "tool": ("learned", "long", "short"),
    "general": ("hot", "now", "short", "long", "learned"),
}

ROUTE_ORGANS: dict[str, tuple[str, ...]] = {
    "surface": ("surface_memory", "social_memory", "event_memory"),
    "object": ("object_memory", "visual_index", "touch_index", "social_memory"),
    "event": ("event_memory", "pattern_memory", "scene_memory"),
    "scene": ("scene_memory", "event_memory", "object_memory"),
    "pattern": ("pattern_memory", "event_memory"),
    "tool": ("tool_memory", "pattern_memory", "event_memory"),
    "general": ("event_memory", "pattern_memory", "object_memory", "surface_memory"),
}

SURFACE_ACTS = {"acknowledge", "continue", "respond", "reflect", "answer"}
OBJECT_WORDS = {"object", "thing", "person", "apple", "this", "that", "it", "vobj"}
EVENT_WORDS = {"happened", "happen", "when", "before", "after", "then", "changed", "event"}
SCENE_WORDS = {"scene", "room", "where", "behind", "nearby", "focus", "background"}
PATTERN_WORDS = {"pattern", "again", "keeps", "usually", "always", "why", "because"}
TOOL_WORDS = {"how", "tool", "fix", "build", "make", "run", "command"}


def _clean(value: Any, limit: int = 220) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())[:limit]


def _tokens(value: Any) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(value or ""))]


def _stable_id(parts: Sequence[Any], prefix: str = "recall") -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"


def classify_recall_route(
    query_text: Any,
    *,
    statement_kind: Any = "",
    required_act: Any = "",
) -> str:
    """Choose the first memory organ group for a thought/query."""

    text = _clean(query_text).lower()
    tokens = set(_tokens(text))
    kind = _clean(statement_kind, 80).lower()
    act = _clean(required_act, 80).lower()

    if act in SURFACE_ACTS or kind in {"greeting", "agreement", "closure", "personal_state"}:
        return "surface"
    if kind in {"question", "claim"} and (tokens & PATTERN_WORDS):
        return "pattern"
    if "what is" in text or "this is" in text or tokens & OBJECT_WORDS:
        return "object"
    if tokens & SCENE_WORDS:
        return "scene"
    if tokens & TOOL_WORDS:
        return "tool"
    if tokens & EVENT_WORDS:
        return "event"
    return "general"


def build_recall_plan(
    query_text: Any,
    *,
    statement_kind: Any = "",
    required_act: Any = "",
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    active_limit: int = DEFAULT_ACTIVE_LIMIT,
) -> dict[str, Any]:
    """Return a serializable bounded recall plan."""

    route = classify_recall_route(query_text, statement_kind=statement_kind, required_act=required_act)
    active = max(1, min(int(active_limit or DEFAULT_ACTIVE_LIMIT), DEFAULT_ACTIVE_LIMIT))
    candidates = max(active, min(int(candidate_limit or DEFAULT_CANDIDATE_LIMIT), DEFAULT_CANDIDATE_LIMIT))
    return {
        "schema": "memory.recall_plan.v1",
        "query_id": _stable_id([query_text, statement_kind, required_act, route], "recall"),
        "query_text": _clean(query_text, 260),
        "route": route,
        "organs": list(ROUTE_ORGANS.get(route, ROUTE_ORGANS["general"])),
        "tiers": list(ROUTE_TIERS.get(route, ROUTE_TIERS["general"])),
        "candidate_limit": candidates,
        "active_limit": active,
        "lookup_policy": "activation_route_then_bounded_candidates",
        "full_sweep_allowed": False,
        "ts": time.time(),
    }


def infer_hit_organ(hit: Mapping[str, Any]) -> str:
    kind = _clean(hit.get("kind"), 120).lower()
    meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
    anchor = hit.get("anchor") if isinstance(hit.get("anchor"), Mapping) else {}
    anchor_kind = _clean(anchor.get("kind"), 120).lower()
    text = " ".join([kind, anchor_kind, _clean(anchor.get("ref"), 160).lower()])

    if "surface" in text or kind in {"utterance_anchor", "trainer_alignment"}:
        return "surface_memory"
    if "object" in text or "vobj" in text or "visual" in text:
        return "object_memory"
    if "scene" in text:
        return "scene_memory"
    if "tool" in text or "procedure" in text or "command" in text:
        return "tool_memory"
    if "pattern" in text or "template" in text or "clause" in text or "relation" in text:
        return "pattern_memory"
    if meta.get("role") or "episode" in text or "event" in text:
        return "event_memory"
    return "general_memory"


def build_recall_container(
    plan: Mapping[str, Any],
    hits: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Store a bounded candidate set and a smaller active evidence index."""

    candidate_limit = max(1, int(plan.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT) or DEFAULT_CANDIDATE_LIMIT))
    active_limit = max(1, int(plan.get("active_limit", DEFAULT_ACTIVE_LIMIT) or DEFAULT_ACTIVE_LIMIT))
    organs = set(str(item) for item in list(plan.get("organs", []) or []))

    candidates: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, Mapping):
            continue
        row = dict(hit)
        organ = infer_hit_organ(row)
        row["memory_organ"] = organ
        row["route_match"] = organ in organs or organ == "general_memory"
        candidates.append(row)
        if len(candidates) >= candidate_limit:
            break

    candidates.sort(
        key=lambda item: (
            bool(item.get("route_match", False)),
            float(item.get("score", 0.0) or 0.0),
            float(item.get("last_seen", item.get("ts", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    active = candidates[:active_limit]
    last = candidates[-1] if candidates else {}
    return {
        "schema": "memory.recall_container.v1",
        "query_id": str(plan.get("query_id", "") or ""),
        "route": str(plan.get("route", "general") or "general"),
        "organs": list(plan.get("organs", []) or []),
        "tiers": list(plan.get("tiers", []) or []),
        "candidate_limit": candidate_limit,
        "active_limit": active_limit,
        "checked_budget": candidate_limit,
        "candidate_count": len(candidates),
        "active_count": len(active),
        "candidates": candidates,
        "active_evidence": active,
        "bookmark": {
            "last_cell_id": str(last.get("cell_id", "") or ""),
            "last_tier": str(last.get("tier", "") or ""),
            "last_score": float(last.get("score", 0.0) or 0.0),
            "exhausted": len(candidates) < candidate_limit,
        },
        "mutation": False,
        "ts": time.time(),
    }
