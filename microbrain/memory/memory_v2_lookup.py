from __future__ import annotations

"""Small read-side helpers for Memory v2.

The v2 writer keeps language evidence in segmented JSONL shelves.  This module
does bounded shelf browsing for responders without reopening mem_cell v1.
"""

import json
import time
from pathlib import Path
from typing import Any, Mapping

from microbrain.memory.segmented_language_memory import norm_text, tokens_for
from microbrain.utils.memdir import resolve_memdir_ctx

QUESTION_STOPWORDS = {
    "what", "is", "are", "was", "were", "a", "an", "the", "do", "does", "did",
    "to", "of", "for", "about", "can", "you", "me", "tell", "give", "explain",
    "why", "how", "when", "where", "who", "which",
}


def empty_syntax_guidance() -> dict[str, Any]:
    return {"preferred_replies": [], "avoid_replies": [], "ddna_targets": {}, "classifiers": []}


def merge_syntax_guidance(*items: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = empty_syntax_guidance()
    seen_reply: set[str] = set()
    seen_avoid: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for reply in list(item.get("preferred_replies", []) or []):
            if not isinstance(reply, Mapping):
                continue
            text = str(reply.get("text", "") or "").strip()
            key = norm_text(text)
            if text and key and key not in seen_reply:
                seen_reply.add(key)
                merged["preferred_replies"].append(dict(reply))
        for avoid in list(item.get("avoid_replies", []) or []):
            text = str(avoid or "").strip()
            key = norm_text(text)
            if text and key and key not in seen_avoid:
                seen_avoid.add(key)
                merged["avoid_replies"].append(text)
        for key, value in dict(item.get("ddna_targets", {}) or {}).items():
            try:
                merged["ddna_targets"][str(key)] = max(float(merged["ddna_targets"].get(str(key), 0.0)), float(value))
            except Exception:
                continue
        for classifier in list(item.get("classifiers", []) or []):
            name = norm_text(classifier).replace(" ", "_")
            if name and name not in merged["classifiers"]:
                merged["classifiers"].append(name)
    return merged


class MemoryV2Lookup:
    def __init__(self, memdir: str | Path):
        self.memdir = Path(memdir)
        self.base_dir = self.memdir / "memory_v2"

    @property
    def phrases_path(self) -> Path:
        return self.base_dir / "tok_graph" / "phrase_patterns.jsonl"

    @property
    def corrections_path(self) -> Path:
        return self.base_dir / "teaching_memory" / "corrections.jsonl"

    def _read_jsonl(self, path: Path, *, max_rows: int = 5000) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
                    if len(rows) >= max_rows:
                        break
        except Exception:
            return []
        return rows

    def _focus_tokens(self, text: str) -> list[str]:
        toks = tokens_for(text)
        focus = [tok for tok in toks if tok not in QUESTION_STOPWORDS]
        return focus or toks

    def _score_text(self, query_tokens: set[str], text: str) -> float:
        toks = set(tokens_for(text))
        if not query_tokens or not toks:
            return 0.0
        overlap = len(query_tokens & toks)
        if overlap <= 0:
            return 0.0
        return overlap / max(1.0, float(len(query_tokens | toks)))

    def search(self, query_text: str, *, limit: int = 15) -> list[dict[str, Any]]:
        query = norm_text(query_text)
        q_tokens = set(self._focus_tokens(query))
        if not query or not q_tokens:
            return []
        hits: list[dict[str, Any]] = []
        for row in self._read_jsonl(self.phrases_path):
            surface = str(row.get("surface", "") or "")
            slots = row.get("slots", {}) if isinstance(row.get("slots", {}), Mapping) else {}
            slot_text = " ".join(str(slots.get(key, "") or "") for key in ("subject", "predicate", "designation", "query_target"))
            score = max(self._score_text(q_tokens, surface), self._score_text(q_tokens, slot_text))
            if norm_text(surface) == query:
                score += 0.35
            if score <= 0.0:
                continue
            hits.append({
                "cell_id": str(row.get("pattern_id", "") or ""),
                "kind": "memory_v2_phrase",
                "tier": "memory_v2",
                "score": round(min(1.0, score + float(row.get("confidence", 0.0) or 0.0) * 0.20), 6),
                "anchor": {"kind": "language/phrase_pattern", "ref": surface, "norm": norm_text(surface)},
                "anchor_text": surface,
                "refs": [surface],
                "meta": {"memory_v2": row, "slots": dict(slots), "role": str(row.get("last_role", "") or "")},
            })
        hits.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return hits[: max(1, int(limit))]

    def answer_question(self, query_text: str) -> dict[str, Any]:
        focus = self._focus_tokens(query_text)
        target = " ".join(focus[-3:]).strip()
        if not target:
            return {"ok": False, "reason": "no_target", "text": ""}
        target_norm = norm_text(target)
        best: dict[str, Any] | None = None
        best_score = 0.0
        for row in self._read_jsonl(self.phrases_path):
            slots = row.get("slots", {}) if isinstance(row.get("slots", {}), Mapping) else {}
            subject = norm_text(slots.get("subject", ""))
            predicate = norm_text(slots.get("predicate", "") or slots.get("designation", ""))
            if not subject or not predicate:
                continue
            score = 0.0
            if subject == target_norm:
                score = 0.92
            elif target_norm in subject or subject in target_norm:
                score = 0.68
            else:
                score = self._score_text(set(tokens_for(target_norm)), subject)
            score += min(0.08, float(row.get("confidence", 0.0) or 0.0) * 0.08)
            if score > best_score:
                best_score = score
                best = row
        if not best or best_score < 0.42:
            return {"ok": False, "reason": "no_v2_answer", "text": "", "target": target}
        slots = best.get("slots", {}) if isinstance(best.get("slots", {}), Mapping) else {}
        subject = str(slots.get("subject", target) or target).strip()
        predicate = str(slots.get("predicate", "") or slots.get("designation", "") or "").strip()
        text = f"{subject} is {predicate}.".strip()
        return {
            "ok": bool(predicate),
            "reason": "memory_v2_phrase_slot",
            "text": text[:1].upper() + text[1:] if text else "",
            "confidence": round(min(1.0, best_score), 6),
            "selected_cell_ids": [str(best.get("pattern_id", "") or "")],
            "hit": best,
            "target": target,
        }

    def syntax_guidance(self, lookup_text: str) -> dict[str, Any]:
        lookup_norm = norm_text(lookup_text)
        q_tokens = set(tokens_for(lookup_norm))
        guidance = empty_syntax_guidance()
        if not lookup_norm:
            return guidance
        for row in self._read_jsonl(self.corrections_path):
            context = str(row.get("context", "") or "")
            bad = str(row.get("bad_utterance", "") or "")
            desired = str(row.get("desired", "") or "")
            if not desired:
                continue
            score = max(self._score_text(q_tokens, context), self._score_text(q_tokens, desired))
            if bad and norm_text(bad) == lookup_norm:
                score += 0.45
            if context and norm_text(context) == lookup_norm:
                score += 0.30
            if score <= 0.0 and bad:
                score = self._score_text(q_tokens, bad) * 0.8
            if score <= 0.0:
                continue
            meta = {
                "kind": "memory_v2_trainer_correction",
                "desired_utterance": desired,
                "bad_utterance": bad,
                "trainer_context": context,
                "trainer_need": str(row.get("need", "") or ""),
                "trainer_style": str(row.get("style", "") or ""),
                "memory_v2": row,
            }
            guidance["preferred_replies"].append({"text": desired, "score": round(min(1.0, score + 0.35), 6), "meta": meta})
            if bad:
                guidance["avoid_replies"].append(bad)
            guidance["classifiers"].append("trainer_correction")
        return guidance


async def get_memory_v2_lookup(ctx: Any) -> MemoryV2Lookup | None:
    lookup = await ctx.get_kv("memory:v2_lookup", None)
    if isinstance(lookup, MemoryV2Lookup):
        return lookup
    try:
        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        lookup = MemoryV2Lookup(memdir)
        await ctx.set_kv("memory:v2_lookup", lookup)
        return lookup
    except Exception:
        return None


async def syntax_guidance_from_memory_v2(ctx: Any, lookup_text: str) -> dict[str, Any]:
    lookup = await get_memory_v2_lookup(ctx)
    if lookup is None:
        return empty_syntax_guidance()
    return lookup.syntax_guidance(lookup_text)
