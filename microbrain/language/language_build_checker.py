from __future__ import annotations

"""Bounded language construction parser and rule checker.

This helper is deliberately not a neuron.  Speech/reasoning organs call it when
they are about to build or emit text, and it inspects only the small token/edge
shelves needed for the current utterance.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from microbrain.memory.segmented_language_memory import norm_text, stable_id, tokens_for
from microbrain.utils.memdir import resolve_memdir_ctx

DEFAULT_MAX_TOKENS = 32
DEFAULT_MAX_EDGE_ROWS = 25_000
DEFAULT_MAX_PHRASE_ROWS = 8_000
TEMP_REJECT_KV_KEY = "language_build:temp_rejects"
TEMP_REJECT_LIMIT = 64


def iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def reject_bucket_key(lookup_text: str) -> str:
    key = norm_text(lookup_text)[:180]
    return key or "__global__"


class LanguageBuildChecker:
    def __init__(
        self,
        memdir: str | Path,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_edge_rows: int = DEFAULT_MAX_EDGE_ROWS,
        max_phrase_rows: int = DEFAULT_MAX_PHRASE_ROWS,
    ) -> None:
        self.memdir = Path(memdir)
        self.base_dir = self.memdir / "memory_v2"
        self.max_tokens = max(4, int(max_tokens or DEFAULT_MAX_TOKENS))
        self.max_edge_rows = max(100, int(max_edge_rows or DEFAULT_MAX_EDGE_ROWS))
        self.max_phrase_rows = max(100, int(max_phrase_rows or DEFAULT_MAX_PHRASE_ROWS))

    @property
    def adjacency_path(self) -> Path:
        return self.base_dir / "tok_graph" / "adjacency.jsonl"

    @property
    def phrases_path(self) -> Path:
        return self.base_dir / "tok_graph" / "phrase_patterns.jsonl"

    def build_plan(self, text: str) -> dict[str, Any]:
        surface = norm_text(text)
        toks = tokens_for(surface)
        tokids = [stable_id("tok", tok, size=8) for tok in toks[: self.max_tokens]]
        pairs = list(zip(toks, toks[1:]))[: max(0, self.max_tokens - 1)]
        edge_support = self._edge_support(pairs)
        phrase_support = self._phrase_support(surface, toks)
        edge_count = len(pairs)
        supported_edges = sum(1 for item in edge_support if item.get("seen"))
        adjacency_ratio = supported_edges / max(1, edge_count) if edge_count else 1.0
        repeated_ratio = self._repeat_ratio(toks)
        stopword_ratio = self._stopword_ratio(toks)
        complexity = self._complexity(toks)
        return {
            "schema": "language.build_plan.v1",
            "surface": surface,
            "token_text": toks[: self.max_tokens],
            "tok_chain": tokids,
            "truncated": len(toks) > self.max_tokens,
            "edges": edge_support,
            "edge_count": edge_count,
            "supported_edges": supported_edges,
            "adjacency_ratio": round(adjacency_ratio, 6),
            "phrase_support": phrase_support,
            "repeat_ratio": round(repeated_ratio, 6),
            "stopword_ratio": round(stopword_ratio, 6),
            "complexity": complexity,
        }

    def check(self, text: str, *, strict: bool = True, source: str = "") -> dict[str, Any]:
        plan = self.build_plan(text)
        toks = list(plan.get("token_text", []) or [])
        flags: list[str] = []
        if not toks:
            flags.append("no_tokens")
        if bool(plan.get("truncated", False)):
            flags.append("too_many_tokens")
        if float(plan.get("repeat_ratio", 0.0) or 0.0) >= 0.34 and len(toks) >= 4:
            flags.append("token_loop")
        if float(plan.get("stopword_ratio", 0.0) or 0.0) >= 0.72 and len(toks) >= 5:
            flags.append("mostly_glue_words")

        exact_phrase = bool((plan.get("phrase_support", {}) or {}).get("exact"))
        adjacency_ratio = float(plan.get("adjacency_ratio", 0.0) or 0.0)
        if len(toks) >= 4 and adjacency_ratio < (0.28 if strict else 0.18) and not exact_phrase:
            flags.append("weak_adjacency")

        score = 0.25
        score += min(0.42, adjacency_ratio * 0.42)
        if exact_phrase:
            score += 0.30
        elif (plan.get("phrase_support", {}) or {}).get("near"):
            score += 0.12
        score -= min(0.22, float(plan.get("repeat_ratio", 0.0) or 0.0) * 0.42)
        if bool(plan.get("truncated", False)):
            score -= 0.20
        score = max(0.0, min(1.0, score))

        hard_flags = {"no_tokens", "too_many_tokens", "token_loop"}
        weak_flags = {"mostly_glue_words", "weak_adjacency"}
        blocked = bool(hard_flags & set(flags))
        if strict and score < 0.34 and bool(weak_flags & set(flags)):
            blocked = True

        return {
            "schema": "language.build_check.v1",
            "ok": not blocked,
            "blocked": blocked,
            "score": round(score, 6),
            "flags": flags,
            "source": str(source or ""),
            "plan": plan,
            "why": self._why(flags, score, exact_phrase, adjacency_ratio),
        }

    def _edge_support(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        wanted = {stable_id("edge", left, right, "next", size=8): (left, right) for left, right in pairs}
        found: dict[str, Mapping[str, Any]] = {}
        if wanted and self.adjacency_path.exists():
            for idx, row in enumerate(iter_jsonl(self.adjacency_path)):
                if idx >= self.max_edge_rows or len(found) >= len(wanted):
                    break
                edge_id = str(row.get("edge_id", "") or "")
                if edge_id in wanted:
                    found[edge_id] = row
        out: list[dict[str, Any]] = []
        for left, right in pairs:
            edge_id = stable_id("edge", left, right, "next", size=8)
            row = found.get(edge_id, {})
            out.append({
                "edge_id": edge_id,
                "left": left,
                "right": right,
                "seen": bool(row),
                "confidence": float(row.get("confidence", 0.0) or 0.0) if isinstance(row, Mapping) else 0.0,
                "use_count": int(row.get("use_count", 0) or 0) if isinstance(row, Mapping) else 0,
            })
        return out

    def _phrase_support(self, surface: str, toks: Sequence[str]) -> dict[str, Any]:
        if not surface or not self.phrases_path.exists():
            return {"exact": False, "near": []}
        focus = set(toks)
        near: list[dict[str, Any]] = []
        exact: dict[str, Any] | None = None
        for idx, row in enumerate(iter_jsonl(self.phrases_path)):
            if idx >= self.max_phrase_rows:
                break
            row_surface = norm_text(row.get("surface", ""))
            if not row_surface:
                continue
            if row_surface == surface:
                exact = {"pattern_id": str(row.get("pattern_id", "") or ""), "confidence": float(row.get("confidence", 0.0) or 0.0)}
                break
            row_toks = set(tokens_for(row_surface))
            if focus and len(focus & row_toks) >= min(3, len(focus)):
                near.append({
                    "pattern_id": str(row.get("pattern_id", "") or ""),
                    "surface": row_surface,
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                })
                if len(near) >= 3:
                    break
        return {"exact": bool(exact), "exact_hit": exact or {}, "near": near}

    @staticmethod
    def _repeat_ratio(toks: Sequence[str]) -> float:
        if not toks:
            return 0.0
        return 1.0 - (len(set(toks)) / max(1, len(toks)))

    @staticmethod
    def _stopword_ratio(toks: Sequence[str]) -> float:
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "to", "of", "for", "and",
            "or", "but", "in", "on", "at", "with", "you", "i", "me", "it", "this",
            "that", "do", "does", "did",
        }
        if not toks:
            return 0.0
        return sum(1 for tok in toks if tok in stop) / max(1, len(toks))

    @staticmethod
    def _complexity(toks: Sequence[str]) -> dict[str, Any]:
        return {
            "tokens": len(toks),
            "unique_tokens": len(set(toks)),
            "has_copula": any(tok in {"is", "are", "was", "were"} for tok in toks),
            "has_question_word": any(tok in {"what", "why", "how", "when", "where", "who", "which"} for tok in toks),
        }

    @staticmethod
    def _why(flags: Sequence[str], score: float, exact_phrase: bool, adjacency_ratio: float) -> str:
        if "token_loop" in flags:
            return "candidate repeats itself like a loop"
        if "too_many_tokens" in flags:
            return "candidate is longer than the speech build budget"
        if "weak_adjacency" in flags:
            return "candidate edges are weak or unseen in the v2 token graph"
        if exact_phrase:
            return "candidate matches a known phrase pattern"
        return f"candidate adjacency support {adjacency_ratio:.2f}, score {score:.2f}"


async def candidate_was_rejected(ctx: Any, text: str, *, lookup_text: str = "") -> bool:
    rejects = await ctx.get_kv(TEMP_REJECT_KV_KEY, {})
    if not isinstance(rejects, Mapping):
        return False
    bucket = rejects.get(reject_bucket_key(lookup_text), [])
    if not isinstance(bucket, list):
        return False
    return norm_text(text) in {str(item or "") for item in bucket}


async def remember_language_failure(ctx: Any, text: str, *, lookup_text: str = "") -> None:
    key = reject_bucket_key(lookup_text)
    reject_norm = norm_text(text)
    if not reject_norm:
        return
    raw = await ctx.get_kv(TEMP_REJECT_KV_KEY, {})
    rejects = dict(raw or {}) if isinstance(raw, Mapping) else {}
    bucket = [str(item or "") for item in list(rejects.get(key, []) or []) if str(item or "")]
    if reject_norm not in bucket:
        bucket.append(reject_norm)
    rejects[key] = bucket[-TEMP_REJECT_LIMIT:]
    await ctx.set_kv(TEMP_REJECT_KV_KEY, rejects)


async def clear_language_failures(ctx: Any, *, lookup_text: str = "") -> None:
    raw = await ctx.get_kv(TEMP_REJECT_KV_KEY, {})
    if not isinstance(raw, Mapping):
        return
    rejects = dict(raw)
    key = reject_bucket_key(lookup_text)
    if key in rejects:
        rejects.pop(key, None)
        await ctx.set_kv(TEMP_REJECT_KV_KEY, rejects)


async def check_language_candidate(
    ctx: Any,
    text: str,
    *,
    strict: bool = True,
    source: str = "",
    lookup_text: str = "",
) -> dict[str, Any]:
    try:
        if await candidate_was_rejected(ctx, text, lookup_text=lookup_text):
            result = {
                "schema": "language.build_check.v1",
                "ok": False,
                "blocked": True,
                "score": 0.0,
                "flags": ["temp_reject_prior"],
                "source": str(source or ""),
                "why": "candidate already failed for this lookup",
                "plan": {},
            }
            await ctx.set_kv("language_build:last_check", result)
            return result
        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        max_tokens = int(await ctx.get_kv("language_build:max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_MAX_TOKENS)
        max_edge_rows = int(await ctx.get_kv("language_build:max_edge_rows", DEFAULT_MAX_EDGE_ROWS) or DEFAULT_MAX_EDGE_ROWS)
        max_phrase_rows = int(await ctx.get_kv("language_build:max_phrase_rows", DEFAULT_MAX_PHRASE_ROWS) or DEFAULT_MAX_PHRASE_ROWS)
        checker = LanguageBuildChecker(
            memdir,
            max_tokens=max_tokens,
            max_edge_rows=max_edge_rows,
            max_phrase_rows=max_phrase_rows,
        )
        result = checker.check(text, strict=strict, source=source)
        await ctx.set_kv("language_build:last_check", result)
        if bool(result.get("blocked", False)):
            await remember_language_failure(ctx, text, lookup_text=lookup_text)
        return result
    except Exception as exc:
        return {
            "schema": "language.build_check.v1",
            "ok": True,
            "blocked": False,
            "score": 0.0,
            "flags": ["checker_unavailable"],
            "source": str(source or ""),
            "why": repr(exc),
            "plan": {},
        }
