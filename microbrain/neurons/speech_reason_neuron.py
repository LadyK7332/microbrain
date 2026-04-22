from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.memory.memory_store import MemoryStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
TOKEN_RE = re.compile(r"[a-z0-9']+")


class SpeechReasonNeuron(BaseNeuron):
    """
    Intent-aware outward expression organ.

    Early behavior will still look a bit babbly because it reuses learned utterance
    shapes, but the utterance is always selected in service of a motive rather than
    free-floating chatter.
    """

    def _norm(self, text: str) -> str:
        return " ".join(TOKEN_RE.findall((text or "").lower())).strip()

    def _tokens(self, text: str) -> List[str]:
        return [tok for tok in TOKEN_RE.findall((text or "").lower()) if tok]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _clamp01(self, value: Any) -> float:
        return max(0.0, min(1.0, self._safe_float(value, 0.0)))

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip()

    def _looks_like_command(self, text: str) -> bool:
        t = str(text or "").strip().lower()
        return t.startswith("/")

    def _is_bad_candidate(self, text: str) -> bool:
        clean = self._clean_text(text)
        if not clean:
            return True
        lower = clean.lower()
        if self._looks_like_command(clean):
            return True
        if len(clean) < 3 or len(clean) > 160:
            return True
        if lower.startswith("available commands"):
            return True
        if lower.startswith("reinforcement snapshot"):
            return True
        if lower.startswith("usage:"):
            return True
        if lower.startswith("no active /r snapshot"):
            return True
        if "snapshot cleared" in lower:
            return True
        if lower.startswith("applied +") or lower.startswith("applied -"):
            return True
        return False

    def _need_lexicon(self, need: str) -> set[str]:
        lowered = str(need or "").lower()
        if lowered == "power":
            return {"power", "low", "cookie", "charge", "battery", "help"}
        if lowered == "interaction":
            return {"answer", "reply", "respond", "clarify", "question", "hello", "hey", "here", "thread", "open"}
        return set()

    def _query_text(self, payload: Dict[str, Any]) -> str:
        need = str(payload.get("need", "") or "")
        style = str(payload.get("style", "") or "")
        message = str(payload.get("message", "") or "")
        vector = payload.get("vector", {}) if isinstance(payload.get("vector", {}), dict) else {}
        options = [need, style, message, str(vector.get("message", "") or ""), str(payload.get("pending_text", "") or "")]
        if need == "power":
            options.extend(["power low", "battery low", "cookie", "charge", "need help", "power request"])
        elif need == "interaction":
            options.extend(["open thread", "reply needed", "answer", "clarify", "social response", "interaction pressure"])
        query = " ".join(part for part in options if part)
        return self._clean_text(query)

    def _style_score(self, text: str, style: str) -> float:
        style = str(style or "direct_simple")
        clean = self._clean_text(text)
        n = len(clean)
        lower = clean.lower()
        if style == "urgent_direct":
            score = 0.10 if n <= 72 else 0.02
            if any(tok in lower for tok in ("need", "soon", "critical", "low", "charge", "cookie", "answer", "reply", "now", "here")):
                score += 0.10
            return score
        if style == "gentle_notice":
            score = 0.08 if n <= 96 else 0.03
            if any(tok in lower for tok in ("later", "bit", "dipping", "help", "here", "open", "clarify", "thread")):
                score += 0.08
            return score
        score = 0.10 if n <= 88 else 0.03
        if any(tok in lower for tok in ("power", "low", "cookie", "help", "charge", "answer", "reply", "clarify", "hello", "hey", "here")):
            score += 0.08
        return score

    def _base_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = self._clean_text(str(payload.get("message", "") or ""))
        return {
            "text": text,
            "source": "fallback",
            "role": "assistant",
            "score": 0.16,
            "kind": "fallback",
        }

    def _render_template(self, template: str, payload: Dict[str, Any], slots: Dict[str, Any] | None = None) -> str:
        merged: Dict[str, Any] = {}
        if isinstance(slots, dict):
            merged.update(slots)
        pending_text = self._clean_text(str(payload.get("pending_text", "") or ""))
        merged.setdefault("pending_text", pending_text)
        merged.setdefault("topic", pending_text or str(merged.get("focus", "") or "thread"))
        merged.setdefault("focus", str(merged.get("focus", "") or pending_text or "that").strip())
        merged.setdefault("greeting", str(merged.get("greeting", "Hey") or "Hey").strip().title())
        try:
            return self._clean_text(template.format(**merged))
        except Exception:
            return self._clean_text(template)

    def _build_interaction_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pressure = payload.get("pressure", {}) if isinstance(payload.get("pressure", {}), dict) else {}
        pending_text = self._clean_text(str(payload.get("pending_text", pressure.get("pending_text", "")) or ""))
        style = str(payload.get("style", "direct_simple") or "direct_simple")

        if bool(pressure.get("greeting", False)):
            text = "Hey, I'm here."
        elif bool(pressure.get("question", False)) or bool(pressure.get("response_request", False)):
            text = "I want to answer that." if style != "urgent_direct" else "I should answer that now."
        elif bool(pressure.get("clarify_ready", False)):
            text = "What outcome should I optimize for?"
        elif pending_text:
            text = f"I still have an open interaction thread around: {pending_text}"
        else:
            text = "I hear the open thread."

        return {
            "text": self._clean_text(text),
            "source": "interaction_renderer",
            "role": "assistant",
            "score": 0.44,
            "kind": "interaction_fallback",
        }

    def _render_utterance_pattern(self, row: Dict[str, Any], payload: Dict[str, Any]) -> str:
        meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
        template = self._clean_text(str(meta.get("template", "") or ""))
        slots = dict(meta.get("slots", {}) or {})
        surface = self._clean_text(str(meta.get("surface", "") or ""))
        if template:
            rendered = self._render_template(template, payload, slots)
            if rendered:
                return rendered
        return surface

    async def _recent_speech_state(self, ctx) -> Dict[str, Any]:
        raw = await ctx.get_kv("speech_reason:last", {})
        return raw if isinstance(raw, dict) else {}

    def _repeat_penalty(self, payload: Dict[str, Any], candidate: Dict[str, Any], recent: Dict[str, Any]) -> float:
        if not recent:
            return 0.0
        recent_text = self._norm(str(recent.get("utterance", "") or ""))
        cand_text = self._norm(str(candidate.get("text", "") or ""))
        if not recent_text or not cand_text or recent_text != cand_text:
            return 0.0

        recent_need = str(recent.get("need", "") or "")
        payload_need = str(payload.get("need", "") or "")
        if recent_need and payload_need and recent_need != payload_need:
            return 0.0

        age_s = max(0.0, time.time() - self._safe_float(recent.get("ts", 0.0), 0.0))
        if age_s > 300.0:
            return 0.0

        source = str(candidate.get("source", "") or "")
        base = 0.20 if source == "fallback" else 0.10
        freshness = 1.0 - min(1.0, age_s / 300.0)
        return round(base * freshness, 4)

    async def _score_candidates(self, ctx, payload: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        recent = await self._recent_speech_state(ctx)
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            score = self._safe_float(item.get("score", 0.0), 0.0)
            penalty = self._repeat_penalty(payload, item, recent)
            if penalty > 0.0:
                score = max(0.0, score - penalty)
                item["repeat_penalty"] = penalty
            item["score"] = round(score, 4)
            scored.append(item)
        return scored

    def _semantic_candidates(self, mem_store: MemoryStore, query: str, need: str, style: str) -> List[Dict[str, Any]]:
        hits = mem_store.search_semantic(query, k=10)
        lexicon = self._need_lexicon(need)
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        now_ts = time.time()
        for rank, item in enumerate(hits):
            if not isinstance(item, dict):
                continue
            text = self._clean_text(str(item.get("text", "") or ""))
            if self._is_bad_candidate(text):
                continue
            norm = self._norm(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)

            meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
            role = str(meta.get("role", "assistant") or "assistant")
            kind = str(meta.get("kind", "") or "")
            sal = item.get("salience", {}) if isinstance(item.get("salience", {}), dict) else {}
            eff = sal
            if hasattr(mem_store, "_effective_salience"):
                try:
                    eff = mem_store._effective_salience(item, now_ts=now_ts)
                except Exception:
                    eff = sal
            reinforcement_pts = self._safe_float(eff.get("reinforcement_pts", sal.get("reinforcement_pts", 0.0)), 0.0)
            satisfaction = self._safe_float(eff.get("satisfaction", sal.get("satisfaction", 0.0)), 0.0)
            salience_score = self._safe_float(eff.get("score", sal.get("score", 0.0)), 0.0)

            token_bonus = 0.0
            if lexicon:
                overlap = len(lexicon & set(self._tokens(text)))
                token_bonus = min(0.18, overlap * 0.05)
            role_bonus = 0.08 if role == "assistant" else 0.05 if role == "user" else 0.0
            kind_bonus = 0.08 if kind == "reinforced" else 0.03 if kind else 0.0
            rank_penalty = rank * 0.015
            score = max(
                0.0,
                0.34
                + min(0.18, reinforcement_pts * 0.05)
                + min(0.12, satisfaction * 0.10)
                + min(0.10, salience_score * 0.10)
                + token_bonus
                + role_bonus
                + kind_bonus
                + self._style_score(text, style)
                - rank_penalty,
            )
            out.append({
                "text": text,
                "source": "semantic_memory",
                "role": role,
                "score": round(score, 4),
                "kind": kind or "semantic",
            })
        return out

    def _utterance_pattern_candidates(self, mem_cell_store: MemCellStore, query: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        hits = mem_cell_store.search_text_cells(query, limit=12)
        style = str(payload.get("style", "direct_simple") or "direct_simple")
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rank, row in enumerate(hits):
            if not isinstance(row, dict) or str(row.get("kind", "") or "") != "utterance_pattern":
                continue
            text = self._render_utterance_pattern(row, payload)
            if self._is_bad_candidate(text):
                continue
            norm = self._norm(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
            role = str(meta.get("role", "assistant") or "assistant")
            act_type = str(meta.get("act_type", "") or "")
            promotion = self._safe_float(row.get("promotion", 0.0), 0.0)
            salience_now = self._safe_float(row.get("current_salience", row.get("activation", 0.0)), 0.0)
            score = max(0.0, 0.42 + min(0.12, promotion * 0.10) + min(0.14, salience_now * 0.12) + (0.08 if role == "assistant" else 0.0) + (0.08 if act_type in ("answer_start", "clarify_target", "clarify_focus", "greet_present", "acknowledge") else 0.0) + self._style_score(text, style) - (rank * 0.015))
            out.append({"text": text, "source": "utterance_pattern", "role": role, "score": round(score, 4), "kind": act_type or "utterance_pattern"})
        return out

    def _memcell_candidates(self, mem_cell_store: MemCellStore, query: str, need: str, style: str) -> List[Dict[str, Any]]:
        hits = mem_cell_store.search_text_cells(query, limit=8)
        lexicon = self._need_lexicon(need)
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rank, row in enumerate(hits):
            if not isinstance(row, dict):
                continue
            anchor = row.get("anchor", {}) if isinstance(row.get("anchor", {}), dict) else {}
            refs = row.get("refs", []) if isinstance(row.get("refs", []), list) else []
            text = self._clean_text(str(anchor.get("ref", "") or ""))
            if not text and refs:
                for ref in refs:
                    if isinstance(ref, dict):
                        text = self._clean_text(str(ref.get("value", "") or ""))
                        if text:
                            break
            if self._is_bad_candidate(text):
                continue
            norm = self._norm(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)

            meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
            role = str(meta.get("role", "assistant") or "assistant")
            reinforcement_pts = self._safe_float(row.get("reinforcement_pts", 0.0), 0.0)
            promotion = self._safe_float(row.get("promotion", 0.0), 0.0)
            salience_now = self._safe_float(row.get("current_salience", row.get("activation", 0.0)), 0.0)
            token_bonus = 0.0
            if lexicon:
                overlap = len(lexicon & set(self._tokens(text)))
                token_bonus = min(0.14, overlap * 0.04)
            role_bonus = 0.06 if role == "assistant" else 0.04 if role == "user" else 0.0
            score = max(
                0.0,
                0.28
                + min(0.16, reinforcement_pts * 0.05)
                + min(0.10, promotion * 0.10)
                + min(0.10, salience_now * 0.10)
                + token_bonus
                + role_bonus
                + self._style_score(text, style)
                - (rank * 0.012),
            )
            out.append({
                "text": text,
                "source": "mem_cell",
                "role": role,
                "score": round(score, 4),
                "kind": str(row.get("kind", "mem_cell") or "mem_cell"),
            })
        return out

    async def _gather_candidates(self, ctx, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        need = str(payload.get("need", "") or "")
        style = str(payload.get("style", "direct_simple") or "direct_simple")
        query = self._query_text(payload)

        candidates: List[Dict[str, Any]] = []
        fallback = self._base_candidate(payload)
        if not self._is_bad_candidate(fallback["text"]):
            candidates.append(fallback)

        mem_store = await ctx.get_kv("memory:store", None)
        if isinstance(mem_store, MemoryStore):
            try:
                candidates.extend(self._semantic_candidates(mem_store, query=query, need=need, style=style))
            except Exception as exc:
                await ctx.log_debug(f"[{self.name}] semantic candidate search failed", error=repr(exc))

        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(mem_cell_store, MemCellStore):
            try:
                candidates.extend(self._memcell_candidates(mem_cell_store, query=query, need=need, style=style))
            except Exception as exc:
                await ctx.log_debug(f"[{self.name}] mem-cell candidate search failed", error=repr(exc))

        dedup: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            text = self._clean_text(candidate.get("text", ""))
            norm = self._norm(text)
            if not norm:
                continue
            prior = dedup.get(norm)
            if prior is None or self._safe_float(candidate.get("score", 0.0), 0.0) > self._safe_float(prior.get("score", 0.0), 0.0):
                dedup[norm] = dict(candidate, text=text)

        ordered = sorted(
            dedup.values(),
            key=lambda item: (
                self._safe_float(item.get("score", 0.0), 0.0),
                item.get("source") != "fallback",
            ),
            reverse=True,
        )
        return ordered[:8]

    def _select_candidate(self, payload: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return self._base_candidate(payload)

        best = dict(candidates[0])
        if str(best.get("source", "") or "") != "fallback":
            return best

        best_non_fallback = None
        for candidate in candidates:
            if str(candidate.get("source", "") or "") == "fallback":
                continue
            if best_non_fallback is None or self._safe_float(candidate.get("score", 0.0), 0.0) > self._safe_float(best_non_fallback.get("score", 0.0), 0.0):
                best_non_fallback = dict(candidate)

        if best_non_fallback is not None and self._safe_float(best_non_fallback.get("score", 0.0), 0.0) >= 0.14:
            return best_non_fallback
        return best

    async def _update_pending_request(self, ctx, event: Event, chosen: Dict[str, Any]) -> None:
        need = str((event.payload or {}).get("need", "") or "") if isinstance(event.payload, dict) else ""
        if need == "power":
            pending_key = "drive:power_pending_request"
        elif need == "interaction":
            pending_key = "drive:interaction_pending_request"
        else:
            return
        pending = await ctx.get_kv(pending_key, None)
        if not isinstance(pending, dict):
            return
        pending["message"] = chosen.get("text", pending.get("message"))
        pending["utterance_source"] = chosen.get("source", "fallback")
        pending["utterance_score"] = self._safe_float(chosen.get("score", 0.0), 0.0)
        await ctx.set_kv(pending_key, pending)

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "speech/reason":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {"message": event.payload}
        outlet = str(payload.get("outlet", "textual") or "textual")
        if outlet == "motion":
            return []

        candidates = await self._gather_candidates(ctx, payload)
        candidates = await self._score_candidates(ctx, payload, candidates)
        candidates.sort(
            key=lambda item: (
                self._safe_float(item.get("score", 0.0), 0.0),
                item.get("source") != "fallback",
            ),
            reverse=True,
        )
        chosen = self._select_candidate(payload, candidates)
        utterance = self._clean_text(chosen.get("text", "") or payload.get("message", ""))
        if not utterance:
            return []

        await self._update_pending_request(ctx, event, chosen)
        await ctx.set_kv(
            "speech_reason:last",
            {
                "ts": time.time(),
                "need": str(payload.get("need", "") or ""),
                "style": str(payload.get("style", "") or ""),
                "utterance": utterance,
                "source": chosen.get("source", "fallback"),
                "score": self._safe_float(chosen.get("score", 0.0), 0.0),
                "candidates": candidates[:5],
            },
        )

        meta = dict(event.meta or {})
        need = str(payload.get("need", "") or "")
        meta.update({
            "kind": f"speech_reason_{need}" if need else "speech_reason_emit",
            "need": need,
            "utterance_style": str(payload.get("style", "direct_simple") or "direct_simple"),
            "utterance_source": chosen.get("source", "fallback"),
            "utterance_score": round(self._safe_float(chosen.get("score", 0.0), 0.0), 4),
        })

        return [
            Event(
                topic="act/speech",
                payload={
                    "text": utterance,
                    "style": "assistant",
                    "channel": str(payload.get("channel", "default") or "default"),
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta=meta,
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["speech/reason"],
        output_topics=["act/speech"],
        priority=10,
        cooldown_sec=0.0,
    )
    yield SpeechReasonNeuron(cfg)
