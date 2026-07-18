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
            return {"power", "low", "charge", "battery", "recharge", "top", "help"}
        if lowered == "interaction":
            return {"answer", "reply", "respond", "clarify", "question", "hello", "hey", "here", "want", "feel"}
        return set()

    def _query_text(self, payload: Dict[str, Any]) -> str:
        need = str(payload.get("need", "") or "")
        style = str(payload.get("style", "") or "")
        message = str(payload.get("message", "") or "")
        vector = payload.get("vector", {}) if isinstance(payload.get("vector", {}), dict) else {}
        pending_text = str(payload.get("pending_text", "") or "")
        options = [need, style, message, pending_text, str(vector.get("message", "") or "")]
        if need == "power":
            options.extend(["power low", "battery low", "charge", "recharge", "need help", "power request"])
        elif need == "interaction":
            options.extend(["interaction pressure", "reply needed", "open thread", "clarify response", "social response"])
        query = " ".join(part for part in options if part)
        return self._clean_text(query)

    def _syntax_lookup_text(self, payload: Dict[str, Any]) -> str:
        pending = self._clean_text(str(payload.get("pending_text", "") or ""))
        if pending:
            return pending
        message = self._clean_text(str(payload.get("message", "") or ""))
        return message

    def _meta_ddna_targets(self, meta: Dict[str, Any]) -> Dict[str, float]:
        raw = meta.get("ddna_targets", {}) if isinstance(meta, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                name = self._norm(str(key or "")).replace(" ", "_")
                if not name:
                    continue
                out[name] = max(out.get(name, 0.0), abs(self._safe_float(value, 1.0)))
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                name = self._norm(str(item or "")).replace(" ", "_")
                if name:
                    out[name] = max(out.get(name, 0.0), 1.0)
        return out

    def _ddna_bonus(self, meta: Dict[str, Any], *, need: str, style: str) -> float:
        targets = self._meta_ddna_targets(meta)
        if not targets:
            return 0.0
        bonus = 0.0
        need = str(need or "")
        style = str(style or "")
        if need == "interaction":
            if "warmth" in targets:
                bonus += min(0.12, targets["warmth"] * 0.035)
            if "friendly" in targets:
                bonus += min(0.12, targets["friendly"] * 0.035)
            if "supportive" in targets:
                bonus += min(0.08, targets["supportive"] * 0.025)
        if style in ("direct_simple", "gentle_notice"):
            if "direct" in targets:
                bonus += min(0.06, targets["direct"] * 0.02)
            if style == "gentle_notice" and "gentle" in targets:
                bonus += min(0.08, targets["gentle"] * 0.025)
        return round(min(0.28, bonus), 4)

    def _syntax_guidance(self, mem_cell_store: MemCellStore | None, lookup_text: str) -> Dict[str, Any]:
        guidance: Dict[str, Any] = {"preferred_replies": [], "avoid_replies": [], "ddna_targets": {}, "classifiers": []}
        if not isinstance(mem_cell_store, MemCellStore) or not lookup_text:
            return guidance
        try:
            hits = mem_cell_store.search_text_cells(lookup_text, limit=16, tiers=("learned", "long", "now", "short"))
        except Exception:
            return guidance
        seen_reply: set[str] = set()
        seen_avoid: set[str] = set()
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta", {}) if isinstance(hit.get("meta", {}), dict) else {}
            kind = str(hit.get("kind", "") or meta.get("kind", "") or "")
            if kind not in {"syntax_rule", "trainer_alignment"}:
                continue
            score = self._safe_float(hit.get("score", 0.0), 0.0)
            ddna = self._meta_ddna_targets(meta)
            for key, value in ddna.items():
                guidance["ddna_targets"][key] = max(float(guidance["ddna_targets"].get(key, 0.0)), value)
            for classifier in list(meta.get("syntax_classifiers", []) or []):
                name = self._norm(str(classifier or "")).replace(" ", "_")
                if name and name not in guidance["classifiers"]:
                    guidance["classifiers"].append(name)
            reply = self._clean_text(str(meta.get("reply_text", "") or meta.get("desired_utterance", "") or ""))
            if reply and not self._is_bad_candidate(reply):
                norm = self._norm(reply)
                if norm and norm not in seen_reply:
                    seen_reply.add(norm)
                    guidance["preferred_replies"].append({"text": reply, "score": score, "meta": meta})
            for avoid in list(meta.get("avoid_replies", []) or []):
                avoid_text = self._clean_text(str(avoid or ""))
                norm = self._norm(avoid_text)
                if avoid_text and norm and norm not in seen_avoid:
                    seen_avoid.add(norm)
                    guidance["avoid_replies"].append(avoid_text)
            bad = self._clean_text(str(meta.get("bad_utterance", "") or meta.get("trainer_bad_utterance", "") or ""))
            norm_bad = self._norm(bad)
            if bad and norm_bad and norm_bad not in seen_avoid:
                seen_avoid.add(norm_bad)
                guidance["avoid_replies"].append(bad)
        return guidance

    def _style_score(self, text: str, style: str) -> float:
        style = str(style or "direct_simple")
        clean = self._clean_text(text)
        n = len(clean)
        if style == "urgent_direct":
            score = 0.10 if n <= 72 else 0.02
            if any(tok in clean.lower() for tok in ("need", "soon", "critical", "low", "charge", "recharge")):
                score += 0.10
            return score
        if style == "gentle_notice":
            score = 0.08 if n <= 96 else 0.03
            if any(tok in clean.lower() for tok in ("soon", "bit", "dipping", "top", "charge")):
                score += 0.08
            return score
        score = 0.10 if n <= 88 else 0.03
        if any(tok in clean.lower() for tok in ("power", "low", "help", "charge", "recharge")):
            score += 0.08
        return score

    def _fallback_variants(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Canned fallback utterances removed after line-live testing."""
        return []

    def _base_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": "",
            "source": "none",
            "role": "assistant",
            "score": 0.0,
            "kind": "no_candidate",
        }

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
        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        lookup_text = self._syntax_lookup_text(payload)
        guidance = self._syntax_guidance(mem_cell_store if isinstance(mem_cell_store, MemCellStore) else None, lookup_text)
        avoid_norms = {self._norm(text) for text in list(guidance.get("avoid_replies", []) or []) if self._norm(str(text or ""))}
        need = str(payload.get("need", "") or "")
        style = str(payload.get("style", "direct_simple") or "direct_simple")
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            score = self._safe_float(item.get("score", 0.0), 0.0)
            cand_norm = self._norm(str(item.get("text", "") or ""))
            if cand_norm and cand_norm in avoid_norms:
                item["syntax_avoid"] = True
                item["score"] = 0.0
                scored.append(item)
                continue
            penalty = self._repeat_penalty(payload, item, recent)
            if penalty > 0.0:
                score = max(0.0, score - penalty)
                item["repeat_penalty"] = penalty
                if str(payload.get("need", "") or "") and self._norm(str(item.get("text", "") or "")) == self._norm(str(recent.get("utterance", "") or "")):
                    # Need-state speech must not repeat the same attempt without a new result.
                    score = 0.0
                    item["repeat_blocked"] = True
            meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
            ddna_bonus = self._ddna_bonus(meta, need=need, style=style)
            if ddna_bonus:
                score += ddna_bonus
                item["ddna_bonus"] = ddna_bonus
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
            if kind == "reinforced":
                kind_bonus = 0.08
            elif kind == "trainer_correction":
                kind_bonus = 0.16
            else:
                kind_bonus = 0.03 if kind else 0.0
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
                + self._ddna_bonus(meta, need=need, style=style)
                - rank_penalty,
            )
            out.append({
                "text": text,
                "source": "semantic_memory",
                "role": role,
                "score": round(score, 4),
                "kind": kind or "semantic",
                "meta": meta,
            })
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
            kind = str(row.get("kind", "") or "")
            meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
            text = self._clean_text(str(anchor.get("ref", "") or ""))
            if kind == "trainer_alignment":
                text = self._clean_text(str(meta.get("desired_utterance", "") or ""))
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

            role = str(meta.get("role", "assistant") or "assistant")
            reinforcement_pts = self._safe_float(row.get("reinforcement_pts", 0.0), 0.0)
            promotion = self._safe_float(row.get("promotion", 0.0), 0.0)
            salience_now = self._safe_float(row.get("current_salience", row.get("activation", 0.0)), 0.0)
            token_bonus = 0.0
            if lexicon:
                overlap = len(lexicon & set(self._tokens(text)))
                token_bonus = min(0.14, overlap * 0.04)
            role_bonus = 0.06 if role == "assistant" else 0.04 if role == "user" else 0.0
            trainer_need = str(meta.get("trainer_need", "") or "")
            trainer_style = str(meta.get("trainer_style", "") or "")
            trainer_bonus = 0.0
            if kind == "trainer_alignment":
                trainer_bonus += 0.22
                if trainer_need and trainer_need == str(need or ""):
                    trainer_bonus += 0.10
                if trainer_style and trainer_style == str(style or ""):
                    trainer_bonus += 0.06
            score = max(
                0.0,
                0.28
                + min(0.16, reinforcement_pts * 0.05)
                + min(0.10, promotion * 0.10)
                + min(0.10, salience_now * 0.10)
                + token_bonus
                + role_bonus
                + trainer_bonus
                + self._style_score(text, style)
                + self._ddna_bonus(meta, need=need, style=style)
                - (rank * 0.012),
            )
            row_tier = str(row.get("tier", "now") or "now")
            row_source = "trainer_alignment" if kind == "trainer_alignment" else ("mem_cell_derived" if row_tier == "derived" else "mem_cell")
            out.append({
                "text": text,
                "source": row_source,
                "role": role,
                "score": round(score, 4),
                "kind": kind or "mem_cell",
                "cell_id": str(row.get("cell_id", "") or ""),
                "tier": row_tier,
                "meta": meta,
            })
        return out

    async def _power_speech_blocked(self, ctx, payload: Dict[str, Any]) -> bool:
        if str(payload.get("need", "") or "") != "power":
            return False
        gate_enabled = bool(await ctx.get_kv("drive:power:speech_gate_enabled", True))
        if not gate_enabled:
            return False
        if bool(payload.get("user_requested", False)):
            return False
        if bool(await ctx.get_kv("drive:power:allow_unsolicited_speech", False)):
            return False
        pressure = payload.get("pressure", {}) if isinstance(payload.get("pressure", {}), dict) else {}
        urgency = self._safe_float(pressure.get("urgency", 0.0), 0.0)
        critical_threshold = self._safe_float(await ctx.get_kv("drive:power:critical_speech_threshold", 0.90), 0.90)
        return urgency < critical_threshold

    async def _emit_power_status_if_due(self, ctx, event: Event, payload: Dict[str, Any]) -> None:
        now = time.time()
        last_status_ts = self._safe_float(await ctx.get_kv("drive:power:last_status_ts", 0.0), 0.0)
        status_cooldown_s = self._safe_float(await ctx.get_kv("drive:power:status_cooldown_s", 120.0), 120.0)
        if last_status_ts > 0.0 and (now - last_status_ts) < status_cooldown_s:
            return
        pressure = payload.get("pressure", {}) if isinstance(payload.get("pressure", {}), dict) else {}
        state = payload.get("state", {}) if isinstance(payload.get("state", {}), dict) else {}
        pct = self._safe_float(pressure.get("pct", state.get("pct", 100.0)), 100.0)
        urgency = self._safe_float(pressure.get("urgency", 0.0), 0.0)
        if urgency >= 0.85:
            text = f"power: critical at {pct:.0f}% | charge soon"
            band = "critical"
        elif urgency >= 0.55:
            text = f"power: low at {pct:.0f}% | charge soon"
            band = "active"
        else:
            text = f"power: dipping at {pct:.0f}% | watch charge"
            band = "rising"
        status_payload = {
            "text": text,
            "kind": "power_need_status",
            "need": "power",
            "band": band,
            "urgency": round(urgency, 4),
            "pct": round(pct, 2),
            "pressure": pressure,
            "speech_allowed": False,
            "ts": now,
        }
        await ctx.set_kv("drive:power:last_status_ts", now)
        await ctx.set_kv("drive:power:last_status", status_payload)
        await ctx.emit(Event(
            topic="ui/status",
            payload=status_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "power_need_status",
                "need": "power",
                "store_in_memory": False,
                "reinforcement_eligible": False,
                "self_output_track": False,
            },
        ))

    async def _gather_candidates(self, ctx, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        need = str(payload.get("need", "") or "")
        style = str(payload.get("style", "direct_simple") or "direct_simple")
        query = self._query_text(payload)

        candidates: List[Dict[str, Any]] = []

        mem_store = await ctx.get_kv("memory:store", None)
        if isinstance(mem_store, MemoryStore):
            try:
                candidates.extend(self._semantic_candidates(mem_store, query=query, need=need, style=style))
            except Exception as exc:
                await ctx.log_debug(f"[{self.name}] semantic candidate search failed", error=repr(exc))

        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(mem_cell_store, MemCellStore):
            try:
                guidance = self._syntax_guidance(mem_cell_store, self._syntax_lookup_text(payload))
                for reply in list(guidance.get("preferred_replies", []) or [])[:4]:
                    if not isinstance(reply, dict):
                        continue
                    reply_text = self._clean_text(str(reply.get("text", "") or ""))
                    if not reply_text or self._is_bad_candidate(reply_text):
                        continue
                    meta = reply.get("meta", {}) if isinstance(reply.get("meta", {}), dict) else {}
                    candidates.append({
                        "text": reply_text,
                        "source": "syntax_rule",
                        "role": "assistant",
                        "score": round(0.72 + min(0.18, self._safe_float(reply.get("score", 0.0), 0.0)), 4),
                        "kind": "syntax_rule_reply",
                        "meta": meta,
                    })
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
        if self._safe_float(best.get("score", 0.0), 0.0) <= 0.0:
            return self._base_candidate(payload)
        return best

    async def _update_pending_request(self, ctx, event: Event, chosen: Dict[str, Any]) -> None:
        need = str((event.payload or {}).get("need", "") or "") if isinstance(event.payload, dict) else ""
        if need != "power":
            return
        pending = await ctx.get_kv("drive:power_pending_request", None)
        if not isinstance(pending, dict):
            return
        pending["message"] = chosen.get("text", pending.get("message"))
        pending["utterance_source"] = chosen.get("source", "fallback")
        pending["utterance_score"] = self._safe_float(chosen.get("score", 0.0), 0.0)
        await ctx.set_kv("drive:power_pending_request", pending)

    async def _note_usage(self, ctx, chosen: Dict[str, Any]) -> None:
        source = str(chosen.get("source", "") or "")
        if source not in {"mem_cell", "mem_cell_derived", "trainer_alignment"}:
            return
        cell_id = str(chosen.get("cell_id", "") or "").strip()
        if not cell_id:
            return
        mem_cell_store = await ctx.get_kv("memory:mem_cell_store", None)
        if not isinstance(mem_cell_store, MemCellStore):
            return
        try:
            mem_cell_store.note_cell_usage(cell_id, success=True)
        except Exception as exc:
            await ctx.log_debug(f"[{self.name}] mem-cell usage note failed", error=repr(exc), cell_id=cell_id)

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

        if await self._power_speech_blocked(ctx, payload):
            await self._emit_power_status_if_due(ctx, event, payload)
            await ctx.set_kv(
                "speech_reason:last_blocked",
                {
                    "ts": time.time(),
                    "need": "power",
                    "reason": "power_status_speech_gate",
                    "payload": payload,
                },
            )
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
        chosen_score = self._safe_float(chosen.get("score", 0.0), 0.0)
        utterance = self._clean_text(chosen.get("text", ""))
        if (not utterance) or (str(payload.get("need", "") or "") and chosen_score <= 0.0):
            need = str(payload.get("need", "") or "")
            if need:
                await ctx.emit(Event(
                    topic="thought/internal",
                    payload={
                        "text": "",
                        "need": need,
                        "state": "unresolved_expression_no_candidate",
                        "kind": "need_expression_blocked",
                        "source_need": need,
                    },
                    source=self.name,
                    correlation_id=event.correlation_id,
                    meta={
                        "channel": "thought",
                        "kind": "need_expression_blocked",
                        "need": need,
                        "store_in_memory": False,
                        "reinforcement_eligible": False,
                        "self_output_track": False,
                        "cognitive_visible": False,
                    },
                ))
            return []

        await self._update_pending_request(ctx, event, chosen)
        await self._note_usage(ctx, chosen)
        await ctx.set_kv(
            "speech_reason:last",
            {
                "ts": time.time(),
                "need": str(payload.get("need", "") or ""),
                "style": str(payload.get("style", "") or ""),
                "utterance": utterance,
                "message": str(payload.get("message", "") or ""),
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
        output_topics=["act/speech", "ui/status"],
        priority=10,
        cooldown_sec=0.0,
    )
    yield SpeechReasonNeuron(cfg)
