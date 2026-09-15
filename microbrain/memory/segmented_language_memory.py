from __future__ import annotations

"""Low-cost language memory for token chains and phrase routes.

This store is deliberately separate from MemCellStore.  It builds the cheap
front-door language layer: tokens, adjacency edges, phrase patterns, and trainer
corrections.  Parser guesses stay weak; trainer corrections get stronger.
"""

import json
import os
import re
import time
from hashlib import blake2b
from pathlib import Path
from typing import Any, Mapping, Sequence

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
GREETING_PHRASES = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
GRATITUDE_PHRASES = {"thank you", "thanks", "thank"}


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").lower().split()).strip()


def clean_text(value: Any, limit: int = 320) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()[:limit]


def tokens_for(value: Any) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(value or ""))]


def stable_id(prefix: str, *parts: Any, size: int = 10) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = blake2b(raw.encode("utf-8", errors="replace"), digest_size=size).hexdigest()
    return f"{prefix}:{digest}"


class SegmentedLanguageMemory:
    """Small segmented language store with streaming upserts."""

    def __init__(self, base_dir: str | Path):
        root = Path(base_dir)
        self.base_dir = root / "memory_v2"
        self.tok_dir = self.base_dir / "tok_graph"
        self.pattern_dir = self.base_dir / "pattern_memory"
        self.evidence_dir = self.base_dir / "language_evidence"
        self.teaching_dir = self.base_dir / "teaching_memory"
        self.tok_dir.mkdir(parents=True, exist_ok=True)
        self.pattern_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.teaching_dir.mkdir(parents=True, exist_ok=True)

    @property
    def tokens_path(self) -> Path:
        return self.tok_dir / "tokens.jsonl"

    @property
    def adjacency_path(self) -> Path:
        return self.tok_dir / "adjacency.jsonl"

    @property
    def phrases_path(self) -> Path:
        return self.tok_dir / "phrase_patterns.jsonl"

    @property
    def corrections_path(self) -> Path:
        return self.teaching_dir / "corrections.jsonl"

    @property
    def slearn_rules_path(self) -> Path:
        return self.pattern_dir / "slearn_rules.jsonl"

    def observe_text(
        self,
        text: str,
        *,
        role: str = "user",
        source: str = "unknown",
        strength: str = "observed",
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        surface = norm_text(text)
        toks = tokens_for(surface)
        ts = time.time()
        if not toks:
            return {
                "schema": "language.parse_pass2.v1",
                "ok": False,
                "reason": "no_tokens",
                "input": clean_text(text),
                "memory_updates": [],
            }

        updates: list[str] = []
        tokids: list[str] = []
        for token in toks:
            tokid = stable_id("tok", token, size=8)
            tokids.append(tokid)
            before = self._find_by_key(self.tokens_path, "tokid", tokid)
            row = self._merge_token(before, token=token, tokid=tokid, role=role, source=source, ts=ts)
            self._upsert_by_key(self.tokens_path, "tokid", tokid, row)
            updates.append(f"token:{token}")

        edge_scores: list[float] = []
        for left, right in zip(toks, toks[1:]):
            edge_id = stable_id("edge", left, right, "next", size=8)
            before = self._find_by_key(self.adjacency_path, "edge_id", edge_id)
            row = self._merge_edge(before, left=left, right=right, edge_id=edge_id, role=role, source=source, ts=ts)
            self._upsert_by_key(self.adjacency_path, "edge_id", edge_id, row)
            edge_scores.append(float(row.get("confidence", 0.0) or 0.0))
            updates.append(f"edge:{left}>{right}")

        intent, route, response_type, note, base_conf = self._infer_route(surface, toks)
        slots = self._slots_for_phrase(surface, toks, intent=intent, route=route)
        pattern_id = stable_id("phrase", surface, intent, route, size=8)
        before = self._find_by_key(self.phrases_path, "pattern_id", pattern_id)
        phrase = self._merge_phrase(
            before,
            pattern_id=pattern_id,
            surface=surface,
            toks=toks,
            tokids=tokids,
            intent=intent,
            route=route,
            response_type=response_type,
            base_conf=base_conf,
            strength=strength,
            role=role,
            source=source,
            note=note,
            ts=ts,
            meta=meta,
            slots=slots,
        )
        self._upsert_by_key(self.phrases_path, "pattern_id", pattern_id, phrase)
        updates.append(f"phrase:{intent}:{route}")

        adjacency_score = sum(edge_scores) / max(1, len(edge_scores)) if len(toks) > 1 else 0.4
        confidence = min(1.0, (adjacency_score * 0.42) + (float(phrase.get("confidence", 0.0) or 0.0) * 0.58))
        return {
            "schema": "language.parse_pass2.v1",
            "ok": True,
            "input": clean_text(text),
            "normalized": surface,
            "tok_chain": tokids,
            "token_text": toks,
            "adjacency_score": round(adjacency_score, 6),
            "phrase_match": pattern_id,
            "intent": intent,
            "route": route,
            "expected_response_type": response_type,
            "confidence": round(confidence, 6),
            "needs_deeper_lookup": route in {"bounded_lookup", "pattern/unknown"},
            "slots": slots,
            "memory_updates": updates,
        }

    def apply_trainer_correction(
        self,
        *,
        correction_text: str,
        context_text: str = "",
        bad_utterance: str = "",
        need: str = "",
        style: str = "",
        source: str = "trainer",
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        correction = clean_text(correction_text, 260)
        context = clean_text(context_text, 360)
        bad = clean_text(bad_utterance, 260)
        ts = time.time()
        observed = self.observe_text(correction, role="assistant", source=source, strength="trainer", meta=meta)
        correction_id = stable_id("corr", context, bad, correction, need, style, size=8)
        row = {
            "schema": "teaching_memory.correction.v1",
            "correction_id": correction_id,
            "context": context,
            "bad_utterance": bad,
            "desired": correction,
            "desired_pattern_id": observed.get("phrase_match", ""),
            "need": str(need or ""),
            "style": str(style or ""),
            "strength": "trainer",
            "confidence": 0.95,
            "avoid_surface": norm_text(bad),
            "expected_response_type": observed.get("expected_response_type", "trainer_corrected_surface"),
            "meta": dict(meta or {}),
            "ts": ts,
            "last_seen": ts,
        }
        self._upsert_by_key(self.corrections_path, "correction_id", correction_id, row)
        return {
            "schema": "language.pass2_trainer_update.v1",
            "correction_id": correction_id,
            "desired_pattern_id": observed.get("phrase_match", ""),
            "context": context,
            "bad_utterance": bad,
            "desired": correction,
            "memory_updates": list(observed.get("memory_updates", [])) + [f"correction:{correction_id}"],
        }

    def observe_text_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        default_role: str = "assistant",
        default_source: str = "batch",
        default_strength: str = "observed",
    ) -> dict[str, Any]:
        """Observe many text fragments while opening each shelf once."""
        if not items:
            return {"schema": "language.pass2_batch.v1", "accepted": 0, "rejected": 0, "memory_updates": []}

        token_rows = self._read_index_by_key(self.tokens_path, "tokid")
        edge_rows = self._read_index_by_key(self.adjacency_path, "edge_id")
        phrase_rows = self._read_index_by_key(self.phrases_path, "pattern_id")
        accepted = 0
        rejected = 0
        updates: list[str] = []
        for item in items:
            if not isinstance(item, Mapping):
                rejected += 1
                continue
            result = self._observe_text_into_indexes(
                token_rows,
                edge_rows,
                phrase_rows,
                str(item.get("text", "") or ""),
                role=str(item.get("role", default_role) or default_role),
                source=str(item.get("source", default_source) or default_source),
                strength=str(item.get("strength", default_strength) or default_strength),
                meta=item.get("meta", {}) if isinstance(item.get("meta", {}), Mapping) else {},
            )
            if result.get("ok"):
                accepted += 1
                updates.extend(list(result.get("memory_updates", []) or []))
            else:
                rejected += 1

        self._write_index_values(self.tokens_path, token_rows)
        self._write_index_values(self.adjacency_path, edge_rows)
        self._write_index_values(self.phrases_path, phrase_rows)
        return {
            "schema": "language.pass2_batch.v1",
            "accepted": accepted,
            "rejected": rejected,
            "memory_updates": updates[:64],
        }

    def apply_slearn_rules(self, rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Store parsed SLEARN rules in v2 without routing through mem-cell v1."""
        if not rules:
            return {"schema": "teaching_memory.slearn_batch.v1", "accepted": 0, "rejected": 0, "saved_rows": 0}

        token_rows = self._read_index_by_key(self.tokens_path, "tokid")
        edge_rows = self._read_index_by_key(self.adjacency_path, "edge_id")
        phrase_rows = self._read_index_by_key(self.phrases_path, "pattern_id")
        correction_rows = self._read_index_by_key(self.corrections_path, "correction_id")
        rule_rows = self._read_index_by_key(self.slearn_rules_path, "rule_id")

        accepted = 0
        rejected = 0
        saved_rows = 0
        ts = time.time()
        for rule in rules:
            if not isinstance(rule, Mapping):
                rejected += 1
                continue
            condition = clean_text(rule.get("condition_text", ""), 420)
            if not condition:
                rejected += 1
                continue

            meta = {
                "slearn_rule": True,
                "rule_kind": str(rule.get("rule_kind", "") or ""),
                "condition_raw": str(rule.get("condition_raw", "") or ""),
                "classifiers": list(rule.get("classifiers", []) or []),
                "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
                "ddna_targets": list(rule.get("ddna_targets", []) or []),
                "source_name": str(rule.get("source_name", "") or ""),
                "source_path": str(rule.get("source_path", "") or ""),
                "source_line": rule.get("source_line"),
            }
            observed = self._observe_text_into_indexes(
                token_rows,
                edge_rows,
                phrase_rows,
                condition,
                role="system",
                source=str(rule.get("source_name", "") or "slearn"),
                strength="trainer",
                meta=meta,
            )
            if not observed.get("ok"):
                rejected += 1
                continue

            reply_text = clean_text(rule.get("reply_text", ""), 260)
            bad = clean_text(rule.get("target_text", ""), 260) if int(rule.get("reinforce_weight", 0) or 0) < 0 else ""
            correction_id = ""
            if reply_text:
                reply_observed = self._observe_text_into_indexes(
                    token_rows,
                    edge_rows,
                    phrase_rows,
                    reply_text,
                    role="assistant",
                    source=str(rule.get("source_name", "") or "slearn"),
                    strength="trainer",
                    meta={**meta, "slearn_reply": True},
                )
                correction_id = stable_id("corr", condition, bad, reply_text, "interaction", "slearn", size=8)
                correction_rows[correction_id] = {
                    "schema": "teaching_memory.correction.v1",
                    "correction_id": correction_id,
                    "context": condition,
                    "bad_utterance": bad,
                    "desired": reply_text,
                    "desired_pattern_id": reply_observed.get("phrase_match", ""),
                    "need": "interaction",
                    "style": "slearn",
                    "strength": "trainer",
                    "confidence": 0.95,
                    "avoid_surface": norm_text(bad),
                    "expected_response_type": reply_observed.get("expected_response_type", "trainer_corrected_surface"),
                    "meta": meta,
                    "ts": ts,
                    "last_seen": ts,
                }
                saved_rows += 1

            identity = {
                "condition": condition,
                "rule_kind": str(rule.get("rule_kind", "") or ""),
                "classifiers": list(rule.get("classifiers", []) or []),
                "reply_text": reply_text,
                "avoid_replies": list(rule.get("avoid_replies", []) or []),
                "suppress_targets": list(rule.get("suppress_targets", []) or []),
            }
            rule_id = stable_id("slearn_rule", json.dumps(identity, ensure_ascii=False, sort_keys=True), size=8)
            before = rule_rows.get(rule_id, {})
            seen_count = int(before.get("seen_count", 0) or 0) + 1 if isinstance(before, Mapping) else 1
            rule_rows[rule_id] = {
                "schema": "pattern_memory.slearn_rule.v1",
                "rule_id": rule_id,
                "condition": condition,
                "condition_norm": norm_text(condition),
                "condition_pattern_id": observed.get("phrase_match", ""),
                "rule_kind": str(rule.get("rule_kind", "") or ""),
                "classifiers": list(rule.get("classifiers", []) or []),
                "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
                "ddna_targets": list(rule.get("ddna_targets", []) or []),
                "reply_text": reply_text,
                "correction_id": correction_id,
                "avoid_replies": list(rule.get("avoid_replies", []) or []),
                "suppress_targets": list(rule.get("suppress_targets", []) or []),
                "confidence": 0.90 if reply_text else 0.82,
                "seen_count": seen_count,
                "source_name": str(rule.get("source_name", "") or ""),
                "source_path": str(rule.get("source_path", "") or ""),
                "source_line": rule.get("source_line"),
                "ts": float(before.get("ts", ts) or ts) if isinstance(before, Mapping) else ts,
                "last_seen": ts,
            }
            accepted += 1
            saved_rows += 1

        self._write_index_values(self.tokens_path, token_rows)
        self._write_index_values(self.adjacency_path, edge_rows)
        self._write_index_values(self.phrases_path, phrase_rows)
        self._write_index_values(self.corrections_path, correction_rows)
        self._write_index_values(self.slearn_rules_path, rule_rows)
        return {
            "schema": "teaching_memory.slearn_batch.v1",
            "accepted": accepted,
            "rejected": rejected,
            "saved_rows": saved_rows,
        }

    def _infer_route(self, surface: str, toks: Sequence[str]) -> tuple[str, str, str, str, float]:
        text = " ".join(toks)
        if text in GREETING_PHRASES:
            return "greeting", "surface/social", "greeting_return", "known greeting phrase", 0.78
        if text in GRATITUDE_PHRASES:
            return "gratitude", "surface/social", "acknowledge_gratitude", "known gratitude phrase", 0.78
        if surface.endswith("?"):
            return "question", "bounded_lookup", "answer_or_clarify", "question marker", 0.42
        match = re.match(r"(.+?)\s+is\s+([a-z_ ]{2,80})$", text)
        if match:
            label = match.group(2).strip().replace(" ", "_")
            return f"classify:{label}", "pattern/teaching", "confirm_or_store", "x is y teaching/classification shape", 0.62
        if len(toks) <= 2:
            return "fragment_or_phrase", "pattern/unknown", "none", "short phrase without known route", 0.28
        return "unknown", "pattern/unknown", "none", "no strong route inferred", 0.24

    def _slots_for_phrase(self, surface: str, toks: Sequence[str], *, intent: str, route: str) -> dict[str, Any]:
        slots: dict[str, Any] = {"pattern_type": intent, "route": route}
        if "is" in toks:
            idx = list(toks).index("is")
            subject = " ".join(toks[:idx]).strip()
            predicate_tokens = list(toks[idx + 1 :])
            if predicate_tokens and predicate_tokens[0] in {"a", "an", "the"}:
                predicate_tokens = predicate_tokens[1:]
            predicate = " ".join(predicate_tokens).strip()
            if subject:
                slots["subject"] = subject
            if predicate:
                slots["predicate"] = predicate
                slots["designation"] = predicate
        if surface.endswith("?"):
            stop = {"what", "is", "are", "a", "an", "the", "do", "does", "did", "you", "me", "tell", "about"}
            focus = [tok for tok in toks if tok not in stop]
            if focus:
                slots["query_target"] = " ".join(focus[-3:]).strip()
        return slots

    def _observe_text_into_indexes(
        self,
        token_rows: dict[str, dict[str, Any]],
        edge_rows: dict[str, dict[str, Any]],
        phrase_rows: dict[str, dict[str, Any]],
        text: str,
        *,
        role: str,
        source: str,
        strength: str,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        surface = norm_text(text)
        toks = tokens_for(surface)
        ts = time.time()
        if not toks:
            return {"ok": False, "reason": "no_tokens", "input": clean_text(text), "memory_updates": []}

        updates: list[str] = []
        tokids: list[str] = []
        for token in toks:
            tokid = stable_id("tok", token, size=8)
            tokids.append(tokid)
            token_rows[tokid] = self._merge_token(token_rows.get(tokid), token=token, tokid=tokid, role=role, source=source, ts=ts)
            updates.append(f"token:{token}")

        edge_scores: list[float] = []
        for left, right in zip(toks, toks[1:]):
            edge_id = stable_id("edge", left, right, "next", size=8)
            edge_rows[edge_id] = self._merge_edge(edge_rows.get(edge_id), left=left, right=right, edge_id=edge_id, role=role, source=source, ts=ts)
            edge_scores.append(float(edge_rows[edge_id].get("confidence", 0.0) or 0.0))
            updates.append(f"edge:{left}>{right}")

        intent, route, response_type, note, base_conf = self._infer_route(surface, toks)
        slots = self._slots_for_phrase(surface, toks, intent=intent, route=route)
        pattern_id = stable_id("phrase", surface, intent, route, size=8)
        phrase_rows[pattern_id] = self._merge_phrase(
            phrase_rows.get(pattern_id),
            pattern_id=pattern_id,
            surface=surface,
            toks=toks,
            tokids=tokids,
            intent=intent,
            route=route,
            response_type=response_type,
            base_conf=base_conf,
            strength=strength,
            role=role,
            source=source,
            note=note,
            ts=ts,
            meta=meta,
            slots=slots,
        )
        updates.append(f"phrase:{intent}:{route}")
        adjacency_score = sum(edge_scores) / max(1, len(edge_scores)) if len(toks) > 1 else 0.4
        confidence = min(1.0, (adjacency_score * 0.42) + (float(phrase_rows[pattern_id].get("confidence", 0.0) or 0.0) * 0.58))
        return {
            "ok": True,
            "input": clean_text(text),
            "normalized": surface,
            "tok_chain": tokids,
            "token_text": toks,
            "adjacency_score": round(adjacency_score, 6),
            "phrase_match": pattern_id,
            "intent": intent,
            "route": route,
            "expected_response_type": response_type,
            "confidence": round(confidence, 6),
            "needs_deeper_lookup": route in {"bounded_lookup", "pattern/unknown"},
            "slots": slots,
            "memory_updates": updates,
        }

    def _merge_token(self, before: Mapping[str, Any] | None, *, token: str, tokid: str, role: str, source: str, ts: float) -> dict[str, Any]:
        row = dict(before or {})
        use_count = int(row.get("use_count", 0) or 0) + 1
        row.update({
            "schema": "tok_graph.token.v1",
            "tokid": tokid,
            "text": token,
            "surface": token,
            "use_count": use_count,
            "confidence": min(1.0, 0.45 + min(0.45, use_count * 0.015)),
            "last_role": role,
            "last_source": source,
            "last_seen": ts,
        })
        row.setdefault("created_ts", ts)
        row.setdefault("meaning_refs", [])
        row.setdefault("phrase_refs", [])
        return row

    def _merge_edge(self, before: Mapping[str, Any] | None, *, left: str, right: str, edge_id: str, role: str, source: str, ts: float) -> dict[str, Any]:
        row = dict(before or {})
        use_count = int(row.get("use_count", 0) or 0) + 1
        row.update({
            "schema": "tok_graph.adjacency.v1",
            "edge_id": edge_id,
            "left_tokid": stable_id("tok", left, size=8),
            "right_tokid": stable_id("tok", right, size=8),
            "left_text": left,
            "right_text": right,
            "relation": "next_token",
            "use_count": use_count,
            "confidence": min(1.0, 0.30 + min(0.55, use_count * 0.025)),
            "last_role": role,
            "last_source": source,
            "last_seen": ts,
        })
        row.setdefault("created_ts", ts)
        return row

    def _merge_phrase(
        self,
        before: Mapping[str, Any] | None,
        *,
        pattern_id: str,
        surface: str,
        toks: Sequence[str],
        tokids: Sequence[str],
        intent: str,
        route: str,
        response_type: str,
        base_conf: float,
        strength: str,
        role: str,
        source: str,
        note: str,
        ts: float,
        meta: Mapping[str, Any] | None,
        slots: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = dict(before or {})
        use_count = int(row.get("use_count", 0) or 0) + 1
        boost = 0.12 if strength == "trainer" else 0.025
        prior = float(row.get("confidence", base_conf) or base_conf)
        row.update({
            "schema": "tok_graph.phrase_pattern.v1",
            "pattern_id": pattern_id,
            "surface": surface,
            "tok_chain": list(tokids),
            "token_text": list(toks),
            "intent": intent,
            "route": route,
            "expected_response_type": response_type,
            "confidence": min(1.0, max(base_conf, prior + boost)),
            "use_count": use_count,
            "last_role": role,
            "last_source": source,
            "last_seen": ts,
            "conversion_note": note,
            "weak_candidate": route == "pattern/unknown" and strength != "trainer",
            "meta": dict(meta or {}),
            "slots": dict(slots or {}),
        })
        row.setdefault("created_ts", ts)
        return row

    def _find_by_key(self, path: Path, key_name: str, key_value: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict) and str(row.get(key_name, "") or "") == key_value:
                        return row
        except Exception:
            return None
        return None

    def _read_index_by_key(self, path: Path, key_name: str) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return rows
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    key = str(row.get(key_name, "") or "")
                    if key:
                        rows[key] = row
        except Exception:
            return rows
        return rows

    def _write_index_values(self, path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as out:
                for row in rows.values():
                    if isinstance(row, Mapping):
                        out.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _upsert_by_key(self, path: Path, key_name: str, key_value: str, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        replaced = False
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as out:
                if path.exists():
                    with path.open("r", encoding="utf-8", errors="replace") as inp:
                        for line in inp:
                            try:
                                existing = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(existing, dict) and str(existing.get(key_name, "") or "") == key_value:
                                out.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
                                replaced = True
                            elif isinstance(existing, dict):
                                out.write(json.dumps(existing, ensure_ascii=False, sort_keys=True) + "\n")
                if not replaced:
                    out.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
