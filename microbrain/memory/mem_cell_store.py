from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from microbrain.memory.memory_store import JSONLStore

TIERS = ("now", "short", "long", "learned")
TOKEN_RE = re.compile(r"[a-z0-9']+")

DEIXIS_TOKENS = {"this", "that", "these", "those"}
DETERMINERS = {"a", "an", "the"} | DEIXIS_TOKENS
COPULA_TOKENS = {"is", "are", "was", "were", "be"}
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}
GREETING_TOKENS = {"hi", "hello", "hey", "yo", "howdy", "moin"}


class MemCellStore:
    """
    Lightweight, tiered memory-cell store.

    Current design goals:
      - grouped shard files under mem_cell/<tier>/
      - cells are small and revisable
      - repeated experience updates existing cells instead of spraying duplicates
      - text can be decomposed into utterance, token, and simple pattern cells
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.mem_cell_dir = self.base_dir / "mem_cell"
        self._stores: Dict[str, JSONLStore] = {}
        for tier in TIERS:
            (self.mem_cell_dir / tier).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _norm_text(text: str) -> str:
        return " ".join(TOKEN_RE.findall((text or "").lower())).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in TOKEN_RE.findall((text or "").lower()) if t]

    def _shard_path(self, tier: str) -> Path:
        tier = str(tier or "now").strip().lower()
        if tier not in TIERS:
            tier = "now"
        shard = time.strftime(f"{tier}_%Y%m%d.jsonl", time.localtime())
        return self.mem_cell_dir / tier / shard

    def _store_for(self, tier: str) -> JSONLStore:
        tier = str(tier or "now").strip().lower()
        if tier not in TIERS:
            tier = "now"
        if tier not in self._stores:
            self._stores[tier] = JSONLStore(str(self._shard_path(tier)))
        return self._stores[tier]

    def _read_shard(self, tier: str) -> List[Dict[str, Any]]:
        path = self._shard_path(tier)
        if not path.exists():
            return []
        try:
            return JSONLStore(str(path)).read_all()
        except Exception:
            return []

    def _write_shard(self, tier: str, rows: List[Dict[str, Any]]) -> None:
        path = self._shard_path(tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if tier in self._stores:
            self._stores[tier] = JSONLStore(str(path))

    @staticmethod
    def _merge_unique_list(left: List[Any], right: List[Any], limit: int = 16) -> List[Any]:
        out: List[Any] = []
        seen = set()
        for item in list(left or []) + list(right or []):
            key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else repr(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def append_cell(self, cell: Dict[str, Any], tier: str = "now") -> None:
        row = dict(cell or {})
        row.setdefault("tier", tier)
        row.setdefault("schema", "mem_cell.v1")
        row.setdefault("ts", time.time())
        row.setdefault("last_seen", row["ts"])
        row.setdefault("encounter_count", 1)
        row.setdefault("revision", 0)
        row.setdefault("links_explicit", [])
        row.setdefault("refs", [])
        row.setdefault("modalities", [])
        row.setdefault("activation", 1.0)
        row.setdefault("promotion", 0.0)
        row.setdefault("decay", 1.0)
        row.setdefault("trust", 0.5)
        self._store_for(tier).append(row)

    def upsert_cell(self, cell: Dict[str, Any], tier: str = "now") -> Dict[str, Any]:
        row = dict(cell or {})
        tier = str(tier or row.get('tier', 'now') or 'now').lower()
        if tier not in TIERS:
            tier = 'now'
        row.setdefault('tier', tier)
        row.setdefault('schema', 'mem_cell.v1')
        now_ts = time.time()
        row.setdefault('ts', now_ts)
        row.setdefault('last_seen', now_ts)
        row.setdefault('encounter_count', 1)
        row.setdefault('revision', 0)
        row.setdefault('links_explicit', [])
        row.setdefault('refs', [])
        row.setdefault('modalities', [])
        row.setdefault('activation', 1.0)
        row.setdefault('promotion', 0.0)
        row.setdefault('decay', 1.0)
        row.setdefault('trust', 0.5)

        rows = self._read_shard(tier)
        row_id = str(row.get('id', '') or '')
        existing_idx = -1
        for i, existing in enumerate(rows):
            if isinstance(existing, dict) and str(existing.get('id', '') or '') == row_id:
                existing_idx = i
                break

        if existing_idx < 0:
            rows.append(row)
            self._write_shard(tier, rows)
            return row

        existing = dict(rows[existing_idx] or {})
        existing['last_seen'] = now_ts
        existing['ts'] = existing.get('ts', now_ts)
        existing['encounter_count'] = int(existing.get('encounter_count', 1) or 1) + 1
        existing['revision'] = int(existing.get('revision', 0) or 0) + 1
        existing['activation'] = min(1.0, float(existing.get('activation', 0.5) or 0.5) + 0.08)
        existing['promotion'] = min(1.0, float(existing.get('promotion', 0.0) or 0.0) + 0.03)
        existing['trust'] = min(1.0, max(float(existing.get('trust', 0.5) or 0.5), float(row.get('trust', 0.5) or 0.5)))
        existing['refs'] = self._merge_unique_list(list(existing.get('refs', []) or []), list(row.get('refs', []) or []), limit=24)
        existing['modalities'] = self._merge_unique_list(list(existing.get('modalities', []) or []), list(row.get('modalities', []) or []), limit=8)
        existing['links_explicit'] = self._merge_unique_list(list(existing.get('links_explicit', []) or []), list(row.get('links_explicit', []) or []), limit=16)

        # merge meta shallowly while preserving prior values
        meta = dict(existing.get('meta', {}) or {})
        meta.update(dict(row.get('meta', {}) or {}))
        existing['meta'] = meta
        rows[existing_idx] = existing
        self._write_shard(tier, rows)
        return existing


    @staticmethod
    def _split_leading_determiner(tokens: Sequence[str]) -> Tuple[Optional[str], List[str]]:
        seq = [str(t or "").strip().lower() for t in (tokens or []) if str(t or "").strip()]
        if not seq:
            return None, []
        first = seq[0]
        if first in DETERMINERS:
            return first, seq[1:]
        return None, seq

    @staticmethod
    def _slice_token_ids(token_ids: Sequence[str], start: int, stop: int) -> List[str]:
        out: List[str] = []
        for token_id in list(token_ids or [])[max(0, int(start)):max(0, int(stop))]:
            token_id = str(token_id or "").strip()
            if token_id:
                out.append(token_id)
        return out

    @staticmethod
    def _canonical_text(parts: Sequence[str]) -> str:
        return " ".join([str(p or "").strip() for p in (parts or []) if str(p or "").strip()]).strip()

    def make_general_pattern_cells(
        self,
        *,
        text: str,
        parent_id: str,
        token_cells: Sequence[Dict[str, Any]],
        pattern_cells: Sequence[Dict[str, Any]],
        role: str,
        tier: str = "now",
    ) -> List[Dict[str, Any]]:
        tokens = [str((c.get("anchor", {}) or {}).get("ref", "") or "") for c in token_cells]
        token_ids = [str(c.get("id", "") or "") for c in token_cells]
        out: List[Dict[str, Any]] = []
        now_ts = time.time()
        text_str = str(text or "").strip()
        text_lower = text_str.lower()
        text_is_question = text_str.endswith("?")

        def add_general(
            *,
            pattern_type: str,
            canonical: str,
            slots: Dict[str, Any],
            start_idx: int = 0,
            stop_idx: Optional[int] = None,
            activation: float = 0.74,
            trust: Optional[float] = None,
        ) -> None:
            surface = self._canonical_text(tokens[start_idx:stop_idx]) or canonical
            digest = hashlib.blake2b(
                f"general|{pattern_type}|{canonical}".encode("utf-8", errors="ignore"),
                digest_size=8,
            ).hexdigest()
            refs = [{"kind": "general_pattern", "value": canonical}]
            for slot_name, slot_value in slots.items():
                if slot_value in (None, "", []):
                    continue
                refs.append({"kind": "slot", "name": str(slot_name), "value": slot_value})
            out.append({
                "id": f"g{digest}",
                "kind": "general_pattern",
                "tier": tier,
                "anchor": {"kind": f"pattern/general/{pattern_type}", "ref": canonical, "norm": self._norm_text(canonical)},
                "refs": refs,
                "modalities": ["text"],
                "links_explicit": [parent_id] + self._slice_token_ids(token_ids, start_idx, len(tokens) if stop_idx is None else stop_idx),
                "activation": activation,
                "promotion": 0.06,
                "decay": 1.0,
                "trust": trust if trust is not None else (0.66 if role == "user" else 0.52),
                "meta": {
                    "role": role,
                    "pattern_type": pattern_type,
                    "canonical": canonical,
                    "surface": surface,
                    "parent_id": parent_id,
                    "slots": dict(slots),
                },
                "ts": now_ts,
                "last_seen": now_ts,
                "encounter_count": 1,
                "revision": 0,
            })

        # Greeting / social-openers
        if tokens and tokens[0] in GREETING_TOKENS:
            add_general(
                pattern_type="social_greeting",
                canonical=f"greeting {tokens[0]}",
                slots={"greeting": tokens[0]},
                start_idx=0,
                stop_idx=min(len(tokens), 2),
                activation=0.62,
            )

        # Question intent scaffold
        if text_is_question and tokens:
            question_word = tokens[0] if tokens[0] in QUESTION_WORDS else None
            focus_tokens = [t for t in tokens if t not in QUESTION_WORDS and t not in DETERMINERS and t not in COPULA_TOKENS]
            focus = self._canonical_text(focus_tokens[:4])
            if question_word or focus:
                canonical_parts = [p for p in [question_word or "question", focus] if p]
                add_general(
                    pattern_type="question_about",
                    canonical=self._canonical_text(canonical_parts),
                    slots={"question_word": question_word, "focus": focus},
                    start_idx=0,
                    stop_idx=len(tokens),
                    activation=0.68,
                )

        # Existence frame: "there is/are ..."
        if len(tokens) >= 3 and tokens[0] == "there":
            try:
                copula_idx = next(i for i, tok in enumerate(tokens[1:], start=1) if tok in COPULA_TOKENS)
            except StopIteration:
                copula_idx = -1
            if copula_idx >= 1 and copula_idx < len(tokens) - 1:
                deixis, entity_tokens = self._split_leading_determiner(tokens[copula_idx + 1 :])
                entity = self._canonical_text(entity_tokens)
                if entity:
                    canonical = self._canonical_text(["there", tokens[copula_idx], entity])
                    add_general(
                        pattern_type="assert_existence",
                        canonical=canonical,
                        slots={"entity": entity, "deixis": deixis, "copula": tokens[copula_idx]},
                        start_idx=0,
                        stop_idx=len(tokens),
                        activation=0.76,
                    )

        # Attribute frame: "this car is fast" / "car is fast"
        copula_idx = -1
        for i, tok in enumerate(tokens):
            if tok in COPULA_TOKENS:
                copula_idx = i
                break
        if 0 < copula_idx < (len(tokens) - 1) and tokens[0] != "there":
            subj_tokens = tokens[:copula_idx]
            attr_tokens = tokens[copula_idx + 1 :]
            deixis, subj_core_tokens = self._split_leading_determiner(subj_tokens)
            subject = self._canonical_text(subj_core_tokens or subj_tokens)
            attribute = self._canonical_text(attr_tokens)
            if subject and attribute:
                canonical = self._canonical_text([subject, tokens[copula_idx], attribute])
                add_general(
                    pattern_type="assert_attribute",
                    canonical=canonical,
                    slots={"subject": subject, "attribute": attribute, "deixis": deixis, "copula": tokens[copula_idx]},
                    start_idx=0,
                    stop_idx=len(tokens),
                    activation=0.82,
                    trust=0.70 if role == "user" else 0.56,
                )

        # User redirected MB to another source/person.
        if role == "user" and len(tokens) >= 4 and any(tok in tokens for tok in ("ask", "check")):
            person = None
            location = None
            for i, tok in enumerate(tokens):
                if tok in ("ask", "check") and i + 1 < len(tokens):
                    person = tokens[i + 1]
                if tok == "in" and i + 1 < len(tokens):
                    location = self._canonical_text(tokens[i + 1 : i + 3])
            if person:
                canonical = self._canonical_text(["ask", person, location or "later"]).strip()
                add_general(
                    pattern_type="social_redirect",
                    canonical=canonical,
                    slots={"person": person, "location": location},
                    start_idx=0,
                    stop_idx=len(tokens),
                    activation=0.72,
                )

        return out

    def make_utterance_pattern_cells(
        self,
        *,
        text: str,
        parent_id: str,
        token_cells: Sequence[Dict[str, Any]],
        general_pattern_cells: Sequence[Dict[str, Any]],
        role: str,
        tier: str = "now",
    ) -> List[Dict[str, Any]]:
        tokens = [str((c.get("anchor", {}) or {}).get("ref", "") or "") for c in token_cells]
        token_ids = [str(c.get("id", "") or "") for c in token_cells]
        out: List[Dict[str, Any]] = []
        now_ts = time.time()
        clean = str(text or "").strip()
        lowered = clean.lower()

        def add_utterance(*, act_type: str, canonical: str, template: str, slots: Dict[str, Any], activation: float = 0.68) -> None:
            digest = hashlib.blake2b(
                f"utterance|{act_type}|{canonical}|{template}".encode("utf-8", errors="ignore"),
                digest_size=8,
            ).hexdigest()
            refs = [{"kind": "utterance_pattern", "value": canonical}]
            for slot_name, slot_value in slots.items():
                if slot_value in (None, "", []):
                    continue
                refs.append({"kind": "slot", "name": str(slot_name), "value": slot_value})
            out.append({
                "id": f"u{digest}",
                "kind": "utterance_pattern",
                "tier": tier,
                "anchor": {"kind": f"utterance/{act_type}", "ref": canonical, "norm": self._norm_text(canonical)},
                "refs": refs,
                "modalities": ["text"],
                "links_explicit": [parent_id] + token_ids[:6],
                "activation": activation,
                "promotion": 0.05,
                "decay": 1.0,
                "trust": 0.64 if role == "assistant" else 0.52,
                "meta": {
                    "role": role,
                    "act_type": act_type,
                    "canonical": canonical,
                    "template": template,
                    "surface": clean or canonical,
                    "parent_id": parent_id,
                    "slots": dict(slots),
                },
                "ts": now_ts,
                "last_seen": now_ts,
                "encounter_count": 1,
                "revision": 0,
            })

        question_gp = None
        for gp in general_pattern_cells or []:
            meta = gp.get("meta", {}) if isinstance(gp.get("meta", {}), dict) else {}
            if str(meta.get("pattern_type", "") or "") == "question_about":
                question_gp = meta
                break

        if tokens and tokens[0] in GREETING_TOKENS:
            greeting = tokens[0]
            add_utterance(
                act_type="greet_present",
                canonical=f"greet {greeting}",
                template="{greeting}. I'm here.",
                slots={"greeting": greeting},
                activation=0.66,
            )

        if question_gp is not None:
            slots = dict(question_gp.get("slots", {}) or {})
            focus = str(slots.get("focus", "") or "").strip()
            if focus:
                add_utterance(
                    act_type="clarify_focus",
                    canonical=f"clarify {focus}",
                    template="What should I optimize for on {focus}?",
                    slots={"focus": focus},
                    activation=0.70,
                )
            add_utterance(
                act_type="answer_start",
                canonical=f"answer {focus or 'thread'}",
                template="I want to answer that.",
                slots={"focus": focus},
                activation=0.64,
            )

        if lowered.startswith(("ok", "okay", "got it", "understood", "i hear", "right")):
            add_utterance(
                act_type="acknowledge",
                canonical="acknowledge thread",
                template="I hear the open thread.",
                slots={},
                activation=0.62,
            )

        if any(tok in tokens for tok in ("answer", "reply", "respond")):
            add_utterance(
                act_type="answer_start",
                canonical="answer open thread",
                template="I want to answer that.",
                slots={},
                activation=0.68,
            )

        if any(tok in tokens for tok in ("missing", "target", "outcome", "optimize")) or "what should" in lowered:
            add_utterance(
                act_type="clarify_target",
                canonical="clarify target",
                template="What outcome should I optimize for?",
                slots={},
                activation=0.72,
            )

        return out

    def make_linker_cells(
        self,
        *,
        parent_id: str,
        general_pattern_cells: Sequence[Dict[str, Any]],
        token_cells: Sequence[Dict[str, Any]],
        role: str,
        tier: str = "now",
    ) -> List[Dict[str, Any]]:
        token_ids = [str(c.get("id", "") or "") for c in token_cells]
        out: List[Dict[str, Any]] = []
        now_ts = time.time()
        for pattern_cell in general_pattern_cells or []:
            gp_id = str(pattern_cell.get("id", "") or "").strip()
            gp_meta = dict(pattern_cell.get("meta", {}) or {})
            pattern_type = str(gp_meta.get("pattern_type", "") or "").strip()
            slots = dict(gp_meta.get("slots", {}) or {})
            if not gp_id or not pattern_type:
                continue
            slot_parts: List[str] = []
            refs: List[Dict[str, Any]] = []
            for slot_name in ("subject", "attribute", "entity", "focus", "person", "location", "question_word", "deixis"):
                slot_value = slots.get(slot_name)
                if slot_value in (None, "", []):
                    continue
                slot_parts.append(f"{slot_name}:{slot_value}")
                refs.append({"kind": "slot", "name": slot_name, "value": slot_value})
            digest = hashlib.blake2b(
                f"linker|{pattern_type}|{gp_id}|{'|'.join(slot_parts)}".encode("utf-8", errors="ignore"),
                digest_size=8,
            ).hexdigest()
            out.append({
                "id": f"l{digest}",
                "kind": "pattern_linker",
                "tier": tier,
                "anchor": {
                    "kind": f"linker/{pattern_type}",
                    "ref": " | ".join(slot_parts) if slot_parts else pattern_type,
                    "norm": self._norm_text(" ".join(slot_parts) if slot_parts else pattern_type),
                },
                "refs": refs,
                "modalities": ["text"],
                "links_explicit": self._merge_unique_list([parent_id, gp_id], token_ids, limit=20),
                "activation": 0.70,
                "promotion": 0.04,
                "decay": 1.0,
                "trust": 0.62 if role == "user" else 0.50,
                "meta": {
                    "role": role,
                    "pattern_type": pattern_type,
                    "general_pattern_id": gp_id,
                    "parent_id": parent_id,
                },
                "ts": now_ts,
                "last_seen": now_ts,
                "encounter_count": 1,
                "revision": 0,
            })
        return out

    def make_text_cell(
        self,
        *,
        text: str,
        topic: str,
        role: str,
        transport_source: str,
        source: str,
        meta: Dict[str, Any] | None = None,
        tier: str = "now",
    ) -> Dict[str, Any]:
        clean = str(text or "").strip()
        norm = self._norm_text(clean)
        ts = time.time()
        digest = hashlib.blake2b(
            f"utterance|{role}|{norm}".encode("utf-8", errors="ignore"),
            digest_size=8,
        ).hexdigest()
        channel = str((meta or {}).get("channel", "") or "")
        anchor_kind = "text/thought" if channel == "thought" else "text/utterance"

        return {
            "id": f"u{digest}",
            "kind": "utterance_anchor",
            "tier": tier,
            "anchor": {
                "kind": anchor_kind,
                "ref": clean[:160],
                "norm": norm[:160],
            },
            "refs": [{"kind": "text", "value": clean}],
            "modalities": ["text"],
            "links_explicit": [],
            "activation": 1.0,
            "promotion": 0.0,
            "decay": 1.0,
            "trust": 0.7 if role == "user" else 0.55,
            "meta": {
                "topic": topic,
                "role": role,
                "transport_source": transport_source,
                "source": source,
                "channel": channel,
            },
            "ts": ts,
            "last_seen": ts,
            "encounter_count": 1,
            "revision": 0,
        }

    def make_token_cells(
        self,
        *,
        text: str,
        parent_id: str,
        role: str,
        tier: str = "now",
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for token in self._tokenize(text):
            digest = hashlib.blake2b(f"token|{token}".encode('utf-8', errors='ignore'), digest_size=8).hexdigest()
            out.append({
                "id": f"t{digest}",
                "kind": "token_anchor",
                "tier": tier,
                "anchor": {"kind": "text/token", "ref": token, "norm": token},
                "refs": [{"kind": "token", "value": token}],
                "modalities": ["text"],
                "links_explicit": [parent_id],
                "activation": 0.9,
                "promotion": 0.0,
                "decay": 1.0,
                "trust": 0.68 if role == 'user' else 0.53,
                "meta": {"role": role, "parent_id": parent_id},
                "ts": time.time(),
                "last_seen": time.time(),
                "encounter_count": 1,
                "revision": 0,
            })
        return out

    def make_pattern_cells(
        self,
        *,
        text: str,
        parent_id: str,
        token_cells: Sequence[Dict[str, Any]],
        role: str,
        tier: str = "now",
    ) -> List[Dict[str, Any]]:
        tokens = [str((c.get('anchor', {}) or {}).get('ref', '') or '') for c in token_cells]
        token_ids = [str(c.get('id', '') or '') for c in token_cells]
        out: List[Dict[str, Any]] = []
        # adjacent bigrams
        for i in range(len(tokens) - 1):
            seq = tokens[i:i+2]
            digest = hashlib.blake2b(f"pattern2|{' '.join(seq)}".encode('utf-8', errors='ignore'), digest_size=8).hexdigest()
            out.append({
                "id": f"p{digest}",
                "kind": "pattern_anchor",
                "tier": tier,
                "anchor": {"kind": "pattern/adjacent_bigram", "ref": ' '.join(seq), "norm": ' '.join(seq)},
                "refs": [{"kind": "pattern", "value": ' '.join(seq)}],
                "modalities": ["text"],
                "links_explicit": [parent_id] + [token_ids[i], token_ids[i+1]],
                "activation": 0.82,
                "promotion": 0.0,
                "decay": 1.0,
                "trust": 0.64 if role == 'user' else 0.50,
                "meta": {"role": role, "pattern_type": "adjacent_bigram", "parent_id": parent_id},
                "ts": time.time(),
                "last_seen": time.time(),
                "encounter_count": 1,
                "revision": 0,
            })
        # adjacent trigrams
        for i in range(len(tokens) - 2):
            seq = tokens[i:i+3]
            digest = hashlib.blake2b(f"pattern3|{' '.join(seq)}".encode('utf-8', errors='ignore'), digest_size=8).hexdigest()
            out.append({
                "id": f"p{digest}",
                "kind": "pattern_anchor",
                "tier": tier,
                "anchor": {"kind": "pattern/adjacent_trigram", "ref": ' '.join(seq), "norm": ' '.join(seq)},
                "refs": [{"kind": "pattern", "value": ' '.join(seq)}],
                "modalities": ["text"],
                "links_explicit": [parent_id] + token_ids[i:i+3],
                "activation": 0.78,
                "promotion": 0.0,
                "decay": 1.0,
                "trust": 0.62 if role == 'user' else 0.48,
                "meta": {"role": role, "pattern_type": "adjacent_trigram", "parent_id": parent_id},
                "ts": time.time(),
                "last_seen": time.time(),
                "encounter_count": 1,
                "revision": 0,
            })
        # simple determiner-ish role pattern for language settling
        if len(tokens) >= 2 and tokens[0] in {"a", "an", "the", "this", "that", "these", "those"}:
            seq = [tokens[0], tokens[1]]
            digest = hashlib.blake2b(f"pattern_role|det_object_intro|{' '.join(seq)}".encode('utf-8', errors='ignore'), digest_size=8).hexdigest()
            out.append({
                "id": f"r{digest}",
                "kind": "pattern_role",
                "tier": tier,
                "anchor": {"kind": "pattern/role", "ref": "det_object_intro", "norm": "det_object_intro"},
                "refs": [{"kind": "pattern_role", "value": "det_object_intro"}],
                "modalities": ["text"],
                "links_explicit": [parent_id, token_ids[0], token_ids[1]],
                "activation": 0.85,
                "promotion": 0.0,
                "decay": 1.0,
                "trust": 0.60 if role == 'user' else 0.46,
                "meta": {"role": role, "pattern_type": "det_object_intro", "surface": ' '.join(seq), "parent_id": parent_id},
                "ts": time.time(),
                "last_seen": time.time(),
                "encounter_count": 1,
                "revision": 0,
            })
        return out

    def ingest_text(
        self,
        *,
        text: str,
        topic: str,
        role: str,
        transport_source: str,
        source: str,
        meta: Optional[Dict[str, Any]] = None,
        tier: str = 'now',
    ) -> Dict[str, Any]:
        utterance = self.make_text_cell(
            text=text, topic=topic, role=role, transport_source=transport_source, source=source, meta=meta, tier=tier
        )
        utterance = self.upsert_cell(utterance, tier=tier)
        token_cells = [self.upsert_cell(c, tier=tier) for c in self.make_token_cells(text=text, parent_id=str(utterance.get('id','')), role=role, tier=tier)]
        pattern_cells = [self.upsert_cell(c, tier=tier) for c in self.make_pattern_cells(text=text, parent_id=str(utterance.get('id','')), token_cells=token_cells, role=role, tier=tier)]
        general_pattern_cells = [
            self.upsert_cell(c, tier=tier)
            for c in self.make_general_pattern_cells(
                text=text,
                parent_id=str(utterance.get('id','')),
                token_cells=token_cells,
                pattern_cells=pattern_cells,
                role=role,
                tier=tier,
            )
        ]
        utterance_pattern_cells = [
            self.upsert_cell(c, tier=tier)
            for c in self.make_utterance_pattern_cells(
                text=text,
                parent_id=str(utterance.get('id','')),
                token_cells=token_cells,
                general_pattern_cells=general_pattern_cells,
                role=role,
                tier=tier,
            )
        ]
        linker_cells = [
            self.upsert_cell(c, tier=tier)
            for c in self.make_linker_cells(
                parent_id=str(utterance.get('id','')),
                general_pattern_cells=general_pattern_cells,
                token_cells=token_cells,
                role=role,
                tier=tier,
            )
        ]
        # back-link strongest immediate pieces into utterance
        if token_cells or pattern_cells or general_pattern_cells or utterance_pattern_cells or linker_cells:
            utterance['links_explicit'] = self._merge_unique_list(
                list(utterance.get('links_explicit', []) or []),
                [
                    c.get('id')
                    for c in (
                        token_cells[:4]
                        + pattern_cells[:4]
                        + general_pattern_cells[:4]
                        + utterance_pattern_cells[:4]
                        + linker_cells[:4]
                    )
                    if c.get('id')
                ],
                limit=16,
            )
            utterance = self.upsert_cell(utterance, tier=tier)
        return {
            'utterance': utterance,
            'tokens': token_cells,
            'patterns': pattern_cells,
            'general_patterns': general_pattern_cells,
            'utterance_patterns': utterance_pattern_cells,
            'linkers': linker_cells,
        }

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    def _value_score(self, row: Dict[str, Any]) -> float:
        activation = self._clamp01(row.get("activation", 0.0))
        promotion = self._clamp01(row.get("promotion", 0.0))
        trust = self._clamp01(row.get("trust", 0.0))
        encounters = max(1.0, float(row.get("encounter_count", 1) or 1))
        encounter_bonus = min(0.35, (encounters - 1.0) * 0.04)
        return activation * 0.30 + promotion * 0.25 + trust * 0.15 + encounter_bonus

    def bump_cell(
        self,
        cell_id: str,
        *,
        activation_delta: float = 0.02,
        promotion_delta: float = 0.008,
    ) -> Optional[Dict[str, Any]]:
        target = str(cell_id or "").strip()
        if not target:
            return None

        for tier in TIERS:
            rows = self._read_shard(tier)
            changed = False
            updated: Optional[Dict[str, Any]] = None
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                if str(row.get("id", "") or "") != target:
                    continue
                row = dict(row)
                row["last_seen"] = time.time()
                row["activation"] = self._clamp01(float(row.get("activation", 0.0) or 0.0) + activation_delta)
                row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + promotion_delta)
                rows[idx] = row
                changed = True
                updated = row
                break
            if changed:
                self._write_shard(tier, rows)
                return updated
        return None

    def probe_candidates(
        self,
        *,
        limit: int = 24,
        tiers: Sequence[str] = ("now", "short", "long"),
    ) -> List[Dict[str, Any]]:
        now_ts = time.time()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for tier in tiers:
            for row in self._iter_rows(tier):
                if not isinstance(row, dict):
                    continue
                kind = str(row.get("kind", "") or "")
                if kind not in {"token_anchor", "pattern_anchor", "pattern_role", "utterance_anchor"}:
                    continue
                activation = self._clamp01(row.get("activation", 0.0))
                if activation <= 0.04 or activation >= 0.72:
                    continue
                encounters = max(1.0, float(row.get("encounter_count", 1) or 1))
                novelty = max(0.0, 1.0 - min(encounters / 8.0, 1.0))
                age_s = max(0.0, now_ts - float(row.get("last_seen", row.get("ts", 0.0)) or 0.0))
                recency = max(0.0, 1.0 - min(age_s / 86400.0, 1.0))
                promotion = self._clamp01(row.get("promotion", 0.0))
                score = (0.46 * novelty) + (0.30 * recency) + (0.16 * (1.0 - activation)) + (0.08 * promotion)
                scored.append((score, dict(row)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [row for _score, row in scored[:max(1, int(limit))]]

    def maintain_lifecycle(
        self,
        *,
        retention_hours: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        retention = {
            "now": 36.0,
            "short": 72.0,
            "long": 96.0,
            "learned": 336.0,
        }
        if isinstance(retention_hours, dict):
            for tier, value in retention_hours.items():
                if tier in retention:
                    try:
                        retention[tier] = float(value)
                    except Exception:
                        pass

        prune_floor = {
            "now": 0.28,
            "short": 0.34,
            "long": 0.42,
            "learned": 0.16,
        }
        stats = {"probed": 0, "promoted": 0, "pruned": 0, "kept": 0}
        now_ts = time.time()
        rewritten: Dict[str, List[Dict[str, Any]]] = {tier: [] for tier in TIERS}

        for tier in TIERS:
            for row in self._iter_rows(tier):
                if not isinstance(row, dict):
                    continue
                row = dict(row)
                age_h = max(0.0, now_ts - float(row.get("last_seen", row.get("ts", now_ts)) or now_ts)) / 3600.0
                value = self._value_score(row)
                encounters = max(1, int(row.get("encounter_count", 1) or 1))
                target_tier = tier

                if tier == "now" and (encounters >= 3 or value >= 0.36):
                    target_tier = "short"
                elif tier == "short" and (encounters >= 5 or value >= 0.54):
                    target_tier = "long"
                elif tier == "long" and (encounters >= 8 or value >= 0.76):
                    target_tier = "learned"

                if target_tier != tier:
                    row["tier"] = target_tier
                    row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + 0.06)
                    stats["promoted"] += 1
                    rewritten[target_tier].append(row)
                    continue

                if age_h > retention.get(tier, 72.0):
                    if value < prune_floor.get(tier, 0.30):
                        stats["pruned"] += 1
                        continue
                    row["activation"] = self._clamp01(float(row.get("activation", 0.0) or 0.0) * 0.94)
                    row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) * 0.985)

                rewritten[tier].append(row)
                stats["kept"] += 1

        for tier in TIERS:
            self._write_shard(tier, rewritten[tier])
        return stats

    def _iter_rows(self, tier: str) -> Iterable[Dict[str, Any]]:
        tier = str(tier or "now").strip().lower()
        if tier not in TIERS:
            return []
        tier_dir = self.mem_cell_dir / tier
        if not tier_dir.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for path in sorted(tier_dir.glob(f"{tier}_*.jsonl")):
            try:
                rows.extend(JSONLStore(str(path)).read_all())
            except Exception:
                continue
        return rows

    def search_text_cells(
        self,
        query_text: str,
        *,
        limit: int = 8,
        tiers: Sequence[str] = ("learned", "long", "now", "short"),
    ) -> List[Dict[str, Any]]:
        q = str(query_text or '').strip()
        if not q:
            return []
        q_norm = self._norm_text(q)
        q_tokens = set(self._tokenize(q))
        if not q_tokens:
            return []

        tier_bias = {"learned": 1.08, "long": 1.0, "now": 0.88, "short": 0.74}
        hits: List[Dict[str, Any]] = []
        now_ts = time.time()

        for tier in tiers:
            for row in self._iter_rows(tier):
                if not isinstance(row, dict):
                    continue
                anchor = row.get('anchor', {}) if isinstance(row.get('anchor', {}), dict) else {}
                anchor_text = str(anchor.get('ref', '') or '').strip()
                ref_texts: List[str] = []
                for ref in row.get('refs', []) if isinstance(row.get('refs', []), list) else []:
                    if isinstance(ref, dict):
                        val = str(ref.get('value', '') or '').strip()
                        if val:
                            ref_texts.append(val)
                candidate = anchor_text or (ref_texts[0] if ref_texts else '')
                if not candidate:
                    continue

                c_norm = self._norm_text(candidate)
                c_tokens = set(self._tokenize(candidate))
                if not c_tokens:
                    continue

                overlap = len(q_tokens & c_tokens)
                if overlap <= 0 and q_norm not in c_norm and c_norm not in q_norm:
                    continue

                token_score = overlap / max(1.0, float(len(q_tokens | c_tokens)))
                contain_bonus = 0.0
                if q_norm and q_norm in c_norm:
                    contain_bonus += 0.35
                elif c_norm and c_norm in q_norm:
                    contain_bonus += 0.22

                age_s = max(0.0, now_ts - float(row.get('last_seen', row.get('ts', 0.0)) or 0.0))
                recency = max(0.0, 1.0 - min(age_s / 86400.0, 1.0))
                activation = max(0.0, min(1.0, float(row.get('activation', 1.0) or 1.0)))
                promotion = max(0.0, min(1.0, float(row.get('promotion', 0.0) or 0.0)))
                trust = max(0.0, min(1.0, float(row.get('trust', 0.5) or 0.5)))
                encounter_count = max(1.0, float(row.get('encounter_count', 1) or 1))
                count_bonus = min(0.10, (encounter_count - 1.0) * 0.01)

                score = (
                    (0.52 * token_score)
                    + contain_bonus
                    + (0.08 * recency)
                    + (0.08 * activation)
                    + (0.06 * promotion)
                    + (0.05 * trust)
                    + count_bonus
                ) * tier_bias.get(str(row.get('tier', tier) or tier), 0.7)

                hits.append({
                    'cell_id': str(row.get('id', '') or ''),
                    'kind': str(row.get('kind', '') or ''),
                    'tier': str(row.get('tier', tier) or tier),
                    'score': round(float(score), 6),
                    'anchor': anchor,
                    'anchor_text': anchor_text,
                    'refs': ref_texts[:3],
                    'modalities': list(row.get('modalities', []) or []),
                    'links_explicit': list(row.get('links_explicit', []) or [])[:12],
                    'ts': float(row.get('ts', 0.0) or 0.0),
                    'last_seen': float(row.get('last_seen', row.get('ts', 0.0)) or 0.0),
                    'activation': activation,
                    'promotion': promotion,
                    'trust': trust,
                    'encounter_count': int(row.get('encounter_count', 1) or 1),
                    'meta': dict(row.get('meta', {}) or {}),
                })

        hits.sort(key=lambda h: float(h.get('score', 0.0)), reverse=True)
        return hits[:max(1, int(limit))]
