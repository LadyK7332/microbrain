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
        self.derived_dir = self.base_dir / "mem_cell_derived"
        self.legacy_dir = self.mem_cell_dir / "_legacy_shards"
        self._stores: Dict[str, JSONLStore] = {}
        self._tier_rows: Dict[str, List[Dict[str, Any]]] = {}
        self._tier_index: Dict[str, Dict[str, int]] = {}
        self._tier_loaded: set[str] = set()
        for tier in TIERS:
            (self.mem_cell_dir / tier).mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _norm_text(text: str) -> str:
        return " ".join(TOKEN_RE.findall((text or "").lower())).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in TOKEN_RE.findall((text or "").lower()) if t]

    def _coerce_tier(self, tier: str) -> str:
        tier = str(tier or "now").strip().lower()
        return tier if tier in TIERS else "now"

    def _shard_path(self, tier: str) -> Path:
        """Return the canonical, rewritten file for one mem-cell tier.

        Older builds wrote daily append shards like short_20260510.jsonl.
        Newer builds keep one living file per tier and rewrite it atomically
        after merging/updating existing cells.
        """
        tier = self._coerce_tier(tier)
        return self.mem_cell_dir / tier / f"{tier}.jsonl"

    def _legacy_paths(self, tier: str) -> List[Path]:
        tier = self._coerce_tier(tier)
        tier_dir = self.mem_cell_dir / tier
        if not tier_dir.exists():
            return []
        canonical = self._shard_path(tier).resolve()
        return [
            path
            for path in sorted(tier_dir.glob(f"{tier}_*.jsonl"))
            if path.resolve() != canonical
        ]

    def _store_for(self, tier: str) -> JSONLStore:
        tier = self._coerce_tier(tier)
        if tier not in self._stores:
            self._stores[tier] = JSONLStore(str(self._shard_path(tier)))
        return self._stores[tier]

    def _rebuild_tier_index(self, tier: str) -> None:
        tier = self._coerce_tier(tier)
        self._tier_index[tier] = {
            str(row.get("id", "") or ""): idx
            for idx, row in enumerate(self._tier_rows.get(tier, []))
            if isinstance(row, dict) and str(row.get("id", "") or "")
        }

    def _merge_cell_rows(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        *,
        touch: bool = False,
    ) -> Dict[str, Any]:
        now_ts = time.time()
        old = dict(existing or {})
        new = dict(incoming or {})
        merged = dict(old)

        old_ts = float(old.get("ts", old.get("last_seen", now_ts)) or now_ts)
        new_ts = float(new.get("ts", new.get("last_seen", now_ts)) or now_ts)
        merged["ts"] = min(old_ts, new_ts)
        merged["last_seen"] = max(
            float(old.get("last_seen", old_ts) or old_ts),
            float(new.get("last_seen", new_ts) or new_ts),
            now_ts if touch else 0.0,
        )
        merged["tier"] = self._coerce_tier(str(new.get("tier", old.get("tier", "now")) or "now"))
        merged["schema"] = str(new.get("schema", old.get("schema", "mem_cell.v1")) or "mem_cell.v1")

        old_count = max(1, int(old.get("encounter_count", 1) or 1))
        new_count = max(1, int(new.get("encounter_count", 1) or 1))
        merged["encounter_count"] = max(old_count, new_count) + (1 if touch else 0)
        merged["revision"] = max(int(old.get("revision", 0) or 0), int(new.get("revision", 0) or 0)) + (1 if touch else 0)

        merged["activation"] = min(
            1.0,
            max(float(old.get("activation", 0.0) or 0.0), float(new.get("activation", 0.0) or 0.0))
            + (0.08 if touch else 0.0),
        )
        merged["promotion"] = min(
            1.0,
            max(float(old.get("promotion", 0.0) or 0.0), float(new.get("promotion", 0.0) or 0.0))
            + (0.03 if touch else 0.0),
        )
        merged["decay"] = min(float(old.get("decay", 1.0) or 1.0), float(new.get("decay", 1.0) or 1.0))
        merged["trust"] = min(1.0, max(float(old.get("trust", 0.5) or 0.5), float(new.get("trust", 0.5) or 0.5)))
        merged["usage_count"] = max(int(old.get("usage_count", 0) or 0), int(new.get("usage_count", 0) or 0))
        merged["successful_recalls"] = max(int(old.get("successful_recalls", 0) or 0), int(new.get("successful_recalls", 0) or 0))
        merged["last_used_ts"] = max(float(old.get("last_used_ts", 0.0) or 0.0), float(new.get("last_used_ts", 0.0) or 0.0))

        for key, limit in (("refs", 32), ("modalities", 8), ("links_explicit", 24)):
            merged[key] = self._merge_unique_list(list(old.get(key, []) or []), list(new.get(key, []) or []), limit=limit)

        old_anchor = old.get("anchor", {}) if isinstance(old.get("anchor", {}), dict) else {}
        new_anchor = new.get("anchor", {}) if isinstance(new.get("anchor", {}), dict) else {}
        merged["anchor"] = {**old_anchor, **{k: v for k, v in new_anchor.items() if v not in (None, "", [])}}

        meta = dict(old.get("meta", {}) or {})
        meta.update(dict(new.get("meta", {}) or {}))
        merged["meta"] = meta
        return merged

    def _dedupe_rows(self, rows: Iterable[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
        tier = self._coerce_tier(tier)
        by_id: Dict[str, Dict[str, Any]] = {}
        anonymous: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row["tier"] = tier
            row_id = str(row.get("id", "") or "").strip()
            if not row_id:
                anonymous.append(row)
                continue
            if row_id in by_id:
                by_id[row_id] = self._merge_cell_rows(by_id[row_id], row, touch=False)
            else:
                by_id[row_id] = row
        return anonymous + list(by_id.values())

    def _archive_legacy_paths(self, tier: str, paths: Sequence[Path]) -> None:
        tier = self._coerce_tier(tier)
        archive_dir = self.legacy_dir / tier
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            try:
                if not path.exists():
                    continue
                target = archive_dir / f"{path.name}.bak"
                n = 1
                while target.exists():
                    target = archive_dir / f"{path.name}.{n}.bak"
                    n += 1
                path.replace(target)
            except Exception:
                continue

    def _load_tier_rows(self, tier: str) -> List[Dict[str, Any]]:
        tier = self._coerce_tier(tier)
        if tier in self._tier_loaded:
            return self._tier_rows.setdefault(tier, [])

        path = self._shard_path(tier)
        rows: List[Dict[str, Any]] = []
        canonical_count = 0
        if path.exists():
            try:
                canonical_rows = JSONLStore(str(path)).read_all()
                canonical_count = len(canonical_rows)
                rows.extend(canonical_rows)
            except Exception:
                canonical_count = 0

        legacy_paths = self._legacy_paths(tier)
        for legacy in legacy_paths:
            try:
                rows.extend(JSONLStore(str(legacy)).read_all())
            except Exception:
                continue

        rows = self._dedupe_rows(rows, tier)
        self._tier_rows[tier] = rows
        self._rebuild_tier_index(tier)
        self._tier_loaded.add(tier)

        # If we saw legacy daily shards or duplicates, collapse them into the
        # canonical file once and archive the old append shards.
        if legacy_paths or (path.exists() and len(rows) != canonical_count):
            self._write_shard(tier, rows)
            self._archive_legacy_paths(tier, legacy_paths)
        elif not path.exists():
            self._write_shard(tier, rows)
        return self._tier_rows[tier]

    def _read_shard(self, tier: str) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._load_tier_rows(tier)]

    def _write_shard(self, tier: str, rows: List[Dict[str, Any]]) -> None:
        tier = self._coerce_tier(tier)
        path = self._shard_path(tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._dedupe_rows(rows, tier)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        self._tier_rows[tier] = rows
        self._rebuild_tier_index(tier)
        self._tier_loaded.add(tier)
        if tier in self._stores:
            self._stores[tier] = JSONLStore(str(path))

    def flush_tier(self, tier: str) -> None:
        tier = self._coerce_tier(tier)
        self._write_shard(tier, self._load_tier_rows(tier))

    def _derived_path(self) -> Path:
        shard = time.strftime("derived_%Y%m%d.jsonl", time.localtime())
        return self.derived_dir / shard

    def _read_derived_rows(self) -> List[Dict[str, Any]]:
        path = self._derived_path()
        if not path.exists():
            return []
        try:
            return JSONLStore(str(path)).read_all()
        except Exception:
            return []

    def _write_derived_rows(self, rows: List[Dict[str, Any]]) -> None:
        path = self._derived_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _iter_derived_rows(self) -> Iterable[Dict[str, Any]]:
        if not self.derived_dir.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for path in sorted(self.derived_dir.glob("derived_*.jsonl")):
            try:
                rows.extend(JSONLStore(str(path)).read_all())
            except Exception:
                continue
        return rows

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
        # Canonical memory is update/merge based. append_cell remains as a
        # compatibility wrapper, but it no longer sprays duplicate rows.
        self.upsert_cell(cell, tier=tier, touch=True, flush=True)

    def upsert_cell(
        self,
        cell: Dict[str, Any],
        tier: str = "now",
        *,
        touch: bool = True,
        flush: bool = True,
    ) -> Dict[str, Any]:
        row = dict(cell or {})
        tier = self._coerce_tier(str(tier or row.get("tier", "now") or "now"))
        row.setdefault("tier", tier)
        row.setdefault("schema", "mem_cell.v1")
        now_ts = time.time()
        row.setdefault("ts", now_ts)
        row.setdefault("last_seen", now_ts)
        row.setdefault("encounter_count", 1)
        row.setdefault("revision", 0)
        row.setdefault("links_explicit", [])
        row.setdefault("refs", [])
        row.setdefault("modalities", [])
        row.setdefault("activation", 1.0)
        row.setdefault("promotion", 0.0)
        row.setdefault("decay", 1.0)
        row.setdefault("trust", 0.5)
        row.setdefault("usage_count", 0)
        row.setdefault("successful_recalls", 0)
        row.setdefault("last_used_ts", 0.0)

        rows = self._load_tier_rows(tier)
        index = self._tier_index.setdefault(tier, {})
        row_id = str(row.get("id", "") or "").strip()

        if not row_id or row_id not in index:
            rows.append(row)
            if row_id:
                index[row_id] = len(rows) - 1
            if flush:
                self._write_shard(tier, rows)
            return row

        existing_idx = index[row_id]
        merged = self._merge_cell_rows(dict(rows[existing_idx] or {}), row, touch=touch)
        rows[existing_idx] = merged
        if flush:
            self._write_shard(tier, rows)
        return merged


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


    def ingest_trainer_alignment(
        self,
        *,
        desired_text: str,
        context_query: str,
        bad_utterance: str = "",
        need: str = "",
        style: str = "",
        source: str = "trainer",
        meta: Optional[Dict[str, Any]] = None,
        tier: str = "learned",
    ) -> Dict[str, Any]:
        desired_clean = str(desired_text or "").strip()
        context_clean = str(context_query or "").strip()
        if not desired_clean or not context_clean:
            return {}

        trainer_meta = dict(meta or {})
        trainer_meta.update({
            "kind": "trainer_correction",
            "trainer_context": context_clean,
            "trainer_bad_utterance": str(bad_utterance or "").strip(),
            "trainer_need": str(need or "").strip(),
            "trainer_style": str(style or "").strip(),
            "trainer_source": str(source or "trainer").strip() or "trainer",
        })

        ingest_result = self.ingest_text(
            text=desired_clean,
            topic="trainer/correction",
            role="assistant",
            transport_source="trainer",
            source=str(source or "trainer"),
            meta=trainer_meta,
            tier=tier,
        )

        utterance = ingest_result.get("utterance", {}) if isinstance(ingest_result, dict) else {}
        utterance_id = str((utterance or {}).get("id", "") or "").strip()
        gp_ids = [str((c or {}).get("id", "") or "").strip() for c in ingest_result.get("general_patterns", []) if str((c or {}).get("id", "") or "").strip()]
        linker_ids = [str((c or {}).get("id", "") or "").strip() for c in ingest_result.get("linkers", []) if str((c or {}).get("id", "") or "").strip()]

        digest = hashlib.blake2b(
            f"trainer|{self._norm_text(context_clean)}|{self._norm_text(desired_clean)}|{str(need or '').strip().lower()}|{str(style or '').strip().lower()}".encode("utf-8", errors="ignore"),
            digest_size=8,
        ).hexdigest()
        now_ts = time.time()
        alignment = {
            "id": f"tr{digest}",
            "kind": "trainer_alignment",
            "tier": tier,
            "anchor": {"kind": "trainer/context", "ref": context_clean[:200], "norm": self._norm_text(context_clean)[:200]},
            "refs": [
                {"kind": "desired_utterance", "value": desired_clean},
                {"kind": "bad_utterance", "value": str(bad_utterance or "").strip()},
                {"kind": "need", "value": str(need or "").strip()},
                {"kind": "style", "value": str(style or "").strip()},
            ],
            "modalities": ["text"],
            "links_explicit": [cell_id for cell_id in ([utterance_id] + gp_ids[:4] + linker_ids[:4]) if cell_id],
            "activation": 1.0,
            "promotion": 0.38,
            "decay": 1.0,
            "trust": 0.96,
            "meta": {
                "role": "assistant",
                "kind": "trainer_correction",
                "desired_utterance": desired_clean,
                "bad_utterance": str(bad_utterance or "").strip(),
                "trainer_need": str(need or "").strip(),
                "trainer_style": str(style or "").strip(),
                "trainer_context": context_clean,
                "source": str(source or "trainer").strip() or "trainer",
            },
            "ts": now_ts,
            "last_seen": now_ts,
            "encounter_count": 1,
            "revision": 0,
        }
        alignment = self.upsert_cell(alignment, tier=tier)
        return {
            "utterance": utterance,
            "alignment": alignment,
            "general_patterns": ingest_result.get("general_patterns", []),
            "linkers": ingest_result.get("linkers", []),
        }

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
        utterance = self.upsert_cell(utterance, tier=tier, touch=True, flush=False)
        token_cells = [self.upsert_cell(c, tier=tier, touch=True, flush=False) for c in self.make_token_cells(text=text, parent_id=str(utterance.get('id','')), role=role, tier=tier)]
        pattern_cells = [self.upsert_cell(c, tier=tier, touch=True, flush=False) for c in self.make_pattern_cells(text=text, parent_id=str(utterance.get('id','')), token_cells=token_cells, role=role, tier=tier)]
        general_pattern_cells = [
            self.upsert_cell(c, tier=tier, touch=True, flush=False)
            for c in self.make_general_pattern_cells(
                text=text,
                parent_id=str(utterance.get('id','')),
                token_cells=token_cells,
                pattern_cells=pattern_cells,
                role=role,
                tier=tier,
            )
        ]
        linker_cells = [
            self.upsert_cell(c, tier=tier, touch=True, flush=False)
            for c in self.make_linker_cells(
                parent_id=str(utterance.get('id','')),
                general_pattern_cells=general_pattern_cells,
                token_cells=token_cells,
                role=role,
                tier=tier,
            )
        ]
        # back-link strongest immediate pieces into utterance
        if token_cells or pattern_cells or general_pattern_cells or linker_cells:
            utterance['links_explicit'] = self._merge_unique_list(
                list(utterance.get('links_explicit', []) or []),
                [
                    c.get('id')
                    for c in (
                        token_cells[:4]
                        + pattern_cells[:4]
                        + general_pattern_cells[:4]
                        + linker_cells[:4]
                    )
                    if c.get('id')
                ],
                limit=16,
            )
            utterance = self.upsert_cell(utterance, tier=tier, touch=False, flush=False)
        self.flush_tier(tier)
        return {
            'utterance': utterance,
            'tokens': token_cells,
            'patterns': pattern_cells,
            'general_patterns': general_pattern_cells,
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


    def note_cell_usage(
        self,
        cell_id: str,
        *,
        success: bool = False,
        activation_delta: float = 0.03,
        promotion_delta: float = 0.012,
    ) -> Optional[Dict[str, Any]]:
        target = str(cell_id or "").strip()
        if not target:
            return None

        now_ts = time.time()
        for tier in TIERS:
            rows = self._read_shard(tier)
            changed = False
            updated: Optional[Dict[str, Any]] = None
            for idx, row in enumerate(rows):
                if not isinstance(row, dict) or str(row.get("id", "") or "") != target:
                    continue
                row = dict(row)
                row["last_used_ts"] = now_ts
                row["usage_count"] = int(row.get("usage_count", 0) or 0) + 1
                if success:
                    row["successful_recalls"] = int(row.get("successful_recalls", 0) or 0) + 1
                row["activation"] = self._clamp01(float(row.get("activation", 0.0) or 0.0) + activation_delta)
                row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + promotion_delta)
                rows[idx] = row
                changed = True
                updated = row
                break
            if changed:
                self._write_shard(tier, rows)
                return updated

        rows = self._read_derived_rows()
        changed = False
        updated = None
        for idx, row in enumerate(rows):
            if not isinstance(row, dict) or str(row.get("id", "") or "") != target:
                continue
            row = dict(row)
            row["last_used_ts"] = now_ts
            row["usage_count"] = int(row.get("usage_count", 0) or 0) + 1
            if success:
                row["successful_recalls"] = int(row.get("successful_recalls", 0) or 0) + 1
            row["activation"] = self._clamp01(float(row.get("activation", 0.0) or 0.0) + activation_delta)
            row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + promotion_delta)
            rows[idx] = row
            changed = True
            updated = row
            break
        if changed:
            self._write_derived_rows(rows)
            return updated
        return None

    def _derived_value_score(self, row: Dict[str, Any]) -> float:
        activation = self._clamp01(row.get("activation", 0.0))
        promotion = self._clamp01(row.get("promotion", 0.0))
        trust = self._clamp01(row.get("trust", 0.0))
        support = max(1.0, float(row.get("support_count", row.get("encounter_count", 1)) or 1.0))
        usage = max(0.0, float(row.get("usage_count", 0) or 0.0))
        recalls = max(0.0, float(row.get("successful_recalls", 0) or 0.0))
        support_bonus = min(0.28, (support - 1.0) * 0.03)
        usage_bonus = min(0.22, usage * 0.02)
        recall_bonus = min(0.18, recalls * 0.03)
        return activation * 0.24 + promotion * 0.18 + trust * 0.14 + support_bonus + usage_bonus + recall_bonus

    def build_compressed_layer(
        self,
        *,
        source_tiers: Sequence[str] = ("long", "learned"),
        min_support_count: int = 2,
        min_encounter_sum: int = 3,
    ) -> Dict[str, int]:
        prior_rows = [row for row in self._iter_derived_rows() if isinstance(row, dict)]
        prior_by_id = {str(row.get("id", "") or ""): dict(row) for row in prior_rows if str(row.get("id", "") or "")}
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for tier in source_tiers:
            if tier not in TIERS:
                continue
            for row in self._iter_rows(tier):
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind", "") or "") != "general_pattern":
                    continue
                meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
                pattern_type = str(meta.get("pattern_type", "") or "").strip().lower()
                canonical = str(meta.get("canonical", "") or "").strip() or str((row.get("anchor", {}) if isinstance(row.get("anchor", {}), dict) else {}).get("ref", "") or "").strip()
                canonical_norm = self._norm_text(canonical)
                if not pattern_type or not canonical_norm:
                    continue
                groups.setdefault((pattern_type, canonical_norm), []).append(dict(row))

        built: List[Dict[str, Any]] = []
        stats = {"groups": 0, "written": 0, "skipped": 0}
        now_ts = time.time()

        for (pattern_type, canonical_norm), rows in groups.items():
            stats["groups"] += 1
            support_count = len(rows)
            encounter_sum = sum(max(1, int(r.get("encounter_count", 1) or 1)) for r in rows)
            if support_count < max(1, int(min_support_count)) and encounter_sum < max(1, int(min_encounter_sum)):
                stats["skipped"] += 1
                continue

            first_anchor = rows[0].get("anchor", {}) if isinstance(rows[0].get("anchor", {}), dict) else {}
            canonical = str(first_anchor.get("ref", "") or "").strip() or canonical_norm
            digest = hashlib.blake2b(f"derived|{pattern_type}|{canonical_norm}".encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
            derived_id = f"d{digest}"
            existing = dict(prior_by_id.get(derived_id, {}) or {})

            parent_ids: List[str] = []
            refs: List[Any] = []
            surfaces: List[str] = []
            source_tiers_seen: List[str] = []
            last_seen = 0.0
            first_seen = 0.0
            total_weight = 0.0
            activation_sum = 0.0
            promotion_sum = 0.0
            trust_sum = 0.0
            parent_usage = 0
            parent_recalls = 0

            slot_values: Dict[str, List[Any]] = {}
            for row in rows:
                meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
                tier = str(row.get("tier", "") or "").strip().lower()
                if tier and tier not in source_tiers_seen:
                    source_tiers_seen.append(tier)
                row_id = str(row.get("id", "") or "").strip()
                if row_id:
                    parent_ids.append(row_id)
                refs = self._merge_unique_list(refs, list(row.get("refs", []) or []), limit=24)
                surface = str(meta.get("surface", "") or "").strip()
                if surface:
                    surfaces = self._merge_unique_list(surfaces, [surface], limit=8)
                slots = meta.get("slots", {}) if isinstance(meta.get("slots", {}), dict) else {}
                for slot_name, slot_val in slots.items():
                    if slot_val in (None, "", []):
                        continue
                    slot_values.setdefault(str(slot_name), [])
                    slot_values[str(slot_name)] = self._merge_unique_list(slot_values[str(slot_name)], [slot_val], limit=8)
                seen_ts = float(row.get("last_seen", row.get("ts", 0.0)) or 0.0)
                ts = float(row.get("ts", seen_ts) or seen_ts)
                last_seen = max(last_seen, seen_ts)
                first_seen = ts if first_seen <= 0.0 else min(first_seen, ts)
                weight = max(1.0, float(row.get("encounter_count", 1) or 1))
                total_weight += weight
                activation_sum += self._clamp01(row.get("activation", 0.0)) * weight
                promotion_sum += self._clamp01(row.get("promotion", 0.0)) * weight
                trust_sum += self._clamp01(row.get("trust", 0.0)) * weight
                parent_usage += int(row.get("usage_count", 0) or 0)
                parent_recalls += int(row.get("successful_recalls", 0) or 0)

            activation = activation_sum / max(1.0, total_weight)
            promotion = min(1.0, (promotion_sum / max(1.0, total_weight)) + min(0.12, support_count * 0.02))
            trust = trust_sum / max(1.0, total_weight)
            usage_count = max(parent_usage, int(existing.get("usage_count", 0) or 0))
            successful_recalls = max(parent_recalls, int(existing.get("successful_recalls", 0) or 0))
            last_used_ts = max(float(existing.get("last_used_ts", 0.0) or 0.0), max([float(r.get("last_used_ts", 0.0) or 0.0) for r in rows] + [0.0]))

            support_refs: List[Dict[str, Any]] = [{"kind": "general_pattern", "value": canonical}]
            for slot_name, vals in slot_values.items():
                if not vals:
                    continue
                slot_val: Any = vals[0] if len(vals) == 1 else vals[:4]
                support_refs.append({"kind": "slot", "name": slot_name, "value": slot_val})
            for surface in surfaces[:4]:
                support_refs.append({"kind": "surface", "value": surface})
            refs = self._merge_unique_list(support_refs, refs, limit=24)

            derived_row = {
                "id": derived_id,
                "schema": "mem_cell.derived.v1",
                "kind": "compressed_general_pattern",
                "tier": "derived",
                "anchor": {"kind": f"compressed/{pattern_type}", "ref": canonical, "norm": canonical_norm},
                "refs": refs,
                "modalities": ["text"],
                "links_explicit": parent_ids[:32],
                "activation": round(float(activation), 6),
                "promotion": round(float(promotion), 6),
                "decay": 1.0,
                "trust": round(float(trust), 6),
                "usage_count": int(usage_count),
                "successful_recalls": int(successful_recalls),
                "last_used_ts": float(last_used_ts),
                "support_count": int(support_count),
                "encounter_count": int(encounter_sum),
                "derived_from": parent_ids[:32],
                "ts": first_seen or now_ts,
                "last_seen": last_seen or now_ts,
                "meta": {
                    "pattern_type": pattern_type,
                    "canonical": canonical,
                    "source_tiers": source_tiers_seen,
                    "support_examples": surfaces[:4],
                    "slots": {k: (v[0] if len(v) == 1 else v[:4]) for k, v in slot_values.items() if v},
                },
            }
            derived_row["score"] = round(float(self._derived_value_score(derived_row)), 6)
            built.append(derived_row)
            stats["written"] += 1

        built.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        self._write_derived_rows(built)
        return stats

    def prune_derived_layer(
        self,
        *,
        rows: Optional[Sequence[Dict[str, Any]]] = None,
        retention_hours: float = 336.0,
        max_rows: int = 512,
    ) -> Dict[str, int]:
        source_rows = [dict(r) for r in (rows if rows is not None else self._iter_derived_rows()) if isinstance(r, dict)]
        now_ts = time.time()
        best_by_norm: Dict[str, Dict[str, Any]] = {}
        pruned = 0

        for row in source_rows:
            row = dict(row)
            age_h = max(0.0, now_ts - float(row.get("last_seen", row.get("ts", now_ts)) or now_ts)) / 3600.0
            value = self._derived_value_score(row)
            row["score"] = round(float(value), 6)
            support = max(1, int(row.get("support_count", row.get("encounter_count", 1)) or 1))
            usage = int(row.get("usage_count", 0) or 0)
            recalls = int(row.get("successful_recalls", 0) or 0)
            if age_h > float(retention_hours or 336.0) and value < 0.24 and usage <= 0 and recalls <= 0 and support <= 1:
                pruned += 1
                continue
            norm = self._norm_text(str((row.get("anchor", {}) if isinstance(row.get("anchor", {}), dict) else {}).get("ref", "") or ""))
            if not norm:
                norm = str(row.get("id", "") or "")
            prior = best_by_norm.get(norm)
            if prior is None or float(row.get("score", 0.0) or 0.0) > float(prior.get("score", 0.0) or 0.0):
                best_by_norm[norm] = row

        kept = sorted(best_by_norm.values(), key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        if max_rows and len(kept) > int(max_rows):
            pruned += len(kept) - int(max_rows)
            kept = kept[: int(max_rows)]
        self._write_derived_rows(kept)
        return {"probed": len(source_rows), "pruned": pruned, "kept": len(kept)}

    def _iter_rows(self, tier: str) -> Iterable[Dict[str, Any]]:
        tier = self._coerce_tier(tier)
        return [dict(row) for row in self._load_tier_rows(tier)]

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

        tier_bias = {"derived": 1.16, "learned": 1.08, "long": 1.0, "now": 0.88, "short": 0.74}
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

        for row in self._iter_derived_rows():
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
            usage_count = max(0.0, float(row.get('usage_count', 0) or 0.0))
            recall_count = max(0.0, float(row.get('successful_recalls', 0) or 0.0))
            count_bonus = min(0.12, (encounter_count - 1.0) * 0.01) + min(0.08, usage_count * 0.015) + min(0.06, recall_count * 0.02)

            score = (
                (0.50 * token_score)
                + contain_bonus
                + (0.08 * recency)
                + (0.08 * activation)
                + (0.08 * promotion)
                + (0.06 * trust)
                + count_bonus
            ) * tier_bias.get('derived', 1.0)

            hits.append({
                'cell_id': str(row.get('id', '') or ''),
                'kind': str(row.get('kind', '') or ''),
                'tier': 'derived',
                'score': round(float(score), 6),
                'anchor': anchor,
                'anchor_text': anchor_text,
                'refs': ref_texts[:4],
                'modalities': list(row.get('modalities', []) or []),
                'links_explicit': list(row.get('links_explicit', []) or [])[:16],
                'ts': float(row.get('ts', 0.0) or 0.0),
                'last_seen': float(row.get('last_seen', row.get('ts', 0.0)) or 0.0),
                'activation': activation,
                'promotion': promotion,
                'trust': trust,
                'encounter_count': int(row.get('encounter_count', 1) or 1),
                'usage_count': int(row.get('usage_count', 0) or 0),
                'successful_recalls': int(row.get('successful_recalls', 0) or 0),
                'meta': dict(row.get('meta', {}) or {}),
            })

        hits.sort(key=lambda h: float(h.get('score', 0.0)), reverse=True)
        return hits[:max(1, int(limit))]
