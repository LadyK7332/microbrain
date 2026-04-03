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
        row = self._ensure_row_defaults(dict(cell or {}), tier=tier)
        if "reinforcement_pts" not in row:
            row["reinforcement_pts"] = 0.0
        self._store_for(tier).append(row)

    def upsert_cell(self, cell: Dict[str, Any], tier: str = "now") -> Dict[str, Any]:
        row = dict(cell or {})
        tier = str(tier or row.get('tier', 'now') or 'now').lower()
        if tier not in TIERS:
            tier = 'now'
        now_ts = time.time()
        row = self._ensure_row_defaults(row, tier=tier, now_ts=now_ts)

        rows = self._read_shard(tier)
        row_id = str(row.get('id', '') or '')
        existing_idx = -1
        for i, existing in enumerate(rows):
            if isinstance(existing, dict) and str(existing.get('id', '') or '') == row_id:
                existing_idx = i
                break

        if existing_idx < 0:
            row['current_salience'] = max(row.get('current_salience', 0.0), 0.92)
            row['activation'] = row['current_salience']
            row['reinforcement_pts'] = max(0.0, self._safe_float(row.get('reinforcement_pts', 0.0), 0.0) + 0.08)
            row['last_reinforced_ts'] = now_ts
            row['salience_updated_ts'] = now_ts
            rows.append(row)
            self._write_shard(tier, rows)
            return row

        existing = self._ensure_row_defaults(dict(rows[existing_idx] or {}), tier=tier, now_ts=now_ts)
        current_salience = self._decayed_salience(existing, now_ts=now_ts, commit=True)
        existing['last_seen'] = now_ts
        existing['ts'] = existing.get('ts', now_ts)
        existing['encounter_count'] = int(existing.get('encounter_count', 1) or 1) + 1
        existing['revision'] = int(existing.get('revision', 0) or 0) + 1
        existing['current_salience'] = self._clamp01(current_salience + 0.16)
        existing['activation'] = existing['current_salience']
        existing['promotion'] = self._clamp01(float(existing.get('promotion', 0.0) or 0.0) + 0.03)
        existing['reinforcement_pts'] = max(0.0, self._safe_float(existing.get('reinforcement_pts', 0.0), 0.0) + 0.12)
        existing['last_reinforced_ts'] = now_ts
        existing['trust'] = min(1.0, max(float(existing.get('trust', 0.5) or 0.5), float(row.get('trust', 0.5) or 0.5)))
        existing['refs'] = self._merge_unique_list(list(existing.get('refs', []) or []), list(row.get('refs', []) or []), limit=24)
        existing['modalities'] = self._merge_unique_list(list(existing.get('modalities', []) or []), list(row.get('modalities', []) or []), limit=8)
        existing['links_explicit'] = self._merge_unique_list(list(existing.get('links_explicit', []) or []), list(row.get('links_explicit', []) or []), limit=16)

        meta = dict(existing.get('meta', {}) or {})
        meta.update(dict(row.get('meta', {}) or {}))
        existing['meta'] = meta
        rows[existing_idx] = existing
        self._write_shard(tier, rows)
        return existing

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
        # back-link strongest immediate pieces into utterance
        if token_cells or pattern_cells:
            utterance['links_explicit'] = self._merge_unique_list(
                list(utterance.get('links_explicit', []) or []),
                [c.get('id') for c in (token_cells[:4] + pattern_cells[:4]) if c.get('id')],
                limit=12,
            )
            utterance = self.upsert_cell(utterance, tier=tier)
        return {
            'utterance': utterance,
            'tokens': token_cells,
            'patterns': pattern_cells,
        }

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _tier_half_life_h(self, tier: str) -> float:
        tier = str(tier or "now").strip().lower()
        half_life = {
            "now": 6.0,
            "short": 18.0,
            "long": 48.0,
            "learned": 168.0,
        }
        return float(half_life.get(tier, 18.0))

    def _ensure_row_defaults(self, row: Dict[str, Any], tier: str | None = None, now_ts: float | None = None) -> Dict[str, Any]:
        row = dict(row or {})
        now_ts = float(now_ts or time.time())
        tier = str(tier or row.get("tier", "now") or "now").strip().lower()
        if tier not in TIERS:
            tier = "now"
        row["tier"] = tier
        row.setdefault("schema", "mem_cell.v1")
        row.setdefault("ts", now_ts)
        row.setdefault("last_seen", row["ts"])
        row.setdefault("encounter_count", 1)
        row.setdefault("revision", 0)
        row.setdefault("links_explicit", [])
        row.setdefault("refs", [])
        row.setdefault("modalities", [])
        row.setdefault("promotion", 0.0)
        row.setdefault("trust", 0.5)
        current_salience = row.get("current_salience", row.get("activation", 1.0))
        row["current_salience"] = self._clamp01(current_salience)
        row["activation"] = row["current_salience"]
        row.setdefault("reinforcement_pts", max(0.0, self._safe_float(row.get("promotion", 0.0), 0.0) * 0.5))
        row.setdefault("salience_updated_ts", row.get("last_seen", row.get("ts", now_ts)))
        row.setdefault("last_reinforced_ts", None)
        row.setdefault("decay", 1.0)
        return row

    def _decayed_salience(self, row: Dict[str, Any], now_ts: float | None = None, *, commit: bool = False) -> float:
        row = self._ensure_row_defaults(row, now_ts=now_ts)
        now_ts = float(now_ts or time.time())
        current = self._clamp01(row.get("current_salience", row.get("activation", 0.0)))
        updated_ts = self._safe_float(row.get("salience_updated_ts", row.get("last_seen", row.get("ts", now_ts))), now_ts)
        age_s = max(0.0, now_ts - updated_ts)
        half_life_s = max(300.0, self._tier_half_life_h(str(row.get("tier", "now") or "now")) * 3600.0)
        decay_factor = 0.5 ** (age_s / half_life_s) if half_life_s > 0.0 else 1.0
        decayed = self._clamp01(current * decay_factor)
        if commit:
            row["current_salience"] = decayed
            row["activation"] = decayed
            row["salience_updated_ts"] = now_ts
        return decayed

    def _retention_score(self, row: Dict[str, Any], now_ts: float | None = None) -> float:
        row = self._ensure_row_defaults(row, now_ts=now_ts)
        salience = self._decayed_salience(row, now_ts=now_ts, commit=False)
        promotion = self._clamp01(row.get("promotion", 0.0))
        trust = self._clamp01(row.get("trust", 0.0))
        encounters = max(1.0, self._safe_float(row.get("encounter_count", 1), 1.0))
        encounter_bonus = min(0.30, (encounters - 1.0) * 0.03)
        reinforcement_pts = max(0.0, self._safe_float(row.get("reinforcement_pts", 0.0), 0.0))
        reinforcement_bonus = min(0.60, reinforcement_pts * 0.12)
        return salience * 0.22 + promotion * 0.20 + trust * 0.12 + encounter_bonus + reinforcement_bonus

    def _value_score(self, row: Dict[str, Any]) -> float:
        return self._retention_score(row, now_ts=time.time())

    def bump_cell(
        self,
        cell_id: str,
        *,
        activation_delta: float = 0.02,
        promotion_delta: float = 0.008,
        reinforcement_delta: float = 0.05,
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
                if not isinstance(row, dict):
                    continue
                if str(row.get("id", "") or "") != target:
                    continue
                row = self._ensure_row_defaults(dict(row), tier=tier, now_ts=now_ts)
                current_salience = self._decayed_salience(row, now_ts=now_ts, commit=True)
                row["last_seen"] = now_ts
                row["current_salience"] = self._clamp01(current_salience + activation_delta)
                row["activation"] = row["current_salience"]
                row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + promotion_delta)
                row["reinforcement_pts"] = max(0.0, self._safe_float(row.get("reinforcement_pts", 0.0), 0.0) + reinforcement_delta)
                row["last_reinforced_ts"] = now_ts
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
                row = self._ensure_row_defaults(row, tier=tier, now_ts=now_ts)
                activation = self._decayed_salience(row, now_ts=now_ts, commit=False)
                if activation <= 0.04 or activation >= 0.72:
                    continue
                encounters = max(1.0, float(row.get("encounter_count", 1) or 1))
                novelty = max(0.0, 1.0 - min(encounters / 8.0, 1.0))
                age_s = max(0.0, now_ts - float(row.get("last_seen", row.get("ts", 0.0)) or 0.0))
                recency = max(0.0, 1.0 - min(age_s / 86400.0, 1.0))
                promotion = self._clamp01(row.get("promotion", 0.0))
                reinforcement = max(0.0, self._safe_float(row.get("reinforcement_pts", 0.0), 0.0))
                score = (0.42 * novelty) + (0.24 * recency) + (0.18 * (1.0 - activation)) + (0.08 * promotion) + min(0.08, reinforcement * 0.02)
                row["activation"] = activation
                row["current_salience"] = activation
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

        retention_floor = {
            "now": 0.22,
            "short": 0.32,
            "long": 0.42,
            "learned": 0.48,
        }
        min_reinforcement = {
            "now": 0.18,
            "short": 0.34,
            "long": 0.52,
            "learned": 0.64,
        }
        cold_floor = {
            "now": 0.08,
            "short": 0.06,
            "long": 0.04,
            "learned": 0.02,
        }
        prune_mode = {
            "now": "or",
            "short": "and",
            "long": "and",
            "learned": "and",
        }
        weak_prune_after_h = {
            "now": 8.0,
            "short": 24.0,
            "long": 72.0,
            "learned": 168.0,
        }
        stats = {"probed": 0, "promoted": 0, "pruned": 0, "kept": 0, "decayed": 0}
        now_ts = time.time()
        rewritten: Dict[str, List[Dict[str, Any]]] = {tier: [] for tier in TIERS}

        for tier in TIERS:
            for raw_row in self._iter_rows(tier):
                if not isinstance(raw_row, dict):
                    continue
                row = self._ensure_row_defaults(raw_row, tier=tier, now_ts=now_ts)
                salience_now = self._decayed_salience(row, now_ts=now_ts, commit=True)
                if salience_now < self._clamp01(raw_row.get("current_salience", raw_row.get("activation", salience_now))):
                    stats["decayed"] += 1

                age_h = max(0.0, now_ts - float(row.get("last_seen", row.get("ts", now_ts)) or now_ts)) / 3600.0
                retention_score = self._retention_score(row, now_ts=now_ts)
                encounters = max(1, int(row.get("encounter_count", 1) or 1))
                reinforcement_pts = max(0.0, self._safe_float(row.get("reinforcement_pts", 0.0), 0.0))
                target_tier = tier

                if tier == "now" and (encounters >= 3 or retention_score >= 0.40 or reinforcement_pts >= 0.42):
                    target_tier = "short"
                elif tier == "short" and (encounters >= 5 or retention_score >= 0.62 or reinforcement_pts >= 0.86):
                    target_tier = "long"
                elif tier == "long" and (encounters >= 8 or retention_score >= 0.86 or reinforcement_pts >= 1.50):
                    target_tier = "learned"

                if target_tier != tier:
                    row["tier"] = target_tier
                    row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) + 0.06)
                    rewritten[target_tier].append(row)
                    stats["promoted"] += 1
                    continue

                age_expired = age_h > retention.get(tier, 72.0)
                weak = (
                    reinforcement_pts < min_reinforcement.get(tier, 0.3)
                    and retention_score < retention_floor.get(tier, 0.3)
                    and salience_now < cold_floor.get(tier, 0.05)
                    and age_h > weak_prune_after_h.get(tier, 0.0)
                )
                mode = prune_mode.get(tier, "and")
                should_prune = (age_expired or weak) if mode == "or" else (age_expired and weak)

                if should_prune:
                    stats["pruned"] += 1
                    continue

                if age_expired:
                    row["promotion"] = self._clamp01(float(row.get("promotion", 0.0) or 0.0) * 0.985)
                    row["reinforcement_pts"] = max(0.0, reinforcement_pts * 0.997)

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

                row = self._ensure_row_defaults(row, tier=tier, now_ts=now_ts)
                age_s = max(0.0, now_ts - float(row.get('last_seen', row.get('ts', 0.0)) or 0.0))
                recency = max(0.0, 1.0 - min(age_s / 86400.0, 1.0))
                activation = self._decayed_salience(row, now_ts=now_ts, commit=False)
                promotion = max(0.0, min(1.0, float(row.get('promotion', 0.0) or 0.0)))
                trust = max(0.0, min(1.0, float(row.get('trust', 0.5) or 0.5)))
                encounter_count = max(1.0, float(row.get('encounter_count', 1) or 1))
                count_bonus = min(0.10, (encounter_count - 1.0) * 0.01)
                reinforcement_pts = max(0.0, self._safe_float(row.get('reinforcement_pts', 0.0), 0.0))
                reinforcement_bonus = min(0.12, reinforcement_pts * 0.03)

                score = (
                    (0.50 * token_score)
                    + contain_bonus
                    + (0.08 * recency)
                    + (0.07 * activation)
                    + (0.05 * promotion)
                    + (0.05 * trust)
                    + count_bonus
                    + reinforcement_bonus
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
                    'current_salience': activation,
                    'promotion': promotion,
                    'reinforcement_pts': reinforcement_pts,
                    'trust': trust,
                    'encounter_count': int(row.get('encounter_count', 1) or 1),
                    'meta': dict(row.get('meta', {}) or {}),
                })

        hits.sort(key=lambda h: float(h.get('score', 0.0)), reverse=True)
        return hits[:max(1, int(limit))]
