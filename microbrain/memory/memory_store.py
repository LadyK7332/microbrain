# Imports
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import math
import hashlib
import json, threading, time

try:
    import onnxruntime as ort
except Exception:
    ort = None
try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None


class JSONLStore:
    def __init__(self, path: str):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):  # create empty file
                pass
        self._lock = threading.Lock()

    # inside JSONLStore.__init__

    def append(self, obj: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_all(self, rows: list[dict]) -> None:
        tmp = self.path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows or []:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)

    def read_all(self) -> list[dict]:
        with self._lock:
            items = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items


class ONNXEmbedder:
    """
    Simple ONNX embedding runner.
    Expects a model that takes 'input_ids' and 'attention_mask' and outputs either:
      - 'last_hidden_state' (B, T, D) or
      - 'pooler_output' (B, D)
    We apply mean pooling over token dimension if last_hidden_state is provided.
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path_or_json: str | None,
        provider: str = "DmlExecutionProvider",
        max_len: int = 256,
    ):
        self.sess = ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
        # Load tokenizer: if you pass a tokenizer JSON path, Tokenizer.from_file works.
        # If None, we fall back to a whitespace tokenizer (works but poorer quality).
        self.max_len = max_len
        self.tokenizer = None
        if (
            Tokenizer is not None
            and tokenizer_path_or_json
            and os.path.exists(tokenizer_path_or_json)
        ):
            self.tokenizer = Tokenizer.from_file(tokenizer_path_or_json)
        # else: leave self.tokenizer = None (will use whitespace fallback)

        outs = [o.name for o in self.sess.get_outputs()]
        self.out_last = "last_hidden_state" if "last_hidden_state" in outs else None
        self.out_pool = "pooler_output" if "pooler_output" in outs else None

    def _ws_tokenize(self, text: str):
        # whitespace tokenization into a tiny vocab: map tokens to pseudo IDs
        toks = text.strip().split()
        # cap length
        toks = toks[: self.max_len]
        # map to small integer IDs based on hash (no external vocab)
        ids = [(hash(t) % 30522) for t in toks]  # 30k-ish
        attn = [1] * len(ids)
        # pad to max_len
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            attn += [0] * pad
        return ids, attn

    def _hf_tokenize(self, text: str):
        # fast tokenizer JSON path case
        enc = self.tokenizer.encode(text)
        ids = enc.ids[: self.max_len]
        attn = [1] * len(ids)
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            attn += [0] * pad
        return ids, attn

    def embed(self, text: str) -> list[float]:
        # Build inputs
        if self.tokenizer:
            ids, attn = self._hf_tokenize(text)
        else:
            ids, attn = self._ws_tokenize(text)

        import numpy as np

        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.array([attn], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.sess.run(None, feeds)

        # Select output
        if self.out_pool:
            vec = outputs[[o.name for o in self.sess.get_outputs()].index(self.out_pool)][0]  # (D,)
        elif self.out_last:
            last = outputs[[o.name for o in self.sess.get_outputs()].index(self.out_last)][
                0
            ]  # (T,D)
            # mean pool over tokens with attention mask
            mask = attention_mask[0][: last.shape[0]].astype(float)
            mask = mask[:, None]
            denom = max(mask.sum(), 1.0)
            vec = (last * mask).sum(axis=0) / denom  # (D,)
        else:
            # fallback: flatten first output and truncate
            arr = outputs[0].reshape(-1)
            vec = arr[:384] if arr.size >= 384 else arr

        # L2 normalize
        n = float((vec**2).sum()) ** 0.5 or 1.0
        return (vec / n).tolist()


class SimpleHashEmbedder:
    """Minimal dependency-free embedding using a hashing trick (deterministic)."""

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.split():
            h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
            i = h % self.dim
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            vec[i] += sign
        # L2 normalize
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    # Allow call-style usage too: embedder("text")
    def __call__(self, text: str) -> list[float]:
        return self.embed(text)


class MemoryStore:
    """
    Persistent memory with optional Ollama embeddings.
    Writes to two JSONL files under base_dir:
      - semantic.jsonl (text, vec, meta, ts)
      - episodic.jsonl (text, meta, ts)
    """

    def __init__(
        self,
        memdir: str | None = None,
        onnx_embed_path: str | None = None,
        onnx_provider: str | None = None,
        tokenizer_path_or_json: str | None = None,
        onnx_max_len: int = 256,
        ollama: Optional[Any] = None,  # NEW: generic LLM client
    ):
        self.ollama = ollama
        # pick a default provider if none is passed
        if onnx_provider is None:
            onnx_provider = "DmlExecutionProvider" if os.name == "nt" else "CPUExecutionProvider"
        mem_root = memdir or os.getenv("MB_MEMDIR") or str(Path.cwd() / "memory")
        self.base_dir = Path(mem_root)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Build an embedder chain: LLM (if it implements .embed) -> ONNX -> Local hasher
        env_onnx = os.getenv("MB_ONNX_EMBED")
        env_tok = os.getenv("MB_TOKENIZER_JSON")

        use_onnx_path = onnx_embed_path or env_onnx
        use_tok_json = tokenizer_path_or_json or env_tok
        try:
            # 1) Prefer the live LLM if it exposes an embedding method
            if ollama is not None and hasattr(ollama, "embed"):
                embedder_obj = ollama  # treat the LLM client as the embedder

            # 2) Otherwise try ONNX (DirectML on Windows if available)
            elif use_onnx_path:
                embedder_obj = ONNXEmbedder(
                    onnx_path=use_onnx_path,
                    tokenizer_path_or_json=use_tok_json,
                    provider=onnx_provider,
                    max_len=onnx_max_len,
                )

            # 3) Last resort: cheap local hasher so we never crash
            else:
                embedder_obj = SimpleHashEmbedder(dim=256)

        except Exception:
            # Absolute last resort fallback, in case ONNX init fails, etc.
            embedder_obj = SimpleHashEmbedder(dim=256)

        self.embedder = embedder_obj
        self.semantic: list[dict] = []  # {text, vec, meta, ts}
        self.episodic: list[dict] = []  # {text, meta, ts}
        self.dim: int | None = None
        self.sem_file = JSONLStore(self.base_dir / "semantic.jsonl")
        self.epi_file = JSONLStore(self.base_dir / "episodic.jsonl")
        self.memdir = memdir
        self.onnx_embed_path = onnx_embed_path
        self.onnx_provider = onnx_provider
        self.onnx_max_len = onnx_max_len

        # Load existing items (if any)
        for row in self.sem_file.read_all():
            row = self._ensure_memory_schema(row)
            self.semantic.append(row)
            if self.dim is None and row.get("vec"):
                self.dim = len(row["vec"])

        for row in self.epi_file.read_all():
            row = self._ensure_memory_schema(row)
            self.episodic.append(row)

        compact_semantic = self._compact_loaded_memory(self.semantic)
        if len(compact_semantic) != len(self.semantic):
            self.semantic = compact_semantic
            self.sem_file.write_all(self.semantic)
        else:
            self.semantic = compact_semantic

        compact_episodic = self._compact_loaded_memory(self.episodic)
        if len(compact_episodic) != len(self.episodic):
            self.episodic = compact_episodic
            self.epi_file.write_all(self.episodic)
        else:
            self.episodic = compact_episodic


    def _ensure_memory_schema(self, item: dict) -> dict:
        """
        Ensure stable schema keys exist for future multi-sense tagging.
        This does NOT assert the senses exist yet; it only pre-creates empty slots.
        """
        if not isinstance(item, dict):
            return {"schema_ver": 2, "text": str(item), "meta": {"schema_ver": 2}, "ts": time.time(),
                    "senses_present": {"vision": False, "audio": False, "touch": False, "proprio": False},
                    "senses": {"vision": [], "audio": [], "touch": [], "proprio": []},
                    "sense_tags": {"vision": {"labels": [], "emb_ref": None, "assets": []},
                                   "audio": {"labels": [], "emb_ref": None, "assets": []},
                                   "touch": {"labels": [], "emb_ref": None, "assets": []},
                                   "proprio": {"labels": [], "emb_ref": None, "assets": []}},
                    "salience": {"score": 0.0, "valence": 0.0, "satisfaction": 0.0, "arousal": 0.0,
                                 "reinforce_sum": 0.0, "reinforce_count": 0, "last_reinforced_ts": None}
                    }

        # Schema upgrade (non-destructive): ensure we are at least v2
        try:
            if int(item.get("schema_ver", 0) or 0) < 2:
                item["schema_ver"] = 2
        except Exception:
            item["schema_ver"] = 2

        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            try:
                if int(meta.get("schema_ver", 0) or 0) < 2:
                    meta["schema_ver"] = 2
            except Exception:
                meta["schema_ver"] = 2

        # Predeclare multi-sense slots (empty for now)
        sp = item.setdefault("senses_present", {})
        s = item.setdefault("senses", {})
        st = item.setdefault("sense_tags", {})

        for ch in ("vision", "audio", "touch", "proprio"):
            sp.setdefault(ch, False)
            s.setdefault(ch, [])
            st.setdefault(ch, {"labels": [], "emb_ref": None, "assets": []})

        # Predeclare salience/valence channels (empty defaults for now)
        sal = item.setdefault("salience", {})
        if isinstance(sal, dict):
            sal.setdefault("score", 0.0)
            sal.setdefault("valence", 0.0)
            sal.setdefault("satisfaction", 0.0)
            sal.setdefault("arousal", 0.0)
            sal.setdefault("reinforce_sum", 0.0)
            sal.setdefault("reinforce_count", 0)
            sal.setdefault("last_reinforced_ts", None)
            sal.setdefault("reinforcement_pts", 0.0)
            sal.setdefault("salience_updated_ts", item.get("ts", time.time()))
            sal.setdefault("decay_half_life_s", 6.0 * 3600.0)

        item.setdefault("memory_key", self._memory_key(item.get("text", ""), item.get("meta", {})))
        item.setdefault("last_seen", item.get("ts", time.time()))
        item.setdefault("encounter_count", 1)
        item.setdefault("revision", 0)

        return item

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _effective_salience(self, item: dict, now_ts: float | None = None) -> dict[str, float]:
        now_ts = float(now_ts or time.time())
        sal = item.get("salience", {}) or {}
        if not isinstance(sal, dict):
            sal = {}

        score = self._safe_float(sal.get("score", 0.0), 0.0)
        valence = self._safe_float(sal.get("valence", 0.0), 0.0)
        satisfaction = self._safe_float(sal.get("satisfaction", 0.0), 0.0)
        arousal = self._safe_float(sal.get("arousal", 0.0), 0.0)
        reinforce_sum = self._safe_float(sal.get("reinforce_sum", 0.0), 0.0)
        reinforce_count = max(0.0, self._safe_float(sal.get("reinforce_count", 0.0), 0.0))
        reinforcement_pts = max(
            0.0,
            self._safe_float(
                sal.get("reinforcement_pts", (0.20 * max(0.0, reinforce_sum)) + (0.08 * reinforce_count)),
                0.0,
            ),
        )
        updated_ts = self._safe_float(sal.get("salience_updated_ts", item.get("ts", now_ts)), now_ts)
        half_life_s = max(60.0, self._safe_float(sal.get("decay_half_life_s", 6.0 * 3600.0), 6.0 * 3600.0))
        age_s = max(0.0, now_ts - updated_ts)
        decay = 0.5 ** (age_s / half_life_s) if half_life_s > 0.0 else 1.0

        return {
            "score": score * decay,
            "valence": valence * decay,
            "satisfaction": satisfaction * decay,
            "arousal": arousal * decay,
            "reinforcement_pts": reinforcement_pts,
            "reinforce_sum": reinforce_sum,
            "reinforce_count": reinforce_count,
            "decay": decay,
            "age_s": age_s,
        }

    @staticmethod
    def _norm_memory_text(text: str) -> str:
        return " ".join(str(text or "").lower().split()).strip()

    def _memory_key(self, text: str, meta: dict | None = None) -> str:
        meta = meta if isinstance(meta, dict) else {}
        role = str(meta.get("role", "") or "").strip().lower()
        kind = str(meta.get("kind", "") or "").strip().lower()
        channel = str(meta.get("channel", "") or "").strip().lower()
        norm = self._norm_memory_text(text)[:500]
        digest = hashlib.blake2b(f"{role}|{kind}|{channel}|{norm}".encode("utf-8", errors="ignore"), digest_size=12).hexdigest()
        return f"m{digest}"

    @staticmethod
    def _merge_unique(left: list, right: list, limit: int = 32) -> list:
        out = []
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

    def _merge_memory_item(self, existing: dict, incoming: dict) -> dict:
        now_ts = time.time()
        old = self._ensure_memory_schema(dict(existing or {}))
        new = self._ensure_memory_schema(dict(incoming or {}))
        merged = dict(old)
        merged["memory_key"] = str(old.get("memory_key") or new.get("memory_key") or self._memory_key(new.get("text", ""), new.get("meta", {})))
        merged["text"] = str(new.get("text") or old.get("text") or "")
        if new.get("vec"):
            merged["vec"] = new.get("vec")
        merged["ts"] = min(float(old.get("ts", now_ts) or now_ts), float(new.get("ts", now_ts) or now_ts))
        merged["last_seen"] = max(float(old.get("last_seen", old.get("ts", now_ts)) or now_ts), float(new.get("last_seen", new.get("ts", now_ts)) or now_ts), now_ts)
        merged["encounter_count"] = max(1, int(old.get("encounter_count", 1) or 1)) + 1
        merged["revision"] = int(old.get("revision", 0) or 0) + 1

        old_meta = dict(old.get("meta", {}) or {})
        old_meta.update(dict(new.get("meta", {}) or {}))
        old_meta.setdefault("schema_ver", 2)
        merged["meta"] = old_meta

        for ch in ("vision", "audio", "touch", "proprio"):
            merged.setdefault("senses_present", {}).setdefault(ch, False)
            merged.setdefault("senses", {}).setdefault(ch, [])
            merged["senses_present"][ch] = bool(
                old.get("senses_present", {}).get(ch, False) or new.get("senses_present", {}).get(ch, False)
            )
            merged["senses"][ch] = self._merge_unique(
                list((old.get("senses", {}) or {}).get(ch, []) or []),
                list((new.get("senses", {}) or {}).get(ch, []) or []),
                limit=24,
            )

        old_sal = dict(old.get("salience", {}) or {})
        new_sal = dict(new.get("salience", {}) or {})
        sal = dict(old_sal)
        for k in ("score", "valence", "satisfaction", "arousal"):
            sal[k] = max(self._safe_float(old_sal.get(k, 0.0), 0.0), self._safe_float(new_sal.get(k, 0.0), 0.0))
        sal["reinforce_sum"] = self._safe_float(old_sal.get("reinforce_sum", 0.0), 0.0) + self._safe_float(new_sal.get("reinforce_sum", 0.0), 0.0)
        sal["reinforce_count"] = max(0, int(old_sal.get("reinforce_count", 0) or 0)) + max(0, int(new_sal.get("reinforce_count", 0) or 0))
        sal["last_reinforced_ts"] = max(
            self._safe_float(old_sal.get("last_reinforced_ts", 0.0), 0.0),
            self._safe_float(new_sal.get("last_reinforced_ts", 0.0), 0.0),
        ) or None
        sal["reinforcement_pts"] = max(
            self._safe_float(old_sal.get("reinforcement_pts", 0.0), 0.0),
            self._safe_float(new_sal.get("reinforcement_pts", 0.0), 0.0),
        )
        sal["salience_updated_ts"] = now_ts
        sal.setdefault("decay_half_life_s", old_sal.get("decay_half_life_s", new_sal.get("decay_half_life_s", 6.0 * 3600.0)))
        merged["salience"] = sal
        return self._ensure_memory_schema(merged)

    def _compact_loaded_memory(self, rows: list[dict]) -> list[dict]:
        by_key: dict[str, dict] = {}
        anonymous: list[dict] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            row = self._ensure_memory_schema(dict(row))
            key = str(row.get("memory_key") or self._memory_key(row.get("text", ""), row.get("meta", {})))
            row["memory_key"] = key
            if not key:
                anonymous.append(row)
                continue
            if key in by_key:
                by_key[key] = self._merge_memory_item(by_key[key], row)
            else:
                by_key[key] = row
        return anonymous + list(by_key.values())

    def _upsert_memory_item(self, rows: list[dict], store: JSONLStore, item: dict) -> dict:
        item = self._ensure_memory_schema(dict(item))
        key = str(item.get("memory_key") or self._memory_key(item.get("text", ""), item.get("meta", {})))
        item["memory_key"] = key
        for idx, existing in enumerate(rows):
            if not isinstance(existing, dict):
                continue
            existing_key = str(existing.get("memory_key") or self._memory_key(existing.get("text", ""), existing.get("meta", {})))
            if existing_key != key:
                continue
            merged = self._merge_memory_item(existing, item)
            rows[idx] = merged
            store.write_all(rows)
            return merged
        rows.append(item)
        store.write_all(rows)
        return item

    def add_semantic(self, text: str, meta: dict | None = None, salience: dict | None = None):
        # Try Ollama embeddings first; if unavailable, fall back to local
        try:
            if self.embedder:
                vec = self.embedder.embed(text)
            else:
                vec = self.ollama.embed(text) or _local_embed(text)
        except Exception:
            vec = _local_embed(text)

        if self.dim is None and vec:
            self.dim = len(vec)

        item = {"text": text, "vec": vec, "meta": meta or {}, "ts": time.time()}
        item["memory_key"] = self._memory_key(text, meta or {})
        if salience is not None:
            salience = dict(salience)
            salience.setdefault("salience_updated_ts", item["ts"])
            salience.setdefault("reinforcement_pts", max(0.0, self._safe_float(salience.get("reinforcement_pts", (0.20 * max(0.0, self._safe_float(salience.get("reinforce_sum", 0.0), 0.0))) + (0.08 * max(0.0, self._safe_float(salience.get("reinforce_count", 0.0), 0.0)))), 0.0)))
            item["salience"] = salience
        item = self._ensure_memory_schema(item)
        self._upsert_memory_item(self.semantic, self.sem_file, item)

    def add_episodic(self, text: str, meta: dict | None = None, salience: dict | None = None):
        item = {"text": text, "meta": meta or {}, "ts": time.time()}
        item["memory_key"] = self._memory_key(text, meta or {})
        if salience is not None:
            salience = dict(salience)
            salience.setdefault("salience_updated_ts", item["ts"])
            salience.setdefault("reinforcement_pts", max(0.0, self._safe_float(salience.get("reinforcement_pts", (0.20 * max(0.0, self._safe_float(salience.get("reinforce_sum", 0.0), 0.0))) + (0.08 * max(0.0, self._safe_float(salience.get("reinforce_count", 0.0), 0.0)))), 0.0)))
            item["salience"] = salience
        item = self._ensure_memory_schema(item)
        self._upsert_memory_item(self.episodic, self.epi_file, item)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return (dot / (na * nb)) if na and nb else 0.0

    def search_semantic(self, query: str, k: int = 5) -> list[dict]:
        try:
            if self.embedder:
                qv = self.embedder.embed(query)
            else:
                qv = self.ollama.embed(query) or _local_embed(query)
        except Exception:
            qv = _local_embed(query)

        scored = []
        now_ts = time.time()
        for it in self.semantic:
            sim = self._cosine(qv, it.get("vec", []))
            eff = self._effective_salience(it, now_ts=now_ts)
            satisfaction = float(eff.get("satisfaction", 0.0) or 0.0)
            valence = float(eff.get("valence", 0.0) or 0.0)
            score = float(eff.get("score", 0.0) or 0.0)
            reinforcement_pts = float(eff.get("reinforcement_pts", 0.0) or 0.0)
            reinforcement_bonus = min(0.20, reinforcement_pts * 0.04)

            # Salience shaping:
            # - transient salience fades unless refreshed
            # - reinforced items retain a modest retrieval advantage
            adj = sim + (0.26 * satisfaction) + (0.10 * valence) + (0.10 * score) + reinforcement_bonus
            scored.append((adj, sim, it))

        scored.sort(key=lambda x: -x[0])
        return [it for _, __, it in scored[:k]]

    def last_episodic(self, n: int = 3) -> list[dict]:
        return self.episodic[-n:]


def _local_embed(text: str) -> list[float]:
    # 256-dim hash vector; no dependencies
    vec = [0.0] * 256
    for b in text.encode("utf-8", errors="ignore"):
        vec[b % 256] += 1.0
    import math

    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]
