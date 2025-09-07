# Imports
from typing import Optional, List


class JSONLStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):  # create empty file
                pass
        self._lock = threading.Lock()

# inside JSONLStore.__init__

    def append(self, obj: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        with self._lock:
            items = []
            with open(self.path, "r", encoding="utf-8") as f:
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
    def __init__(self, onnx_path: str, tokenizer_path_or_json: str | None,
                 provider: str = "DmlExecutionProvider", max_len: int = 256):
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=[provider, "CPUExecutionProvider"]
        )
        # Load tokenizer: if you pass a tokenizer JSON path, Tokenizer.from_file works.
        # If None, we fall back to a whitespace tokenizer (works but poorer quality).
        self.max_len = max_len
        self.tokenizer = None
        if Tokenizer is not None and tokenizer_path_or_json and os.path.exists(tokenizer_path_or_json):
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
            last = outputs[[o.name for o in self.sess.get_outputs()].index(self.out_last)][0]  # (T,D)
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

class MemoryStore:
    """
    Persistent memory with optional Ollama embeddings.
    Writes to two JSONL files under base_dir:
      - semantic.jsonl (text, vec, meta, ts)
      - episodic.jsonl (text, meta, ts)
    """
    def __init__(self, memdir: Optional[str] = None, onnx_embed_path: Optional[str] = None, onnx_provider: Optional[str] = None, onnx_max_len: int = 256):
        self.ollama = ollama
        self.base_dir = base_dir
        self.embedder = embedder
        self.semantic: list[dict] = []  # {text, vec, meta, ts}
        self.episodic: list[dict] = []  # {text, meta, ts}
        self.dim: int | None = None
        self.sem_file = JSONLStore(os.path.join(base_dir, "semantic.jsonl"))
        self.epi_file = JSONLStore(os.path.join(base_dir, "episodic.jsonl"))
        self.memdir = memdir
        self.onnx_embed_path = onnx_embed_path
        self.onnx_provider = onnx_provider
        self.onnx_max_len = onnx_max_len
        self.embedder = None
        if self.onnx_embed_path:
            try:
                self.embedder = ONNXEmbedder(self.onnx_embed_path, provider=self.onnx_provider, max_len=self.onnx_max_len)
            except Exception as e:
                # Fallback or raise—your call. For now, just print/log.
                print(f"[MemoryStore] ONNX init failed: {e}")


        # Load existing items (if any)
        for row in self.sem_file.read_all():
            self.semantic.append(row)
            if self.dim is None and row.get("vec"):
                self.dim = len(row["vec"])

        for row in self.epi_file.read_all():
            self.episodic.append(row)

    def add_semantic(self, text: str, meta: dict | None = None):
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
        self.semantic.append(item)
        self.sem_file.append(item)

    def add_episodic(self, text: str, meta: dict | None = None):
        item = {"text": text, "meta": meta or {}, "ts": time.time()}
        self.episodic.append(item)
        self.epi_file.append(item)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b: return 0.0
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return (dot / (na*nb)) if na and nb else 0.0

    def search_semantic(self, query: str, k: int = 5) -> list[dict]:
        try:
            if self.embedder:
                qv = self.embedder.embed(query)
            else:
                qv = self.ollama.embed(query) or _local_embed(query)
        except Exception:
            qv = _local_embed(query)

        scored = [(self._cosine(qv, it["vec"]), it) for it in self.semantic]
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:k]]

    def last_episodic(self, n: int = 3) -> list[dict]:
        return self.episodic[-n:]

def _local_embed(text: str) -> list[float]:
        # 256-dim hash vector; no dependencies
        vec = [0.0] * 256
        for b in text.encode("utf-8", errors="ignore"):
            vec[b % 256] += 1.0
        import math
        n = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v/n for v in vec]