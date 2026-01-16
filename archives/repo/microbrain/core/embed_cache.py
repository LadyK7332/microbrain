from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class EmbedCache:
    def __init__(self, path: str = "./cache/embeddings.jsonl") -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: dict[str, list[float]] = {}
        if self.path.exists():
            for line in self.path.open("r", encoding="utf-8"):
                try:
                    item = json.loads(line)
                    self.data[item["key"]] = item["vec"]
                except Exception:
                    pass

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self.data.get(self._hash(text))

    def add(self, text: str, vec: list[float]):
        key = self._hash(text)
        with self.lock:
            self.data[key] = vec
            with self.path.open("a", encoding="utf-8") as f:
                json.dump({"key": key, "vec": vec}, f)
                f.write("\n")
