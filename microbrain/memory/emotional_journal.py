# microbrain/memory/emotion_journal.py
from pathlib import Path
from typing import Dict, Any, List
import json, time


class EmotionJournal:
    """JSONL-backed emotion log: one event per line."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("ts", int(time.time()))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record(
        self,
        actor: str,
        text: str,
        valence: float = 0.0,
        arousal: float = 0.0,
        salience: float = 0.0,
        tags: List[str] | None = None,
        **extra: Any,
    ) -> None:
        entry: Dict[str, Any] = {
            "actor": actor,
            "text": text,
            "valence": float(valence),
            "arousal": float(arousal),
            "salience": float(salience),
            "tags": list(tags or []),
        }
        if extra:
            entry.update(extra)
        self.append(entry)

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        buf: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    buf.append(json.loads(line))
                except Exception:
                    pass
        return buf[-n:]

    def query_tags(self, *tags: str) -> List[Dict[str, Any]]:
        if not tags:
            return self.recent(50)
        want = {t.lower() for t in tags}
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ev_tags = [str(t).lower() for t in e.get("tags", [])]
                    if any(t in ev_tags for t in want):
                        out.append(e)
                except Exception:
                    pass
        return out

    def rolling_valence(self, k: int = 100) -> float:
        vals = [float(e.get("valence", 0.0)) for e in self.recent(k)]
        return sum(vals) / len(vals) if vals else 0.0
