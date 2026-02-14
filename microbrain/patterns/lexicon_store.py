from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from microbrain.memory.memory_store import JSONLStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str) -> list[str]:
    # Non-LLM tokenizer: fast, stable, debuggable.
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class TokenStats:
    token: str
    count: int = 0
    last_seen_ts: float = 0.0
    # Optional future knobs:
    salience_bias: float = 0.0  # can be nudged by /r reinforce later


class LexiconStore:
    """
    Append-only token observation log + in-memory index.

    File:
      memdir/lexicon.jsonl  (kind=token_seen)
    """

    def __init__(self, memdir: str | Path) -> None:
        self.memdir = Path(memdir)
        self.memdir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._log = JSONLStore(str(self.memdir / "lexicon.jsonl"))
        self._stats: dict[str, TokenStats] = {}

        # Load existing (cheap; if this grows huge later we can add compaction)
        try:
            for row in self._log.read_all():
                if not isinstance(row, dict):
                    continue
                if row.get("kind") != "token_seen":
                    continue
                tok = str(row.get("token", "") or "").strip().lower()
                if not tok:
                    continue
                ts = float(row.get("ts", 0.0) or 0.0)
                with self._lock:
                    st = self._stats.get(tok) or TokenStats(token=tok)
                    st.count += 1
                    st.last_seen_ts = max(st.last_seen_ts, ts)
                    self._stats[tok] = st
        except Exception:
            pass

    def observe_text(
        self,
        text: str,
        role: str = "user",
        channel: str = "default",
        ts: Optional[float] = None,
        salience: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        ts = float(ts if ts is not None else time.time())
        tokens = simple_tokenize(text)

        if not tokens:
            return []

        # Append-only log rows + in-memory counts
        with self._lock:
            for tok in tokens:
                st = self._stats.get(tok) or TokenStats(token=tok)
                st.count += 1
                st.last_seen_ts = ts
                self._stats[tok] = st

                row: dict[str, Any] = {
                    "kind": "token_seen",
                    "token": tok,
                    "role": role,
                    "channel": channel,
                    "ts": ts,
                }
                if salience is not None:
                    row["salience"] = salience
                self._log.append(row)

        return tokens

    def get(self, token: str) -> Optional[TokenStats]:
        tok = str(token or "").strip().lower()
        if not tok:
            return None
        with self._lock:
            return self._stats.get(tok)

    def top(self, n: int = 25, min_count: int = 2) -> list[TokenStats]:
        with self._lock:
            items = [st for st in self._stats.values() if st.count >= min_count]
        items.sort(key=lambda s: (s.count, s.last_seen_ts), reverse=True)
        return items[: max(0, int(n))]
