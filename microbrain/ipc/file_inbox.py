# microbrain/ipc/file_inbox.py
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_slug(s: str, max_len: int = 48) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_ipc_token(memdir: str | Path) -> str:
    memdir = Path(memdir)
    tok_path = memdir / "ipc_token.txt"
    tok = tok_path.read_text(encoding="utf-8").strip()
    if not tok:
        raise RuntimeError(f"ipc_token.txt is empty: {tok_path}")
    return tok


@dataclass
class IPCFileWriter:
    memdir: Path
    src: str = "lobe"
    inbox_rel: Path = Path("ipc/inbox")
    token_rel: Path = Path("ipc_token.txt")

    # Dedupe: key -> (last_payload_hash, last_emit_ms)
    _dedupe: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # If same key+hash within this window, skip writing
    dedupe_window_ms: int = 1500

    def _token(self) -> str:
        tok = (self.memdir / self.token_rel).read_text(encoding="utf-8").strip()
        if not tok:
            raise RuntimeError(f"Missing/empty token at: {self.memdir / self.token_rel}")
        return tok

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        correlation_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        dedupe_key: Optional[str] = None,
        dedupe_payload: bool = True,
    ) -> Optional[Path]:
        """
        Writes a single IPC message JSON into memdir/ipc/inbox.

        Returns the written path, or None if deduped/skipped.
        """
        ts_ms = _now_ms()
        corr = correlation_id or uuid.uuid4().hex

        # Dedupe logic (prevents disk hammering)
        if dedupe_key:
            ph = 0
            if dedupe_payload:
                # stable-ish hash by JSON canonicalization
                ph = hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            last = self._dedupe.get(dedupe_key)
            if last:
                last_hash, last_ms = last
                if (not dedupe_payload or ph == last_hash) and (ts_ms - last_ms) <= self.dedupe_window_ms:
                    return None
            self._dedupe[dedupe_key] = (ph, ts_ms)

        msg = {
            "v": 1,
            "ts_ms": ts_ms,
            "timestamp": ts_ms / 1000.0,
            "src": self.src,
            "topic": topic,
            "auth": self._token(),
            "payload": payload,
            "correlation_id": corr,
            "meta": meta or {"channel": "ipc_file"},
        }

        slug = _safe_slug(topic)
        fname = f"{ts_ms}-{corr[:8]}-{slug}.json"
        out_path = self.memdir / self.inbox_rel / fname

        _atomic_write_text(out_path, json.dumps(msg, ensure_ascii=False, indent=2))
        return out_path


@dataclass
class DrawerDoneAnnouncer:
    """
    Convenience wrapper: after a lobe writes to a drawer, announce completion once.
    """
    writer: IPCFileWriter
    done_topic: str = "drawer/done"

    def announce_done(
        self,
        *,
        drawer: str,
        data_ref: str,
        note: str = "write_complete",
        extra: Optional[Dict[str, Any]] = None,
        dedupe_window_ms: Optional[int] = None,
    ) -> Optional[Path]:
        payload: Dict[str, Any] = {
            "drawer": drawer,
            "data_ref": data_ref,
            "note": note,
        }
        if extra:
            payload.update(extra)

        # Dedupe per drawer+data_ref so you don’t spam “done” for the same artifact
        key = f"done:{drawer}:{data_ref}"
        old = self.writer.dedupe_window_ms
        if dedupe_window_ms is not None:
            self.writer.dedupe_window_ms = dedupe_window_ms

        try:
            return self.writer.publish(
                self.done_topic,
                payload,
                dedupe_key=key,
                dedupe_payload=True,
            )
        finally:
            self.writer.dedupe_window_ms = old