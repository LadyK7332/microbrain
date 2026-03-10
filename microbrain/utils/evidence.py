from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

from microbrain.utils.memdir import resolve_memdir_ctx


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def chain_hash(prev_chain: str, file_hash: str, ts: float, kind: str, rel_path: str) -> str:
    seed = f"{prev_chain}|{file_hash}|{ts:.6f}|{kind}|{rel_path}".encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).hexdigest()


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def ensure_evidence_session(ctx, trigger: dict | None = None) -> tuple[str, Path]:
    sess_id = str(await ctx.get_kv("er:session_id", "") or "")
    sess_dir = str(await ctx.get_kv("er:session_dir", "") or "")
    if sess_id and sess_dir:
        return sess_id, Path(sess_dir)

    memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
    base = memdir / "evidence"
    base.mkdir(parents=True, exist_ok=True)

    sess_id = time.strftime("session-%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    sess_path = base / sess_id
    (sess_path / "audio").mkdir(parents=True, exist_ok=True)
    (sess_path / "video").mkdir(parents=True, exist_ok=True)

    trigger = dict(trigger or {})
    session_row = {
        "session_id": sess_id,
        "created_ts": time.time(),
        "trigger": trigger,
    }
    with open(sess_path / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_row, f, indent=2, ensure_ascii=False)

    await ctx.set_kv("er:session_id", sess_id)
    await ctx.set_kv("er:session_dir", str(sess_path))
    await ctx.set_kv("er:session_chain", "")
    return sess_id, sess_path


async def append_evidence_index(ctx, session_dir: Path, kind: str, rel_path: str, ts: float, sha256: str, extra: dict | None = None) -> str:
    prev_chain = str(await ctx.get_kv("er:session_chain", "") or "")
    ch = chain_hash(prev_chain, sha256, ts, kind, rel_path)
    row = {
        "ts": float(ts),
        "kind": kind,
        "path": rel_path.replace("\\", "/"),
        "sha256": sha256,
        "chain": ch,
    }
    if extra:
        row.update(extra)
    _append_jsonl(session_dir / "index.jsonl", row)
    await ctx.set_kv("er:session_chain", ch)
    return ch


def write_wav_mono_i16(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
