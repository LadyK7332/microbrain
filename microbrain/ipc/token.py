# microbrain/ipc/token.py
from __future__ import annotations

import os
import secrets
from pathlib import Path

DEFAULT_TOKEN_PATH = Path(r"Z:\memory\ipc_token.txt")

def ensure_token_file(path: Path = DEFAULT_TOKEN_PATH) -> str:
    """
    Ensures a per-boot-ish token exists.
    If the file doesn't exist or is empty, it creates one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    return token

def read_token(path: Path = DEFAULT_TOKEN_PATH) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"IPC token file is empty: {path}")
    return token

if __name__ == "__main__":
    tok = ensure_token_file()
    print(f"IPC token ready at {DEFAULT_TOKEN_PATH}")
    print(f"token_len={len(tok)}")
