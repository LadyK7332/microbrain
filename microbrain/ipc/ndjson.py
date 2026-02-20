# microbrain/ipc/ndjson.py
from __future__ import annotations

import json
from typing import Any

MAX_LINE_BYTES = 1024 * 1024  # 1MB safety cap

def dumps_line(obj: Any) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def loads_line(line: bytes) -> Any:
    return json.loads(line.decode("utf-8"))

def is_too_big(line: bytes) -> bool:
    return len(line) > MAX_LINE_BYTES
