from __future__ import annotations

import json
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LlamaCppClient:
    """Talks to llama-server (OpenAI-like endpoints) with legacy fallback."""

    def __init__(self, host: str = "http://127.0.0.1:8080", model: str | None = None):
        self.host = host.rstrip("/")
        self.model = model or "default"

    def generate(self, prompt: str, **kw: Any) -> str:
        # Try OpenAI /v1/completions
        try:
            res = _post_json(
                f"{self.host}/v1/completions",
                {"model": self.model, "prompt": prompt, **kw},
            )
            ch = (res.get("choices") or [{}])[0]
            return ch.get("text", "") or ""
        except Exception:
            pass
        # Fallback to legacy /completion
        try:
            res = _post_json(f"{self.host}/completion", {"prompt": prompt, **kw})
            return res.get("content", "") or ""
        except Exception:
            return ""

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        try:
            res = _post_json(
                f"{self.host}/v1/chat/completions",
                {"model": self.model, "messages": messages, **kw},
            )
            ch = (res.get("choices") or [{}])[0]
            return ch.get("message", {}).get("content", "") or ""
        except Exception:
            # naive prompt fallback
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return self.generate(prompt, **kw)
