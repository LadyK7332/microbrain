# C:\aiproj\microbrain\microbrain\llm_backend.py

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from microbrain.llamacpp_client import LlamaCppClient
from microbrain.llm.llama_runtime import ensure_llama_server
from microbrain.mind import _pick_model

"""
Adapter between MicroBrain's existing llama.cpp / DeepSeek setup and the
async llm_generate(prompt, meta) API that LLMReasonerNeuron expects.

Flow:

    - ensure_llama_server(...)  (using the same env-style config as mind.py)
    - create a shared LlamaCppClient(host, model)
    - each llm_generate() call runs the sync client.chat() in a worker thread

Environment variables used (same family as mind.py):

    MB_LLAMA_HOST     (default: 127.0.0.1)
    MB_LLAMA_PORT     (default: 8080)
    MB_LLAMA_MODEL    (REQUIRED: path to .gguf)
    MB_LLAMA_SERVER   (optional: path to llama-server binary)
    MB_LLAMA_BACKEND  (default: "auto")
    MB_LLAMA_NGL      (default: "999")
    MB_LLAMA_ARGS     (extra CLI args for llama-server)
    MB_LLAMA_WAIT     (default: "180" seconds)
"""

_client: Optional[LlamaCppClient] = None

def _build_client() -> LlamaCppClient:
    # Mirror mind.py's runtime config as closely as possible
    host = os.getenv("MB_LLAMA_HOST", "127.0.0.1")
    port = int(os.getenv("MB_LLAMA_PORT", "8080"))

    # Use the same picker that mind.py uses
    model_path = _pick_model(os.getenv("MB_LLAMA_MODEL"))
    server_path = os.getenv("MB_LLAMA_SERVER") or None
    threads = os.cpu_count() or 4
    ngl = int(os.getenv("MB_LLAMA_NGL", "999"))
    extra = os.getenv("MB_LLAMA_ARGS", "")
    backend = os.getenv("MB_LLAMA_BACKEND", "auto")
    wait_sec = int(os.getenv("MB_LLAMA_WAIT", "180"))

    if backend.lower() == "cpu":
        os.environ["MB_LLAMA_NGL"] = "0"

    ensure_llama_server(
        model_path=model_path,
        server_path=server_path,
        host=host,
        port=port,
        threads=threads,
        ngl=ngl,
        backend=backend,
        extra_args=extra,
        wait_sec=wait_sec,
    )

    base = f"http://{host}:{port}"
    return LlamaCppClient(host=base, model=None)


def _get_client() -> LlamaCppClient:
    global _client
    if _client is not None:
        return _client
    _client = _build_client()
    return _client


def _sync_llama_call(prompt: str, meta: Dict[str, Any]) -> str:
    client = _get_client()

    messages = [
        {"role": "system", "content": "You are MicroBrain's reasoning core."},
        {"role": "user", "content": prompt},
    ]

    # Try chat() first, like Agent.step() does
    try:
        reply = client.chat(messages, options={"temperature": 0.2})
        return reply
    except Exception:
        pass

    # Fallback to plain generate()
    try:
        return client.generate(prompt, options={"temperature": 0.2})
    except Exception as exc:
        # Return an explicit error string instead of raising, so the neuron can log it
        return f"(llm error) {exc!r}"



async def llm_generate(prompt: str, meta: Dict[str, Any]) -> str:
    """
    Async API expected by LLMReasonerNeuron:

        async def llm_generate(prompt: str, meta: Dict[str, Any]) -> str

    We run the sync llama.cpp call in a worker thread. To handle the llama-server
    "just came up, still loading" window (503s / empty replies), we retry a few times
    if we get an empty or obvious error string.
    """
    loop = asyncio.get_running_loop()
    last_reply: str = ""

    for attempt in range(3):
        reply = await loop.run_in_executor(None, _sync_llama_call, prompt, meta)
        last_reply = str(reply or "").strip()

        # Accept any non-empty, non-explicit-error reply
        if last_reply and not last_reply.startswith("(llm error"):
            return last_reply

        # Backoff a bit before retrying – give the model time to finish loading
        await asyncio.sleep(1.0 + 0.5 * attempt)

    # Final fallback so the neuron has *something* to show
    return last_reply or "(llm error) reasoning backend returned empty reply after retries"

