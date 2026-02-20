from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

APP = FastAPI(title="MicroBrain WebUI Adapter", version="0.1")

# Set this to anything; you’ll enter it in Open WebUI as the API Key later.
MB_WEBUI_API_KEY = os.environ.get("MB_WEBUI_API_KEY", "microbrain-dev-key")

# ----- OpenAI-compatible models -----

class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "microbrain"

@APP.get("/v1/models")
def list_models() -> Dict[str, Any]:
    # Open WebUI expects at least one model
    return {"object": "list", "data": [ModelItem(id="microbrain-core").model_dump()]}

# ----- OpenAI-compatible chat completions -----

class ChatMessage(BaseModel):
    role: str
    content: Any  # WebUI may send strings or structured content

class ChatCompletionReq(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

def _auth_ok(auth_header: Optional[str]) -> bool:
    if not auth_header:
        return False
    # OpenAI-style: "Bearer <key>"
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == MB_WEBUI_API_KEY
    return False

def _extract_user_text(messages: List[ChatMessage]) -> str:
    # Grab last user message content as a best-effort string
    for m in reversed(messages):
        if m.role == "user":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""

@APP.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionReq, authorization: Optional[str] = Header(default=None)):
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_text = _extract_user_text(req.messages)

    # TODO (next step): forward to MicroBrain event bus (input/text) and wait for reply.
    # For now: stub reply so we can verify WebUI wiring + streaming works.
    reply = f"[adapter ok] You said: {user_text}"

    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"

    if req.stream:
        async def gen():
            # 1) role chunk
            first = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {first}\n\n"

            # 2) content chunk(s)
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": reply}, "finish_reason": None}],
            }
            yield f"data: {chunk}\n\n"

            # 3) done
            done = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {done}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    # non-streaming
    return JSONResponse(
        {
            "id": resp_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }
    )

# ----- “cockpit” telemetry endpoints (WebUI won’t call these automatically, but we will use them) -----

@APP.get("/mb/status")
def mb_status(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # TODO: return real MicroBrain telemetry
    return {
        "ts": int(time.time() * 1000),
        "state": "stub",
        "notes": "Next step: wire to MicroBrain telemetry bus",
    }
