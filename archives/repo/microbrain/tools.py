# New imports due to crash on 09/07/2025
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Union

# toolfn def
ToolFn = Union[Callable[..., Any], Callable[..., Awaitable[Any]]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, *args, **kwargs) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"tool_missing:{name}"}
        fn = self._tools[name]
        try:
            result = fn(*args, **kwargs)
            if inspect.iscoroutine(result):
                result = await result
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": f"tool_error:{e}"}

    def __init__(self):
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn):
        self._tools[name] = fn

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = self._tools.get(name)
        if not fn:
            return {"ok": False, "error": f"unknown_tool:{name}"}
        try:
            return {"ok": True, "result": fn(args)}
        except Exception as e:
            return {"ok": False, "error": f"tool_error:{e}"}


def tool_time(_: dict[str, Any]) -> dict[str, Any]:
    import datetime as dt

    return {"utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"}


def tool_search_mem(args: dict[str, Any]) -> dict[str, Any]:
    q = args.get("query", "")
    k = int(args.get("k", 5))
    hits = AGENT["mem"].search_semantic(q, k=k)
    return {"matches": [{"text": h["text"], "meta": h["meta"]} for h in hits]}
