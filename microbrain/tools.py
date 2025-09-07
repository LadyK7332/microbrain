class ToolRegistry:
    def __init__(self): self._tools: Dict[str, ToolFn] = {}
    def register(self, name: str, fn: ToolFn): self._tools[name] = fn
    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(name); 
        if not fn: return {"ok": False, "error": f"unknown_tool:{name}"}
        try: return {"ok": True, "result": fn(args)}
        except Exception as e: return {"ok": False, "error": f"tool_error:{e}"}

def tool_time(_: Dict[str, Any]) -> Dict[str, Any]:
    import datetime as dt
    return {"utc": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"}

def tool_search_mem(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args.get("query", "")
    k = int(args.get("k", 5))
    hits = AGENT["mem"].search_semantic(q, k=k)
    return {"matches": [{"text": h["text"], "meta": h["meta"]} for h in hits]}