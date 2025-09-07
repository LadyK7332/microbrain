from typing import Any, Dict, Optional, List
import json
import urllib.request
import urllib.error

def _http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))

class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: str = "llama3",
                use_chat: bool = True, embed_model: Optional[str] = None):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = model
        self.use_chat = use_chat
        self.embed_model = embed_model or model

    # Chat-style (multi-turn) preferred
    def chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        url = f"{self.host}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        if options: payload["options"] = options
        res = _http_json(url, payload)
        return res.get("message", {}).get("content", "")

    # Simple generate fallback
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}  # <-- fix model_model typo
        if options: payload["options"] = options
        res = _http_json(url, payload)
        return res.get("response", "")

    # Embeddings
    def embed(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": (self.embed_model or self.model), "prompt": text}  # <-- honor embed_model
        try:
            res = _http_json(url, payload)
            return res.get("embedding", [])
        except Exception:
            return []