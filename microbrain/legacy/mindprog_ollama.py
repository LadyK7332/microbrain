#!/usr/bin/env python3
"""
mindprog_ollama.py — HRM/LLM scaffold wired to Ollama

- Uses Ollama's /api/chat (preferred) or /api/generate for text.
- Uses Ollama's /api/embeddings for semantic memory.
- Agent loop: perceive -> plan -> act -> reflect
- Tooling stub included; easy to add more tools.

Quick start:
  python mindprog_ollama.py --model llama3 --demo
  python mindprog_ollama.py --model mistral --ask "Plan my afternoon"

Requires: Ollama running locally (default http://127.0.0.1:11434).
"""

from __future__ import annotations
import argparse, json, os, sys, time, threading, pyttsx3, queue
from typing import Any, Dict, List, Optional, Callable
import urllib.request
import urllib.error
import queue
import pyttsx3
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer
import onnxruntime as ort
# replace this:
# from tokenizers import Tokenizer

# with this:
try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None
# =============== Ollama client ===============

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
            return []  # let MemoryStore fall back to _local_embed

# =============== Memory (semantic + episodic) ===============
class JSONLStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):  # create empty file
                pass
        self._lock = threading.Lock()

# inside JSONLStore.__init__

    def append(self, obj: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        with self._lock:
            items = []
            with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
            return items

def _local_embed(text: str) -> list[float]:
        # 256-dim hash vector; no dependencies
        vec = [0.0] * 256
        for b in text.encode("utf-8", errors="ignore"):
            vec[b % 256] += 1.0
        import math
        n = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v/n for v in vec]

class ONNXEmbedder:
    """
    Simple ONNX embedding runner.
    Expects a model that takes 'input_ids' and 'attention_mask' and outputs either:
      - 'last_hidden_state' (B, T, D) or
      - 'pooler_output' (B, D)
    We apply mean pooling over token dimension if last_hidden_state is provided.
    """
    def __init__(self, onnx_path: str, tokenizer_path_or_json: str | None,
                 provider: str = "DmlExecutionProvider", max_len: int = 256):
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=[provider, "CPUExecutionProvider"]
        )
        # Load tokenizer: if you pass a tokenizer JSON path, Tokenizer.from_file works.
        # If None, we fall back to a whitespace tokenizer (works but poorer quality).
        self.max_len = max_len
        self.tokenizer = None
        if Tokenizer is not None and tokenizer_path_or_json and os.path.exists(tokenizer_path_or_json):
            self.tokenizer = Tokenizer.from_file(tokenizer_path_or_json)
        # else: leave self.tokenizer = None (will use whitespace fallback)

        outs = [o.name for o in self.sess.get_outputs()]
        self.out_last = "last_hidden_state" if "last_hidden_state" in outs else None
        self.out_pool = "pooler_output" if "pooler_output" in outs else None

    def _ws_tokenize(self, text: str):
        # whitespace tokenization into a tiny vocab: map tokens to pseudo IDs
        toks = text.strip().split()
        # cap length
        toks = toks[: self.max_len]
        # map to small integer IDs based on hash (no external vocab)
        ids = [(hash(t) % 30522) for t in toks]  # 30k-ish
        attn = [1] * len(ids)
        # pad to max_len
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            attn += [0] * pad
        return ids, attn

    def _hf_tokenize(self, text: str):
        # fast tokenizer JSON path case
        enc = self.tokenizer.encode(text)
        ids = enc.ids[: self.max_len]
        attn = [1] * len(ids)
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            attn += [0] * pad
        return ids, attn

    def embed(self, text: str) -> list[float]:
        # Build inputs
        if self.tokenizer:
            ids, attn = self._hf_tokenize(text)
        else:
            ids, attn = self._ws_tokenize(text)

        import numpy as np
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.array([attn], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.sess.run(None, feeds)

        # Select output
        if self.out_pool:
            vec = outputs[[o.name for o in self.sess.get_outputs()].index(self.out_pool)][0]  # (D,)
        elif self.out_last:
            last = outputs[[o.name for o in self.sess.get_outputs()].index(self.out_last)][0]  # (T,D)
            # mean pool over tokens with attention mask
            mask = attention_mask[0][: last.shape[0]].astype(float)
            mask = mask[:, None]
            denom = max(mask.sum(), 1.0)
            vec = (last * mask).sum(axis=0) / denom  # (D,)
        else:
            # fallback: flatten first output and truncate
            arr = outputs[0].reshape(-1)
            vec = arr[:384] if arr.size >= 384 else arr

        # L2 normalize
        n = float((vec**2).sum()) ** 0.5 or 1.0
        return (vec / n).tolist()

class MemoryStore:
    """
    Persistent memory with optional Ollama embeddings.
    Writes to two JSONL files under base_dir:
      - semantic.jsonl (text, vec, meta, ts)
      - episodic.jsonl (text, meta, ts)
    """
    def __init__(self, ollama, base_dir: str, embedder=None):
        self.ollama = ollama
        self.base_dir = base_dir
        self.embedder = embedder
        self.semantic: list[dict] = []  # {text, vec, meta, ts}
        self.episodic: list[dict] = []  # {text, meta, ts}
        self.dim: int | None = None

        self.sem_file = JSONLStore(os.path.join(base_dir, "semantic.jsonl"))
        self.epi_file = JSONLStore(os.path.join(base_dir, "episodic.jsonl"))

        # Load existing items (if any)
        for row in self.sem_file.read_all():
            self.semantic.append(row)
            if self.dim is None and row.get("vec"):
                self.dim = len(row["vec"])

        for row in self.epi_file.read_all():
            self.episodic.append(row)

    def add_semantic(self, text: str, meta: dict | None = None):
        # Try Ollama embeddings first; if unavailable, fall back to local
        try:
            if self.embedder:
                vec = self.embedder.embed(text)
            else:
                vec = self.ollama.embed(text) or _local_embed(text)
        except Exception:
            vec = _local_embed(text)

        if self.dim is None and vec:
            self.dim = len(vec)

        item = {"text": text, "vec": vec, "meta": meta or {}, "ts": time.time()}
        self.semantic.append(item)
        self.sem_file.append(item)

    def add_episodic(self, text: str, meta: dict | None = None):
        item = {"text": text, "meta": meta or {}, "ts": time.time()}
        self.episodic.append(item)
        self.epi_file.append(item)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b: return 0.0
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return (dot / (na*nb)) if na and nb else 0.0

    def search_semantic(self, query: str, k: int = 5) -> list[dict]:
        try:
            if self.embedder:
                qv = self.embedder.embed(query)
            else:
                qv = self.ollama.embed(query) or _local_embed(query)
        except Exception:
            qv = _local_embed(query)

        scored = [(self._cosine(qv, it["vec"]), it) for it in self.semantic]
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:k]]

    def last_episodic(self, n: int = 3) -> list[dict]:
        return self.episodic[-n:]

# =============== Tools ===============

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]

class ToolRegistry:
    def __init__(self): self._tools: Dict[str, ToolFn] = {}
    def register(self, name: str, fn: ToolFn): self._tools[name] = fn
    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(name); 
        if not fn: return {"ok": False, "error": f"unknown_tool:{name}"}
        try: return {"ok": True, "result": fn(args)}
        except Exception as e: return {"ok": False, "error": f"tool_error:{e}"}

# Example tools
def tool_time(_: Dict[str, Any]) -> Dict[str, Any]:
    import datetime as dt
    return {"utc": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"}

def tool_search_mem(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args.get("query", "")
    k = int(args.get("k", 5))
    hits = AGENT["mem"].search_semantic(q, k=k)
    return {"matches": [{"text": h["text"], "meta": h["meta"]} for h in hits]}
# =============== MicroBrain (multi-neuron) ===============
from dataclasses import dataclass, field
from collections import deque
from typing import Deque

@dataclass
class MBMessage:
    role: str  # 'user' | 'neuron' | 'system'
    content: str
    meta: dict = field(default_factory=dict)

class Neuron:
    name: str = "neuron"
    def __init__(self, ollama, mem): self.ollama, self.mem = ollama, mem
    def step(self, inbox: list[MBMessage], goal: str) -> MBMessage:
        return MBMessage(role="neuron", content="(noop)", meta={"neuron": self.name})

class PlannerNeuron(Neuron):
    name = "planner"
    def step(self, inbox, goal):
        context = "\n".join(m.content for m in inbox[-5:])
        prompt = (
            "You are a pragmatic planner. Based on GOAL and CONTEXT, output 3 numbered next actions.\n"
            f"GOAL:\n{goal}\n\nCONTEXT:\n{context}\n\n"
            "Rules: Be concise; 1-2 lines per action; if goal is achieved, say: DONE."
        )
        out = self.ollama.chat(
            [{"role":"system","content":"You plan in crisp steps."},
             {"role":"user","content":prompt}],
            options={"temperature":0.2}
        )
        self.mem.add_semantic(f"[PLAN]\n{out}", {"neuron":"planner"})
        return MBMessage(role="neuron", content=out, meta={"neuron": self.name})

class ReasonerNeuron(Neuron):
    name = "reasoner"
    def step(self, inbox, goal):
        context = "\n".join(m.content for m in inbox[-5:])
        prompt = (
            "You are a careful reasoner. Improve or execute the current plan step.\n"
            f"GOAL:\n{goal}\n\nCONTEXT:\n{context}\n\n"
            "Respond with a short rationale and a concrete result. If the task is complete, end with: DONE."
        )
        out = self.ollama.chat(
            [{"role":"system","content":"Be precise, brief, actionable."},
             {"role":"user","content":prompt}],
            options={"temperature":0.2}
        )
        self.mem.add_semantic(f"[REASON]\n{out}", {"neuron":"reasoner"})
        return MBMessage(role="neuron", content=out, meta={"neuron": self.name})

class MemoryNeuron(Neuron):
    name = "memory"
    def step(self, inbox, goal):
        # pull a query from goal or last msg; then stuff top-k recalls into memory/output
        query = (inbox[-1].content if inbox else goal)[:500]
        hits = self.mem.search_semantic(query, k=5)
        summary = "\n".join(f"- {h['text'][:280]}" for h in hits)
        out = f"Top recalls for: {query[:80]}\n{summary or '(none)'}"
        # also tuck a compact memo for cross-neuron use
        self.mem.add_episodic(f"[RECALL]\n{out}", {"neuron":"memory"})
        return MBMessage(role="neuron", content=out, meta={"neuron": self.name})

# Optional: a code-focused neuron (kept minimal)
class CoderNeuron(Neuron):
    name = "coder"
    def step(self, inbox, goal):
        context = "\n".join(m.content for m in inbox[-5:])
        prompt = (
            "You are a coding assistant. Based on GOAL and CONTEXT, produce a small code snippet or patch.\n"
            f"GOAL:\n{goal}\n\nCONTEXT:\n{context}\n\n"
            "If no code is needed, say 'SKIP'."
        )
        out = self.ollama.chat(
            [{"role":"system","content":"You write correct, minimal code."},
             {"role":"user","content":prompt}],
            options={"temperature":0.2}
        )
        self.mem.add_semantic(f"[CODE]\n{out}", {"neuron":"coder"})
        return MBMessage(role="neuron", content=out, meta={"neuron": self.name})

class MicroBrain:
    def __init__(self, ollama, mem, neuron_names: list[str]):
        self.ollama, self.mem = ollama, mem
        registry = {
            "planner": PlannerNeuron, "reasoner": ReasonerNeuron,
            "memory": MemoryNeuron, "coder": CoderNeuron
        }
        self.neurons = [registry[n](ollama, mem) for n in neuron_names if n in registry]
        self.bus: Deque[MBMessage] = deque(maxlen=50)

    def run(self, goal: str, steps: int = 8) -> list[MBMessage]:
        self.bus.append(MBMessage(role="system", content=f"GOAL: {goal}"))
        for i in range(steps):
            for neuron in self.neurons:
                msg = neuron.step(list(self.bus), goal)
                self.bus.append(msg)
                # stop condition
                if "DONE" in msg.content.upper():
                    return list(self.bus)
        return list(self.bus)
# =============== Voice I/O ===============
class TTS:
    def __init__(self, rate: int = 175):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        # Optional: choose a voice; comment out if default is fine
        # for v in self.engine.getProperty("voices"):
        #     if "Zira" in v.name or "Aria" in v.name:  # example pick
        #         self.engine.setProperty("voice", v.id); break

    def say(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()

class STT:
    def __init__(self, vosk_model_path: str, device: int | None = None, samplerate: int = 0):
        if not vosk_model_path or not os.path.isdir(vosk_model_path):
            raise RuntimeError(f"Invalid --vosk-model-path: {vosk_model_path}")

        # Resolve device & sample rate
        if device is not None:
            info = sd.query_devices(device)
            if not samplerate or samplerate <= 0:
                samplerate = int(info.get("default_samplerate") or 44100)
        else:
            info = sd.query_devices(None, "input")
            if not samplerate or samplerate <= 0:
                samplerate = int(info.get("default_samplerate") or 44100)

        self.device = device
        self.rate = int(samplerate)

        # Debug prints
        print(f"[voice] STT using device={self.device} rate={self.rate}")
        print(f"[voice] Device info: {info}")

        self.model = VoskModel(vosk_model_path)
        self.rec = KaldiRecognizer(self.model, self.rate)
        self.rec.SetWords(True)
        self._q: "queue.Queue[bytes]" = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:  # non-fatal stream status
            pass
        self._q.put(bytes(indata))

    def listen_once(self, prompt_tts: "TTS | None" = None, prompt_text: str = "Listening.") -> str:
        if prompt_tts:
            prompt_tts.say(prompt_text)

        kwargs = dict(samplerate=self.rate, blocksize=4096, dtype="int16", channels=1,
                    callback=self._callback)
        if self.device is not None:
            kwargs["device"] = self.device

        import time as _t, json as _json
        t0 = _t.time()
        got_bytes = 0
        with sd.RawInputStream(**kwargs):
            partial = ""
            while True:
                data = self._q.get()
                got_bytes += len(data)
                if self.rec.AcceptWaveform(data):
                    res = _json.loads(self.rec.Result())
                    text = (res.get("text") or "").strip()
                    if text:
                        print(f"[voice] final='{text}' bytes={got_bytes}")
                        return text
                else:
                    p = _json.loads(self.rec.PartialResult()).get("partial", "")
                    if p:
                        partial = p
                if _t.time() - t0 > 5:
                    print(f"[voice] timeout 5s; partial='{partial}' bytes={got_bytes}")
                    return partial.strip()

def voice_chat_loop(agent, stt: STT, tts: TTS):
    tts.say("Voice mode ready. Say 'exit' to quit.")
    while True:
        user_text = stt.listen_once(tts, "Listening.")
        if not user_text:
            tts.say("I didn't catch that. Please repeat.")
            continue
        if user_text.lower().strip() in ("exit", "quit", "stop"):
            tts.say("Goodbye.")
            break

        # Log to episodic memory and fetch context like your text REPL
        agent.mem.add_episodic(f"USER SAID: {user_text}", {"mode": "voice"})

        context_lines = []
        for hit in agent.mem.search_semantic(user_text, k=5):
            context_lines.append(f"- {hit['text'][:200]}")
        recent = agent.mem.last_episodic(3)
        if recent:
            context_lines.append("\nRecent:")
            for e in recent:
                context_lines.append(f"- {e['text'][:200]}")

        context_text = "\n".join(context_lines)
        messages = [
            {"role": "system", "content": agent.system},
            {"role": "user", "content":
                f"{context_text}\n\nUser: {user_text}\nPlan step-by-step, then answer briefly."}
        ]

        # Generate reply
        reply = agent.ollama.chat(messages, options={"temperature": 0.2})
        agent.mem.add_semantic(f"ASSISTANT SAID: {reply[:500]}", {"mode": "voice"})
        tts.say(reply[:500])  # speak first ~500 chars to avoid very long monologues

# =============== Agent ===============

DEFAULT_SYSTEM = (
    "You are a careful, helpful planner. Use memory judiciously, call tools when needed, "
    "think in small steps, and provide concise next actions."
)

class Agent:
    def __init__(self, ollama: OllamaClient, mem: MemoryStore, tools: ToolRegistry, system: str = DEFAULT_SYSTEM):
        self.ollama = ollama
        self.mem = mem
        self.tools = tools
        self.system = system

    def step(self, user_input: str) -> str:
        # Log perception
        self.mem.add_episodic(f"USER: {user_input}", {"role":"user"})
        # Retrieve context
        sem = self.mem.search_semantic(user_input, k=5)
        epis = self.mem.last_episodic(3)

        context_lines = []
        if sem: 
            context_lines.append("Relevant semantic memory:")
            for h in sem: context_lines.append(f"- {h['text']}")
        if epis:
            context_lines.append("\nRecent episodic memory:")
            for e in epis: context_lines.append(f"- {e['text']}")

        context_text = "\n".join(context_lines)
        prompt = (
            f"{self.system}\n\n"
            f"{context_text}\n\n"
            f"User: {user_input}\n"
            f"Plan step-by-step, call tools if useful, then answer."
        )

        # Prefer chat endpoint to keep system/user roles
        messages = [{"role":"system","content":self.system},
                    {"role":"user","content":user_input}]
        if context_lines: messages.insert(1, {"role":"system","content":"\n".join(context_lines)})

        try:
            reply = self.ollama.chat(messages, options={"temperature":0.2})
        except Exception:
            reply = self.ollama.generate(prompt, options={"temperature":0.2})

        # Store semantic reflection of reply for future retrieval
        self.mem.add_semantic(reply, {"role":"assistant"})
        self.mem.add_episodic(f"ASSISTANT: {reply}", {"role":"assistant"})
        return reply

# =============== Wiring & CLI ===============

AGENT: Dict[str, Any] = {}

def build_agent(model: str, host: Optional[str], embed_model: Optional[str],
                memdir: str, onnx_embed_path: Optional[str],
                onnx_provider: str, onnx_max_len: int) -> Agent:
    embedder = None
    if onnx_embed_path:
        # auto-detect tokenizer.json in the same folder as model.onnx
        model_dir = os.path.dirname(onnx_embed_path)
        tokenizer_json = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_json):
            tokenizer_json = None  # fallback to whitespace tokenizer
        embedder = ONNXEmbedder(onnx_embed_path, tokenizer_json,
                                provider=onnx_provider, max_len=onnx_max_len)

    oll = OllamaClient(host=host, model=model, use_chat=True, embed_model=embed_model)
    mem = MemoryStore(ollama=oll, base_dir=memdir, embedder=embedder)  # <--- pass embedder
    tools = ToolRegistry()  # <-- instantiate
    tools.register("time", tool_time)
    tools.register("search_mem", tool_search_mem)
    agent = Agent(ollama=oll, mem=mem, tools=tools)
    AGENT["mem"] = mem
    return agent

# =============== Voice I/O ===============
class TTS:
    """
    Offline TTS using Windows SAPI via pyttsx3.
    You can pass a name substring (e.g., 'Zira', 'Aria', 'Female') to pick a mature feminine voice.
    """
    def __init__(self, rate: int = 175, volume: float = 1.0, preferred: str = ""):
        self.engine = pyttsx3.init()
        # rate / volume
        try:
            self.engine.setProperty("rate", rate)
        except Exception:
            pass
        try:
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))
        except Exception:
            pass

        # voice selection by substring (case-insensitive)
        self.chosen_voice = None
        try:
            voices = self.engine.getProperty("voices") or []
            pref = (preferred or "").lower()
            if pref:
                for v in voices:
                    name = (getattr(v, "name", "") or "").lower()
                    # heuristic: prefer English female if present
                    if pref in name:
                        self.engine.setProperty("voice", v.id)
                        self.chosen_voice = v
                        break
            # If no match and we want a feminine voice, try common female voices by name
            if not self.chosen_voice and pref:
                for tag in ("female", "zira", "aria", "susan", "eva"):
                    for v in voices:
                        if tag in (v.name or "").lower():
                            self.engine.setProperty("voice", v.id)
                            self.chosen_voice = v
                            break
                    if self.chosen_voice:
                        break
        except Exception:
            pass

    def list_voices(self) -> list[str]:
        try:
            return [f"{i}: {v.name} ({v.id})" for i, v in enumerate(self.engine.getProperty("voices") or [])]
        except Exception:
            return []

    def say(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()


class STT:
    """
    Vosk STT with sounddevice. Uses device default samplerate unless you pass one.
    """
    def __init__(self, vosk_model_path: str, device: int | None = None, samplerate: int = 0):
        if not vosk_model_path or not os.path.isdir(vosk_model_path):
            raise RuntimeError("Invalid --vosk-model-path (folder not found).")

        # Resolve device + samplerate
        self.device = device
        if device is not None:
            info = sd.query_devices(device)
            if not samplerate or samplerate <= 0:
                samplerate = int(info["default_samplerate"])
        else:
            # fallback to default input device’s rate
            info = sd.query_devices(None, "input")
            if not samplerate or samplerate <= 0:
                samplerate = int(info["default_samplerate"])

        self.rate = samplerate
        self.model = VoskModel(vosk_model_path)
        self.rec = KaldiRecognizer(self.model, self.rate)
        self.rec.SetWords(True)
        self._q: "queue.Queue[bytes]" = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            # non-fatal stream status (overruns etc.)
            pass
        self._q.put(bytes(indata))

    def listen_once(self, prompt_tts: "TTS | None" = None, prompt_text: str = "Listening.") -> str:
        if prompt_tts:
            prompt_tts.say(prompt_text)

        # Use explicit device if provided
        kwargs = dict(samplerate=self.rate, blocksize=4096, dtype="int16", channels=1, callback=self._callback)
        if self.device is not None:
            kwargs["device"] = self.device

        # Collect ~5 seconds or until a final result
        import time as _t
        t0 = _t.time()
        with sd.RawInputStream(**kwargs):
            partial = ""
            while True:
                data = self._q.get()
                if self.rec.AcceptWaveform(data):
                    import json as _json
                    res = _json.loads(self.rec.Result())
                    text = (res.get("text") or "").strip()
                    if text:
                        return text
                else:
                    import json as _json
                    p = _json.loads(self.rec.PartialResult()).get("partial", "")
                    if p:
                        partial = p
                if _t.time() - t0 > 5:
                    return partial.strip()


def voice_chat_loop(agent, stt: STT, tts: TTS):
    # announce + show the chosen voice
    chosen = getattr(tts, "chosen_voice", None)
    if chosen:
        agent.mem.add_episodic(f"[VOICE] Using TTS voice: {chosen.name}", {"mode": "voice"})
    tts.say("Voice mode ready. Say 'exit' to quit.")

    while True:
        user_text = stt.listen_once(tts, "Listening.")
        if not user_text:
            tts.say("I didn't catch that. Please repeat.")
            continue
        if user_text.lower().strip() in ("exit", "quit", "stop"):
            tts.say("Goodbye.")
            break

        # Save user input
        agent.mem.add_episodic(f"USER SAID: {user_text}", {"mode": "voice"})

        # Build short context (semantic recalls + last few episodic)
        context_lines = []
        for hit in agent.mem.search_semantic(user_text, k=5):
            context_lines.append(f"- {hit['text'][:200]}")
        recent = agent.mem.last_episodic(3)
        if recent:
            context_lines.append("\nRecent:")
            for e in recent:
                context_lines.append(f"- {e['text'][:200]}")
        context_text = "\n".join(context_lines)

        # Ask the LLM
        messages = [
            {"role": "system", "content": agent.system},
            {"role": "user", "content": f"{context_text}\n\nUser: {user_text}\nPlan step-by-step, then answer briefly."}
        ]
        reply = agent.ollama.chat(messages, options={"temperature": 0.2})
        agent.mem.add_semantic(f"ASSISTANT SAID: {reply[:500]}", {"mode": "voice"})

        # Speak the first ~500 chars to avoid very long monologues
        tts.say(reply[:500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", type=str, default=None,
                help="Ollama model to use for embeddings (e.g., nomic-embed-text)")
    ap.add_argument("--model", type=str, default="llama3", help="Ollama model name (e.g., llama3, mistral, qwen2.5, codellama)")
    ap.add_argument("--memdir", type=str, default="memory",
                help="Directory for persistent memory JSONL files")
    ap.add_argument("--host", type=str, default=None, help="Ollama host URL (default http://127.0.0.1:11434 or $OLLAMA_HOST)")
    ap.add_argument("--ask", type=str, help="Ask once and exit")
    ap.add_argument("--demo", action="store_true", help="Interactive REPL")
    ap.add_argument("--onnx-embed-path", type=str, default=None,
                help="Path to an ONNX embedding model (.onnx). If set, RX 6600 via DirectML will be used for embeddings.")
    ap.add_argument("--onnx-provider", type=str, default="DmlExecutionProvider",
                help="ONNX Runtime EP (default: DmlExecutionProvider). Fallbacks: CPUExecutionProvider")
    ap.add_argument("--onnx-max-len", type=int, default=256,
                help="Max tokens for ONNX tokenizer input (default 256).")
    ap.add_argument("--microbrain", action="store_true",
                help="Run MicroBrain multi-neuron scheduler instead of single-agent REPL.")
    ap.add_argument("--goal", type=str, default="Plan useful next steps for this workspace.",
                    help="High-level goal for the MicroBrain run.")
    ap.add_argument("--neurons", type=str, default="planner,reasoner,memory",
                    help="Comma-separated neuron set (e.g., planner,reasoner,memory,coder).")
    ap.add_argument("--steps", type=int, default=8,
                    help="Max scheduler steps before stopping.")
    ap.add_argument("--k", type=int, default=5,
                    help="Top-k semantic recalls per memory query.")
    ap.add_argument("--voice", action="store_true",
                help="Enable voice I/O mode (mic in, TTS out).")
    ap.add_argument("--vosk-model-path", type=str, default=None,
                    help="Path to an unzipped Vosk STT model directory.")
    ap.add_argument("--mic-device", type=int, default=None,
                    help="Input device index (see: python -m sounddevice).")
    ap.add_argument("--sample-rate", type=int, default=0,
                    help="Mic sample rate. 0 = use device default.")
    ap.add_argument("--tts-rate", type=int, default=175,
                    help="Speech rate (words/min).")
    ap.add_argument("--tts-volume", type=float, default=1.0,
                    help="TTS volume 0.0–1.0.")
    ap.add_argument("--tts-voice", type=str, default="",
                    help="Preferred voice name substring (e.g. 'Zira', 'Aria', 'Female').")


    args = ap.parse_args()
    agent = build_agent(args.model, args.host, args.embed_model,
                    args.memdir, args.onnx_embed_path,
                    args.onnx_provider, args.onnx_max_len)
    if args.microbrain:
        names = [n.strip().lower() for n in args.neurons.split(",") if n.strip()]
        mb = MicroBrain(agent.ollama, agent.mem, names)
        trace = mb.run(goal=args.goal, steps=args.steps)
        print("\n=== MicroBrain trace ===")
        for m in trace:
            tag = m.meta.get("neuron", m.role)
            print(f"[{tag}] {m.content[:400]}")
        return
# ----- after building `agent` and printing the "Loaded N/M" banner -----
# ----- Voice block (single source of truth) -----
    if args.voice:
        if not args.vosk_model_path:
            print("[error] --vosk-model-path is required for --voice")
            return

        try:
            # Use your known-good combo by default unless overridden
            mic_idx = args.mic_device if args.mic_device is not None else 1       # MME index 1
            samp_rate = args.sample_rate if args.sample_rate and args.sample_rate > 0 else 44100
            print(f"[voice] init with device={mic_idx} samplerate={samp_rate}")

            stt = STT(args.vosk_model_path, device=mic_idx, samplerate=samp_rate)
            tts = TTS(rate=args.tts_rate, volume=args.tts_volume, preferred=args.tts_voice)

            if args.tts_voice and not getattr(tts, "chosen_voice", None):
                print(f"[warn] Preferred voice '{args.tts_voice}' not found. Available voices:")
                for line in tts.list_voices():
                    print("  ", line)

        except Exception as e:
            print(f"[error] Voice setup failed: {e}")
            import traceback; traceback.print_exc()
            return

        try:
            voice_chat_loop(agent, stt, tts)
        except Exception as e:
            print(f"[error] Voice loop crashed: {e}")
            import traceback; traceback.print_exc()
        return
# ----- end voice block -----

    # Debug banner
    print(f"Loaded {len(agent.mem.semantic)} semantic items, "
        f"{len(agent.mem.episodic)} episodic items from {args.memdir}")

    # Prime some semantic memory so search_mem has something to find
    agent.mem.add_semantic("You can use the tool 'time' to get UTC time.")
    agent.mem.add_semantic("Search memory using the 'search_mem' tool with a query and k.")

    if args.ask:
        print(agent.step(args.ask))
        return

    if args.demo:
        print(f"Mind demo (Ollama model: {args.model}). Type 'exit' to quit.")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye"); break
            if text.lower() in {"exit","quit"}: print("bye"); break
            print(agent.step(text))
        return

    ap.print_help()

if __name__ == "__main__":
    main()