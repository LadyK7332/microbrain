# Imports
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from microbrain.memory.emotion_journal import EmotionJournal

try:
    from microbrain.llamacpp_client import LlamaCppClient
except Exception:
    LlamaCppClient = None  # optional fallback
from microbrain.tools import ToolRegistry

# pull the default system prompt from config; fall back if not present
try:
    from microbrain.config import DEFAULT_SYSTEM
except Exception:
    DEFAULT_SYSTEM = (
        "You are a helpful, concise agent. Think step-by-step and prefer short, actionable outputs."
    )


# Agent Class Creation
class Agent:
    def __init__(
        self,
        llm: Any | None = None,
        ollama: Any | None = None,
        client: Any | None = None,
        memory=None,
        tools=None,
        logger=None,
    ):
        # normalize aliases: prefer llm, then client, then ollama
        chosen = llm or client or ollama
        if chosen is None:
            raise ValueError("Agent requires an LLM client (llm/client/ollama)")

        # keep multiple names for back-compat across the codebase
        self.llm = chosen
        self.client = chosen
        self.ollama = chosen

        self.memory = memory
        self.tools = tools or ToolRegistry()
        self.logger = logger
        self.system = DEFAULT_SYSTEM
        # pDNA + emotion journal plumbing
        self._pdna_path = self._resolve_pdna_path()
        self.pdna = self._load_pdna(self._pdna_path)

        ej_path = os.getenv("MB_EMO_JOURNAL_PATH")
        if not ej_path:
            base = (
                getattr(self.memory, "memdir", None)
                or getattr(self.memory, "base_dir", None)
                or "."
            )
            ej_path = str(Path(base) / "emotion_journal.jsonl")
        self.emotions = EmotionJournal(ej_path)

        # simple rules switch you can toggle later
        self.rules = {"autosalience": True}

    # --- pDNA helpers -------------------------------------------------
    def _resolve_pdna_path(self) -> str:
        base = getattr(self.memory, "memdir", None) or getattr(self.memory, "base_dir", None) or "."
        return os.getenv("MB_PDNA_PATH") or str(Path(base) / "pdna.json")

    def _load_pdna(self, path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # sensible defaults
            return {"fun": 0.5, "salience": 0.0, "eval": 0.5}

    def _save_pdna(self) -> None:
        try:
            Path(self._pdna_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._pdna_path, "w", encoding="utf-8") as f:
                json.dump(self.pdna, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_pdna(
        self, *, fun: float | None = None, eval: float | None = None, y: float = 0.0, x: float = 0.6
    ) -> None:
        """Rule-of-thumb: if fun >= x, salience += y + eval (your earlier idea)."""
        if fun is not None:
            self.pdna["fun"] = float(fun)
        if eval is not None:
            self.pdna["eval"] = float(eval)
        if float(self.pdna.get("fun", 0.0)) >= float(x):
            self.pdna["salience"] = (
                float(self.pdna.get("salience", 0.0)) + float(y) + float(self.pdna.get("eval", 0.0))
            )
        self._save_pdna()

    # ------------------------------------------------------------------
    def step(self, user_input: str) -> str:
        if self.memory is None:
            return (
                "Memory subsystem is unavailable; I can chat, but can’t recall/write memories yet."
            )
        # Log perception
        self.memory.add_episodic(f"USER: {user_input}", {"role": "user"})

        # Retrieve context
        sem = self.memory.search_semantic(user_input, k=5)
        epis = self.memory.last_episodic(3)

        context_lines = []
        if sem:
            context_lines.append("Relevant semantic memory:")
            for h in sem:
                context_lines.append(f"- {h['text']}")
        if epis:
            context_lines.append("\nRecent episodic memory:")
            for e in epis:
                context_lines.append(f"- {e['text']}")

        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_input},
        ]
        if context_lines:
            messages.insert(1, {"role": "system", "content": "\n".join(context_lines)})

        # Prefer chat; fall back to generate
        try:
            reply = self.llm.chat(messages, options={"temperature": 0.2})
        except Exception:
            prompt = (
                f"{self.system}\n\n"
                f"{'\n'.join(context_lines)}\n\n"
                f"User: {user_input}\n"
                f"Plan step-by-step, call tools if useful, then answer."
            )
            reply = self.llm.generate(prompt, options={"temperature": 0.2})
        # Store semantic reflection of reply for future retrieval
        self.memory.add_semantic(reply, {"role": "assistant"})
        self.memory.add_episodic(f"ASSISTANT: {reply}", {"role": "assistant"})
        return reply

    async def complete(self, user_input: str) -> str:
        # run the sync step() on a worker thread so callers can await us
        return await asyncio.to_thread(self.step, user_input)
