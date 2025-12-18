# Imports
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from microbrain.memory.emotional_journal import EmotionJournal

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
    #   def __init__(self, llm, memory, tools, logger, bus=None, *args, **kwargs):
    def __init__(
        self,
        llm,
        memory,
        tools,
        logger,
        bus: Any | None = None,
        ejournal: EmotionJournal | None = None,
    ):
        # normalize aliases: prefer llm, then client, then ollama
        chosen = llm
        if chosen is None:
            raise ValueError("Agent requires an LLM client (llm/client/ollama)")

        # keep multiple names for back-compat across the codebase
        self.llm = chosen
        self.client = chosen
        self.ollama = chosen
        self.memory = memory
        self.tools = tools or ToolRegistry()
        self.logger = logger
        self.bus = bus
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
        self.ejournal = ejournal

    def _emit(self, topic: str, **payload):
        try:
            if self.bus is not None:
                self.bus.publish(topic, **payload)
        except Exception:
            # keep the agent resilient; bus issues shouldn’t crash interaction
            pass

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

    def _estimate_affect(self, text: str) -> tuple[float, float, float]:
        """Return (valence[-1..1], arousal[0..1], salience[0..1]) from a quick heuristic."""
        t = text.lower()
        pos = {"great", "good", "nice", "love", "awesome", "yay", "glad", "thanks", "cool", "win"}
        neg = {"bad", "sad", "angry", "hate", "terrible", "awful", "nope", "ugh", "fail"}
        intens = {"very", "really", "so", "extremely", "super", "!!!", "!!", "!"}

        words = re.findall(r"[a-z']+", t)
        if not words:
            return 0.0, 0.0, 0.1

        npos = sum(w in pos for w in words)
        nneg = sum(w in neg for w in words)
        nint = sum(w in intens for w in words) + t.count("!") + t.count("?")

        valence = (npos - nneg) / max(1, (npos + nneg))
        valence = max(-1.0, min(1.0, valence))

        arousal = min(1.0, (nint / max(5, len(words))) * 3.0)

        has_code = ("```" in text) or ("{" in text and "}" in text)
        longish = len(text) > 280
        questions = t.count("?")
        salience = 0.2 + 0.3 * has_code + 0.2 * longish + 0.3 * min(1.0, questions / 3)
        salience = max(0.0, min(1.0, salience))

        return float(valence), float(arousal), float(salience)

    # ------------------------------------------------------------------
    def step(self, user_input: str) -> str:
        # --- USER turn affect/journal/meta ---
        u_val, u_aro, u_sal = self._estimate_affect(user_input)
        self._emit("user.affect.estimated", valence=u_val, arousal=u_aro, salience=u_sal)

        if hasattr(self, "ejournal") and self.ejournal:
            try:
                self.ejournal.record(
                    actor="user",
                    text=user_input,
                    valence=u_val,
                    arousal=u_aro,
                    salience=u_sal,
                    tags=["auto"],
                )
            except Exception:
                if self.logger:
                    self.logger.debug("EmotionJournal record (user) failed", exc_info=True)

        u_meta = {"role": "user", "valence": u_val, "arousal": u_aro, "salience": u_sal}
        # Write user turn into memory with affect metadata
        if hasattr(self, "memory") and self.memory:
            try:
                self.memory.add_semantic(user_input, u_meta)
                self.memory.add_episodic(f"USER: {user_input}", u_meta)
                self._emit("user.memory.write", wrote_semantic=True, wrote_episodic=True, **u_meta)
            except Exception:
                if self.logger:
                    self.logger.debug("Memory write (user) failed", exc_info=True)

        # Optional pDNA nudge
        if u_sal > 0.25:
            self._emit("user.pdna.feedback", valence=u_val, salience=u_sal)
        # --- end USER turn block ---

        if self.memory is None:
            return (
                "Memory subsystem is unavailable; I can chat, but can’t recall/write memories yet."
            )
        # Log perception
        self.memory.add_episodic(f"USER: {user_input}", {"role": "user"})
        self._emit("agent.input", text=user_input)
        self._emit("nl.input", text=user_input)

        # Retrieve context
        sem = self.memory.search_semantic(user_input, k=5)
        epis = self.memory.last_episodic(3)
        self._emit(
            "agent.context.built", semantic_count=len(sem or []), episodic_count=len(epis or [])
        )

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
        self._emit("agent.llm.request", message_count=len(messages))

        # Prefer chat; fall back to generate
        try:
            reply = self.llm.chat(messages, options={"temperature": 0.2})
            self._emit("agent.llm.reply", text=reply)
        except Exception:
            context_block = "\n".join(context_lines)
            prompt = (
                f"{self.system}\n\n"
                f"{context_block}\n\n"
                f"User: {user_input}\n"
                f"Plan step-by-step, call tools if useful, then answer."
            )
            reply = self.llm.generate(prompt, options={"temperature": 0.2})
        val, aro, sal = self._estimate_affect(reply)
        self._emit("agent.affect.estimated", valence=val, arousal=aro, salience=sal)
        self._emit("llm.output", text=reply, salience=sal)
        # log to emotion journal
        if hasattr(self, "ejournal") and self.ejournal:
            try:
                self.ejournal.record(
                    actor="assistant",
                    text=reply,
                    valence=val,
                    arousal=aro,
                    salience=sal,
                    tags=["auto"],
                )
            except Exception:
                if self.logger:
                    self.logger.debug("EmotionJournal record failed", exc_info=True)
        # Store semantic reflection of reply for future retrieval
        meta = {"role": "assistant", "valence": val, "arousal": aro, "salience": sal}
        self.memory.add_semantic(reply, meta)
        self.memory.add_episodic(f"ASSISTANT: {reply}", meta)
        self._emit("agent.memory.write", wrote_semantic=True, wrote_episodic=True, **meta)
        if sal > 0.25:
            self._emit("agent.pdna.feedback", valence=val, salience=sal)
        return reply

    async def complete(self, user_input: str) -> str:
        # run the sync step() on a worker thread so callers can await us
        return await asyncio.to_thread(self.step, user_input)
