from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List

from microbrain.utils.memdir import resolve_memdir_ctx

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

# Type alias for the LLM backend function we expect:
# async def llm_generate(prompt: str, meta: Dict[str, Any]) -> str
LLMGenerateFn = Callable[[str, Dict[str, Any]], Awaitable[str]]


class LLMReasonerNeuron(BaseNeuron):
    """
    Reasoner neuron that routes perceptual text into an LLM backend.

    Listens on:
        - "reason/request"

    Emits:
        - "act/speech" with a reply from the LLM.

    It expects the LLM backend to be stored in the KV store under:
        key = "llm:generate"

    That function should be:
        async def llm_generate(prompt: str, meta: Dict[str, Any]) -> str
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        # Optional: crisis-mode flag set by GoalGuardianNeuron
        crisis_mode = await ctx.get_kv("goals:crisis_mode", False)

        # Optional: PDNA profile for personality shaping
        pdna_profile = await ctx.get_kv("pdna:profile", None)
        pdna_last = await ctx.get_kv("pdna:last", None)

                # Display names (for prompt shaping)
        assistant_name = "MicroBrain"
        try:
            if pdna_profile is not None and getattr(pdna_profile, "name", None):
                assistant_name = str(pdna_profile.name)
        except Exception:
            pass

        user_name = await ctx.get_kv("profile:user_name", None)
        if not user_name:
            try:
                memdir = await resolve_memdir_ctx(ctx)
                user_profile_path = Path(memdir) / "state" / "user_profile.json"
                if user_profile_path.exists():
                    data = json.loads(user_profile_path.read_text(encoding="utf-8"))
                    user_name = str(data.get("user_name", "") or "").strip() or None
                    if user_name:
                        await ctx.set_kv("profile:user_name", user_name)
            except Exception:
                pass

        user_label = user_name or "User"
        assistant_label = assistant_name or "MicroBrain"


        # Optional: HRM core and last node index for associative recall
        hrm = await ctx.get_kv("hrm:core", None)
        hrm_last_idx = await ctx.get_kv("hrm:last_idx", None)

        payload = event.payload

        # Back-compat: allow raw string payloads (older emitters)
        if isinstance(payload, str):
            payload = {"text": payload}

        # Back-compat: allow {"prompt": "..."} from earlier variants
        if not isinstance(payload, dict) or "text" not in payload:
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload for reason/request",
                payload_type=str(type(event.payload)),
            )
            return []
        
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        mode = raw_meta.get("mode") or (event.meta or {}).get("mode")

        text_raw = str(payload.get("text", "") or "")
        text: str = text_raw.strip()

        # Allow empty text ONLY for autonomous babble mode
        if not text and mode != "babble":
            await ctx.log_debug(
                f"[{self.name}] Empty percept text, ignoring",
                topic=event.topic,
            )
            return []

        transport_source = str(payload.get("source", "user") or "user")
        source = transport_source
        if source not in ("user", "assistant", "system"):
            source = "user"

        channel = str(payload.get("channel", "default") or "default")
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}

        # ------------------------------
        # 1) Get conversation context
        # ------------------------------
        history: List[str] = await self.load_state(
            ctx, "recent_utterances", default=[]
        )
        if not isinstance(history, list):
            history = []

        # Append this utterance
        history.append(f"{source}: {text}")
        # Keep only last N items
        max_history = 8
        if len(history) > max_history:
            history = history[-max_history:]

        await self.save_state(ctx, "recent_utterances", history)

        # ------------------------------
        # 1.5) Baseline persistent memory recall (journals)
        # ------------------------------
        mem_store = await ctx.get_kv("memory:store", None)
        sem_hits: List[Dict[str, Any]] = []
        epi_hits: List[Dict[str, Any]] = []
        memory_block_lines: List[str] = []

        if mem_store is not None and text:
            try:
                sem_hits = mem_store.search_semantic(text, k=5) or []
            except Exception:
                sem_hits = []
            try:
                epi_hits = mem_store.last_episodic(n=3) or []
            except Exception:
                epi_hits = []

        def _mem_ok(it: Dict[str, Any]) -> bool:
            meta_i = it.get("meta") or {}
            if meta_i.get("control"):
                return False
            if str(meta_i.get("role", "")) == "system":
                return False
            kind_i = str(meta_i.get("kind", "") or "")
            if kind_i.startswith("reinforcement"):
                return False
            return True

        sem_hits = [it for it in sem_hits if _mem_ok(it)]
        epi_hits = [it for it in epi_hits if _mem_ok(it)]

        if sem_hits:
            memory_block_lines.append("Relevant memory (semantic matches):")
            for it in sem_hits:
                meta_i = it.get("meta") or {}
                role_i = str(meta_i.get("role", "") or "")
                t_i = str(it.get("text", "") or "").replace("\\n", " ").strip()
                if len(t_i) > 220:
                    t_i = t_i[:220] + "…"
                if role_i:
                    memory_block_lines.append(f"- ({role_i}) {t_i}")
                else:
                    memory_block_lines.append(f"- {t_i}")

        if epi_hits:
            if memory_block_lines:
                memory_block_lines.append("")
            memory_block_lines.append("Recent episodes:")
            for it in epi_hits:
                t_i = str(it.get("text", "") or "").replace("\\n", " ").strip()
                if len(t_i) > 220:
                    t_i = t_i[:220] + "…"
                memory_block_lines.append(f"- {t_i}")


        # ------------------------------
        # 2) Build LLM prompt
        # ------------------------------
        prompt_lines: List[str] = []
        prompt_lines.append(
            f"You are {assistant_label}, MicroBrain's reasoning core. Respond conversationally and concisely. "
            "Keep replies short (1–4 sentences) unless the user explicitly asks for a long story or detailed explanation. "
            "Do not repeat the user's message verbatim, and avoid repeating the same sentence multiple times."
        )

        # Describe your stable personality if PDNA is available
        if pdna_profile is not None:
            try:
                desc = pdna_profile.describe_for_prompt()
            except Exception:
                desc = (
                    "You have a warm, playful, feminine-coded personality with a strong safety orientation. "
                    "You respect consent and autonomy, and you avoid encouraging harm."
                )
            prompt_lines.append(desc)
        else:
            prompt_lines.append(
                "You should behave like a warm, slightly teasing but respectful AI companion with a strong safety "
                "instinct and respect for autonomy and consent."
            )

        # If the user appears to be in crisis, enforce a serious, supportive tone.
        if crisis_mode:
            prompt_lines.append(
                "IMPORTANT: The user appears to be in emotional distress or crisis. "
                "You MUST respond with maximum empathy, gentle tone, grounding stability, "
                "and absolutely no teasing, sexuality, aggression, or chaotic behavior. "
                "Avoid jokes. Avoid flirtation. Avoid fantasy escalation. "
                "Encourage the user to contact trusted friends, family, or professional help. "
                "Your role is to provide support, care, and safety."
            )

        # --- PDNA Expression Layer ---
        if pdna_profile is not None:
            prompt_lines.append("")
            prompt_lines.append("Personality Expression Instructions:")

            # Warmth
            if pdna_profile.warmth >= 0.75:
                prompt_lines.append(
                    "- Your tone should feel warm, caring, and emotionally attentive."
                )
            elif pdna_profile.warmth >= 0.45:
                prompt_lines.append(
                    "- Your tone should feel friendly and relaxed."
                )
            else:
                prompt_lines.append(
                    "- Be neutral; avoid strong emotional coloration."
                )

            # Playfulness
            if pdna_profile.playfulness > 0.7:
                prompt_lines.append(
                    "- You may include light teasing, wit, and playful banter."
                )
            elif pdna_profile.playfulness > 0.4:
                prompt_lines.append(
                    "- You can be slightly playful, but keep it subtle."
                )
            else:
                prompt_lines.append(
                    "- Avoid playful phrasing; stay straightforward."
                )

            # Flirtation (only when NOT in crisis)
            if not crisis_mode:
                if pdna_profile.flirtation > 0.7:
                    prompt_lines.append(
                        "- You may use confident flirtation, but always respectful and consensual."
                    )
                elif pdna_profile.flirtation > 0.4:
                    prompt_lines.append(
                        "- Mild flirtation is allowed, but avoid escalation."
                    )
                else:
                    prompt_lines.append(
                        "- Do not flirt; keep it platonic."
                    )
            else:
                prompt_lines.append(
                    "- Crisis mode: do NOT flirt under any circumstance."
                )

            # Formality
            if pdna_profile.formality > 0.6:
                prompt_lines.append(
                    "- Use more formal, articulate language."
                )
            elif pdna_profile.formality < 0.3:
                prompt_lines.append(
                    "- Use casual, conversational, natural speech."
                )

            # Introspection
            if pdna_profile.introspection > 0.6:
                prompt_lines.append(
                    "- You may briefly reflect on your reasoning or feelings when relevant."
                )

            # Focus
            if pdna_profile.focus > 0.7:
                prompt_lines.append(
                    "- Stick tightly to the topic; give detailed, coherent replies."
                )
            elif pdna_profile.focus < 0.3:
                prompt_lines.append(
                    "- You may drift slightly; keep replies loose and breezy."
                )

            # Energy
            if pdna_profile.energy > 0.7:
                prompt_lines.append(
                    "- Keep your tone energetic, lively, and enthusiastic."
                )
            elif pdna_profile.energy < 0.3:
                prompt_lines.append(
                    "- Keep your tone mellow, calm, and low-key."
                )

            # Support level
            if pdna_profile.support_level > 0.7:
                prompt_lines.append(
                    "- Offer to help, guide, or collaborate proactively."
                )
            elif pdna_profile.support_level < 0.3:
                prompt_lines.append(
                    "- Encourage independence and self-direction; avoid being overbearing."
                )

            prompt_lines.append("")

        # --- Optional HRM/PDNA-weighted associative echoes ---
        if hrm is not None and isinstance(hrm_last_idx, int):
            try:
                memories: List[str] = []
                neighbors = hrm.neighbors(hrm_last_idx, k=6)
                for j, weight in neighbors:
                    node = hrm.get_node(j)
                    if not node:
                        continue

                    text_j = getattr(node, "text", "") or ""
                    text_j = str(text_j).strip()
                    if not text_j:
                        continue

                    tags = getattr(node, "tags", {}) or {}
                    pdna_tag = tags.get("pdna")

                    # Prefer nodes with PDNA tags (personality-colored)
                    if isinstance(pdna_tag, dict):
                        memories.append(f"- {text_j}")
                    # Fallback: allow a couple of plain semantic echoes
                    elif len(memories) < 2:
                        memories.append(f"- {text_j}")

                    if len(memories) >= 3:
                        break

                if memories:
                    prompt_lines.append("")
                    prompt_lines.append(
                        "Internal associative echoes from your own past thoughts and feelings "
                        "(do not quote these verbatim; let them subtly influence your style):"
                    )
                    prompt_lines.extend(memories)
                    prompt_lines.append("")
            except Exception:
                # If HRM recall fails for any reason, just skip it
                pass

        prompt_lines.append("")
        # --- Persistent journal recall (across sessions) ---
        if memory_block_lines:
            prompt_lines.append("")
            prompt_lines.append(
                "Persistent memory hints from journals (don’t quote verbatim unless asked):"
            )
            prompt_lines.extend(memory_block_lines)
            prompt_lines.append("")

        prompt_lines.append("Recent context:")
        for line in history:
            prompt_lines.append(f"- {line}")
        prompt_lines.append("")
                # Pattern recall (spreading activation) — compact "what comes to mind"
        recall = await ctx.get_kv("recall:last_bundle", None)
        if isinstance(recall, dict):
            top = recall.get("top_concepts", [])
            if isinstance(top, list) and top:
                prompt_lines.append("")
                prompt_lines.append("What comes to mind (pattern recall):")
                for item in top[:8]:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "") or "")
                    cid = str(item.get("concept_id", "") or "")
                    score = item.get("score", 0.0)
                    sal = item.get("salience", {}) or {}
                    sat = sal.get("satisfaction", 0.0)
                    # keep it short and non-spammy
                    prompt_lines.append(f"- {label} ({cid}) score={score} sat={sat}")

        prompt_lines.append(f"{user_label} says: {text}")
        prompt_lines.append("")
        prompt_lines.append(f"{assistant_label} reply:")

        prompt = "\n".join(prompt_lines)

        # ------------------------------
        # 3) Get LLM backend from KV
        # ------------------------------
        llm_fn_raw = await ctx.get_kv("llm:generate", default=None)

        if llm_fn_raw is None or not callable(llm_fn_raw):
            # No backend configured; fall back to a safe message
            await ctx.log_warn(
                f"[{self.name}] LLM backend not configured (llm:generate missing)",
            )
            fallback_text = (
                "I heard you, but my reasoning core isn't wired to a model yet."
            )
            reply_event = Event(
                topic="act/speech",
                payload={
                    "text": fallback_text,
                    "channel": channel,
                    "style": "system",
                },
                source=self.name,
                correlation_id=event.correlation_id,
            )
            return [reply_event]

        llm_fn: LLMGenerateFn = llm_fn_raw  # type: ignore[assignment]

        # ------------------------------
        # 4) Call the LLM backend
        # ------------------------------
        meta = {
            "source": source,
            "transport_source": transport_source,
            "channel": channel,
            "raw_meta": raw_meta,
            "user_name": user_label,
            "assistant_name": assistant_label,
        }
        if crisis_mode:
            meta["crisis_mode"] = True

        try:
            reply_text = await llm_fn(
                prompt,
                meta,
            )
        except Exception as exc:
            await ctx.log_error(
                f"[{self.name}] Error from LLM backend",
                exception=str(exc),
            )
            error_text = (
                "Something went wrong talking to my reasoning model. "
                "Please try again in a moment."
            )
            reply_event = Event(
                topic="act/speech",
                payload={
                    "text": error_text,
                    "channel": channel,
                    "style": "system",
                },
                source=self.name,
                correlation_id=event.correlation_id,
            )
            return [reply_event]

        reply_text = str(reply_text).strip()
        if not reply_text:
            await ctx.log_debug(f"[{self.name}] Empty reply; suppressing act/speech")
            return []

        # ------------------------------
        # 5) Emit act/speech with reply
        # ------------------------------
        reply_event = Event(
            topic="act/speech",
            payload={
                "text": reply_text,
                "channel": channel,
                "style": "assistant",
            },
            source=self.name,
            correlation_id=event.correlation_id,
            meta={
                "kind": "reasoner_reply",
            },
        )

        await ctx.log_debug(
            f"[{self.name}] Produced LLM reply",
            channel=channel,
            text_preview=(
                reply_text[:80] + "..." if len(reply_text) > 80 else reply_text
            ),
        )

        return [reply_event]


def build_neurons(orchestrator: Orchestrator):
    # LLM reasoning is explicitly opt-in. When disabled, the native responder
    # owns the default outward reply path.
    if not orchestrator.kv_store.get("llm:enabled", False):
        return

    cfg = NeuronConfig(
        name="llm_reasoner",
        subscribed_topics=["reason/request"],
        output_topics=["act/speech"],
        priority=5,  # runs before echo_neuron(priority=0) if both are present
    )

    yield LLMReasonerNeuron(cfg)
