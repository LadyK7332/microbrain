from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


class RecollectionNeuron(BaseNeuron):


    async def _mem_cell_store(self, ctx) -> Optional[MemCellStore]:
        store = await ctx.get_kv("memory:mem_cell_store", None)
        if store is not None:
            return store
        try:
            memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
            store = MemCellStore(memdir)
            await ctx.set_kv("memory:mem_cell_store", store)
            return store
        except Exception:
            return None

    def _mem_cell_snippets(self, hits: List[Dict[str, Any]], *, limit: int = 5) -> List[str]:
        snippets: List[str] = []
        for hit in hits[:max(1, int(limit))]:
            anchor_text = str(hit.get("anchor_text", "") or "").strip()
            refs = hit.get("refs", []) if isinstance(hit.get("refs", []), list) else []
            body = anchor_text or (str(refs[0]) if refs else "")
            if not body:
                continue
            if len(body) > 220:
                body = body[:220] + "..."
            snippets.append(f"- {body} [tier={hit.get('tier', 'now')} score={float(hit.get('score', 0.0)):.3f}]")
        return snippets

    def _build_native_recollection_reply(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "I don't have a strong recollection anchor for that yet."
        lead = hits[0]
        score = float(lead.get("score", 0.0) or 0.0)
        anchor_text = str(lead.get("anchor_text", "") or "").strip()
        refs = lead.get("refs", []) if isinstance(lead.get("refs", []), list) else []
        body = anchor_text or (str(refs[0]) if refs else "")
        if len(body) > 180:
            body = body[:177] + "..."
        if score >= 0.70:
            prefix = "I remember something close to that:"
        elif score >= 0.45:
            prefix = "I have a partial recollection:"
        else:
            prefix = "I only have a weak recollection:"
        return f"{prefix} {body}"
    """
    Recollection / memory search neuron (v2).

    Listens on:
        - "memory/recollect"

    Emits:
        - "reason/request"  (for LLMReasonerNeuron)

    Behavior v2:
      - Takes a natural-language recollection query (e.g. "remember that funny thing
        about jaguars you said early last week?").
      - Uses HRM to create a temporary query node and then find nearest neighbors.
      - Optionally biases neighbors using temporal tags stored on HRM nodes:
          - node.day_index
          - node.week_index
          - node.local_weekday
      - Collects a few of the most relevant past text nodes.
      - Builds a prompt asking the reasoning core to "remember" and paraphrase
        what it likely said or meant back then.
      - Instructs the LLM to prefer honesty ("I'm not sure") over fabrication if
        nothing matches well.
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

        payload = event.payload
        if not isinstance(payload, dict):
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload for memory/recollect",
                payload_type=str(type(payload)),
            )
            return []

        query_text = str(payload.get("query_text", "") or "").strip()
        raw_user_text = str(payload.get("raw_user_text", "") or "").strip()
        channel = str(payload.get("channel", "default"))
        source = str(payload.get("source", "user"))
        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}

        if not query_text and raw_user_text:
            query_text = raw_user_text

        if not query_text:
            await ctx.log_debug(
                f"[{self.name}] Empty recollection query; nothing to do",
                channel=channel,
            )
            return []

        # Optional time hint: infer from user phrasing ("yesterday", "last week", etc.)
        time_hint = self._extract_time_hint(raw_user_text or query_text)

        # Optional context cues: modality / window / mood / novelty (best-effort)
        context_cues = self._extract_context_cues(raw_user_text or query_text, raw_meta)

        mem_cell_store = await self._mem_cell_store(ctx)
        mem_cell_hits: List[Dict[str, Any]] = []
        mem_cell_snippets: List[str] = []
        if mem_cell_store is not None:
            try:
                mem_cell_hits = mem_cell_store.search_text_cells(query_text, limit=6, tiers=("now", "short", "long"))
                mem_cell_snippets = self._mem_cell_snippets(mem_cell_hits, limit=5)
            except Exception:
                mem_cell_hits = []
                mem_cell_snippets = []

        # Try to get HRM core
        hrm = await ctx.get_kv("hrm:core", None)
        if hrm is None:
            bundle = {
                "seed_kind": "recollection",
                "query_text": query_text,
                "channel": channel,
                "time_hint": time_hint,
                "top_cells": mem_cell_hits[:5],
                "top_concepts": [],
                "kind": "mem_cell_recollection_bundle",
            }
            await ctx.set_kv("recall:last_bundle", bundle)
            if mem_cell_hits:
                return [
                    Event(topic="memory/recall_context", payload=bundle, source=self.name, correlation_id=event.correlation_id),
                    Event(
                        topic="act/speech",
                        payload={"text": self._build_native_recollection_reply(mem_cell_hits), "channel": channel, "style": "assistant"},
                        source=self.name,
                        correlation_id=event.correlation_id,
                        meta={"kind": "recollection_native_memcell"},
                    ),
                ]

            # No HRM and no mem-cell hits; fall back to a simple native response.
            return [Event(
                topic="act/speech",
                payload={
                    "text": "I don't have a reliable recollection anchor for that right now.",
                    "channel": channel,
                    "style": "assistant",
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "recollection_fallback_native"},
            )]

        # ------------------------------
        # Build a temporary HRM query node & find neighbors
        # ------------------------------
        try:
            query_node = hrm.observe(query_text, role="recollection_query")
        except Exception as exc:
            await ctx.log_error(
                f"[{self.name}] Error creating HRM query node",
                exception=str(exc),
            )
            query_node = None

        memory_snippets: List[str] = []
        neighbor_debug: List[Dict[str, Any]] = []

        if query_node is not None:
            try:
                # Pull a bigger set, then re-rank with time bias
                base_neighbors: List[Tuple[int, float]] = hrm.neighbors(
                    query_node.idx, k=24
                )
            except Exception as exc:
                await ctx.log_error(
                    f"[{self.name}] Error getting HRM neighbors",
                    exception=str(exc),
                )
                base_neighbors = []

            # Compute time bias if possible
            biased_neighbors: List[Tuple[int, float]] = []
            today_day_index = int(time.time() // 86400)

            for idx, weight in base_neighbors:
                try:
                    node = hrm.get_node(idx)
                except Exception:
                    node = None

                if not node:
                    continue

                # Default: no bias
                effective_weight = weight
                in_window = False

                if time_hint is not None:
                    in_window = self._node_matches_time_hint(
                        node=node,
                        hint=time_hint,
                        today_day_index=today_day_index,
                    )

                    # Simple scheme:
                    #  - if in window: boost slightly
                    #  - if not in window: slight penalty
                    if in_window:
                        effective_weight *= 1.4
                    else:
                        effective_weight *= 0.85

                cue_mult = self._context_bias_multiplier(node=node, cues=context_cues)
                effective_weight *= cue_mult

                biased_neighbors.append((idx, effective_weight))
                neighbor_debug.append(
                    {
                        "idx": idx,
                        "base_weight": weight,
                        "effective_weight": effective_weight,
                        "in_window": in_window,
                        "cue_mult": cue_mult,
                    }
                )

            # Sort by effective weight, highest first
            biased_neighbors.sort(key=lambda t: t[1], reverse=True)

            # Collect top few memory snippets
            for idx, eff_weight in biased_neighbors:
                try:
                    node = hrm.get_node(idx)
                except Exception:
                    node = None
                if not node:
                    continue

                text_j = getattr(node, "text", "") or ""
                text_j = str(text_j).strip()
                if not text_j:
                    continue

                if len(text_j) > 220:
                    text_j = text_j[:220] + "..."

                memory_snippets.append(f"- {text_j}")
                if len(memory_snippets) >= 5:
                    break

        # Blend in memory-cell snippets as a parallel bounded memory source.
        for snippet in mem_cell_snippets:
            if snippet not in memory_snippets:
                memory_snippets.append(snippet)
            if len(memory_snippets) >= 6:
                break

        # ------------------------------
        # Build recollection prompt
        # ------------------------------
        prompt_lines: List[str] = []

        prompt_lines.append(
            "The user is asking you to remember something you said or did in the past."
        )
        if raw_user_text:
            prompt_lines.append(
                f"User's recollection request (verbatim): {raw_user_text}"
            )
        else:
            prompt_lines.append(
                f"User's recollection request (parsed): {query_text}"
            )
        prompt_lines.append("")

        if time_hint is not None:
            prompt_lines.append(
                f"Timeframe (interpreted from the user's phrasing): {time_hint}"
            )
            prompt_lines.append("")

        if context_cues:
            prompt_lines.append("Context cues detected (best-effort):")
            prompt_lines.append(json.dumps(context_cues, ensure_ascii=False))
            prompt_lines.append("")

        if memory_snippets:
            prompt_lines.append(
                "Here are some of your most relevant past thoughts or messages, "
                "retrieved from your long-term memory (HRM):"
            )
            prompt_lines.extend(memory_snippets)
        else:
            prompt_lines.append(
                "No clearly matching memories could be found in your long-term memory."
            )

        prompt_lines.append("")
        prompt_lines.append(
            "Based on this, try to remember and explain what you probably said or meant. "
            "If nothing matches well, be honest that you don't clearly remember, "
            "rather than inventing details."
        )
        prompt_lines.append(
            "Keep it brief (3–6 sentences) and speak in first person as MicroBrain."
        )

        prompt = "\n".join(prompt_lines)

        await ctx.log_debug(
            f"[{self.name}] Built recollection prompt",
            channel=channel,
            has_memories=bool(memory_snippets),
            num_memories=len(memory_snippets),
            time_hint=time_hint,
        )

        reason_payload: Dict[str, Any] = {
            "text": prompt,
            "source": "system",
            "channel": channel,
            "raw_meta": {
                "mode": "recollection",
                "raw_user_text": raw_user_text,
                "query_text": query_text,
                "time_hint": time_hint,
                "neighbor_debug": neighbor_debug[:10],  # small sample
                "raw_meta": raw_meta,
            },
        }

        reason_event = Event(
            topic="reason/request",
            payload=reason_payload,
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"kind": "recollection_request"},
        )

        return [reason_event]

    # ------------------------------------------------------------------
    # Time hint extraction
    # ------------------------------------------------------------------
    def _extract_time_hint(self, text: str) -> Optional[str]:
        """
        Very simple phrase-to-hint mapping.
        Returns a small string label like "yesterday", "last_week", "early_last_week", etc.
        """
        lowered = (text or "").lower()
        if not lowered:
            return None

        # Order matters: more specific phrases first
        if "early last week" in lowered:
            return "early_last_week"
        if "late last week" in lowered:
            return "late_last_week"
        if "last week" in lowered:
            return "last_week"
        if "yesterday" in lowered:
            return "yesterday"
        if "last night" in lowered:
            return "last_night"
        if "this morning" in lowered:
            return "this_morning"
        if "earlier today" in lowered or "earlier this day" in lowered:
            return "earlier_today"
        if "today" in lowered:
            return "today"
        if "breakfast" in lowered:
            return "breakfast"

        # Add more phrases as needed over time
        return None
    
    # ------------------------------------------------------------------
    # Context cue extraction + bias
    # ------------------------------------------------------------------
    def _extract_context_cues(self, text: str, raw_meta: Dict[str, Any]) -> Dict[str, Any]:
        cues: Dict[str, Any] = {}

        lowered = (text or "").lower()

        # Modality cues (best-effort keyword sniffing)
        modalities: List[str] = []
        if any(k in lowered for k in ("see", "look", "screen", "window", "image", "picture", "vision")):
            modalities.append("vision")
        if any(k in lowered for k in ("hear", "audio", "sound", "voice", "said", "music")):
            modalities.append("audio")
        if any(k in lowered for k in ("touch", "feel", "tactile", "grip", "pressure")):
            modalities.append("touch")
        if modalities:
            cues["modalities"] = sorted(set(modalities))

        # Window/app hint from raw_meta (if present)
        for k in ("window_title", "app", "application", "window"):
            v = raw_meta.get(k)
            if isinstance(v, str) and v.strip():
                cues["window_hint"] = v.strip()
                break

        # Mood hint (lightweight; only if explicitly mentioned)
        for mood in ("happy", "sad", "angry", "anxious", "calm", "excited"):
            if mood in lowered:
                cues["mood_hint"] = mood
                break

        # Novelty cue
        if any(k in lowered for k in ("new", "first time", "never", "novel")):
            cues["novelty_hint"] = True

        return cues

    def _context_bias_multiplier(self, node: Any, cues: Dict[str, Any]) -> float:
        if not cues:
            return 1.0

        mult = 1.0

        # If modality cues exist, reward nodes whose tags/role/topic mention them.
        modalities = cues.get("modalities")
        if isinstance(modalities, list) and modalities:
            node_role = str(getattr(node, "role", "") or "").lower()
            node_topic = str(getattr(node, "topic", "") or "").lower()
            node_mod = str(getattr(node, "modality", "") or "").lower()

            matched = any(m in node_role or m in node_topic or m == node_mod for m in modalities)
            mult *= 1.20 if matched else 0.95

        # Window hint (only if node exposes it)
        win = cues.get("window_hint")
        if isinstance(win, str) and win:
            node_win = str(getattr(node, "window_title", "") or getattr(node, "window", "") or "")
            if node_win and win.lower() in node_win.lower():
                mult *= 1.15

        # Mood hint (only if node exposes it)
        mood = cues.get("mood_hint")
        node_mood = getattr(node, "mood", None)
        if isinstance(mood, str) and mood and node_mood is not None:
            if str(node_mood).lower() == mood:
                mult *= 1.10

        return mult

    # ------------------------------------------------------------------
    # Time window matching against node tags
    # ------------------------------------------------------------------
    def _node_matches_time_hint(
        self,
        node: Any,
        hint: str,
        today_day_index: int,
    ) -> bool:
        """
        Decide whether a given HRM node falls into the interpreted time window
        for the given hint. We use the node's day_index/week_index/local_weekday
        if present; otherwise, we can't match.
        """
        day_index = getattr(node, "day_index", None)
        week_index = getattr(node, "week_index", None)
        local_weekday = getattr(node, "local_weekday", None)

        # If we don't have any time tags, we can't say it matches.
        if day_index is None and week_index is None:
            return False

        # Compute some basics from today.
        current_day_index = today_day_index
        current_week_index = current_day_index // 7

        # Helper to check day range
        def in_day_range(start_offset: int, end_offset: int) -> bool:
            if day_index is None:
                return False
            start = current_day_index + start_offset
            end = current_day_index + end_offset
            return start <= day_index <= end

        # Helper to check exact week
        def in_week_offset(offset: int) -> bool:
            if week_index is None:
                return False
            target_week = current_week_index + offset
            return week_index == target_week

        # Interpret hints. These are approximate by design.
        if hint == "yesterday":
            return in_day_range(-1, -1)

        if hint == "today":
            return in_day_range(0, 0)

        if hint == "earlier_today":
            if not in_day_range(0, 0):
                return False
            # Could refine later with local_hour; for now, "same day" is enough.
            return True

        if hint == "last_night":
            # Approximate: either late yesterday or very early today.
            if in_day_range(-1, -1):
                return True
            return in_day_range(0, 0)

        if hint == "breakfast":
            if not in_day_range(0, 0):
                return False
            local_hour = getattr(node, "local_hour", None)
            if local_hour is None:
                return True
            return local_hour < 11
        
        if hint == "last_week":
            return in_week_offset(-1)

        if hint == "early_last_week":
            if not in_week_offset(-1):
                return False
            # Early part ~ Mon-Wed
            if local_weekday is None:
                return True
            return local_weekday in (0, 1, 2)

        if hint == "late_last_week":
            if not in_week_offset(-1):
                return False
            # Late part ~ Thu-Sun
            if local_weekday is None:
                return True
            return local_weekday in (3, 4, 5, 6)

        # Unknown hint -> don't match
        return False


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["memory/recollect"],
        output_topics=["reason/request", "act/speech", "memory/recall_context"],
        priority=8,  # runs after router/introspect, before general chatter if needed
    )
    yield RecollectionNeuron(cfg)
