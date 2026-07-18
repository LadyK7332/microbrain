from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "its", "just", "me", "my", "of", "or",
    "our", "so", "that", "the", "then", "there", "this", "to", "was", "we", "what", "when",
    "where", "who", "why", "will", "with", "you", "your",
}

THREAD_MARKERS = {
    "pressure", "scene", "conversation", "context", "memory", "need", "needs", "power",
    "maintenance", "uplift", "curiosity", "novelty", "prediction", "expectation", "question",
    "supposition", "relationship", "relationships", "object", "objects", "ddna", "field", "fields",
    "learning", "learn", "reasoning", "thought", "input", "interaction", "response",
}


def _norm_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t]


def _meaningful(tokens: Iterable[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if len(tok) < 3 and tok not in {"mi", "mb"}:
            continue
        if tok in STOP_WORDS:
            continue
        if tok not in out:
            out.append(tok)
    return out[:24]


def _extract_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return ""


class ConversationSceneNeuron(BaseNeuron):
    """
    Rolling conversation.scene drawer.

    Conversation is treated as a verbal scene: it stays in RAM/KV, guides
    continuity, and is not dumped to durable memory by default.  It summarizes
    active topic, recent claims, active objects/relationships, and unresolved
    questions so response selection can continue the current thread instead of
    falling back to random-safe chatter.
    """

    DEFAULT_TTL_S = 20 * 60.0
    DEFAULT_MAX_TURNS = 24

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if self._is_control_plane(event):
            return []

        now = time.time()
        scene = await self._load_scene(ctx, now)
        changed = False
        outputs: List[Event] = []

        if event.topic in {"percept/text", "act/speech"}:
            text = _extract_text(event.payload)
            if not text or text.lstrip().startswith("/"):
                return []
            role = "assistant" if event.topic == "act/speech" else "user"
            if isinstance(event.payload, Mapping):
                role = str(event.payload.get("source", role) or role)
                if event.topic == "act/speech":
                    role = "assistant"
            self._add_turn(scene, text=text, role=role, event=event, now=now)
            changed = True

        elif event.topic == "thought/internal":
            payload = event.payload if isinstance(event.payload, Mapping) else {"text": event.payload}
            text = str(payload.get("text", "") or "").strip()
            kind = str(payload.get("kind", "") or payload.get("type", "") or "thought")
            if text:
                self._add_thought(scene, text=text, kind=kind, now=now)
                changed = True

        elif event.topic == "question/unresolved":
            q_text = ""
            if isinstance(event.payload, Mapping):
                mods = event.payload.get("modalities", {}) if isinstance(event.payload.get("modalities", {}), Mapping) else {}
                q_mod = mods.get("question", {}) if isinstance(mods.get("question", {}), Mapping) else {}
                q_text = str(q_mod.get("text", "") or event.payload.get("question", "") or "").strip()
            if q_text:
                self._add_unresolved(scene, q_text, now=now)
                changed = True

        if not changed:
            return []

        self._recompute(scene, now=now)
        await ctx.set_kv("conversation:scene", scene)
        await ctx.set_kv("conversation:current_scene", scene)
        await ctx.set_kv("conversation:summary", scene.get("summary", {}))

        outputs.append(
            Event(
                topic="conversation/scene",
                payload=scene,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "conversation.scene",
                    "channel": "thought",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                    "cognitive_visible": False,
                },
            )
        )
        return outputs

    async def _load_scene(self, ctx, now: float) -> Dict[str, Any]:
        scene = await ctx.get_kv("conversation:scene", {})
        if not isinstance(scene, dict):
            scene = {}
        last_ts = float(scene.get("updated_at", 0.0) or 0.0)
        ttl_s = float(await ctx.get_kv("conversation:scene_ttl_s", self.DEFAULT_TTL_S) or self.DEFAULT_TTL_S)
        if not scene or (last_ts and now - last_ts > ttl_s):
            scene = {
                "schema_ver": "conversation.scene.v1",
                "kind": "scene.object",
                "subkind": "conversation.scene",
                "created_at": now,
                "updated_at": now,
                "participants": [],
                "turns": [],
                "thoughts": [],
                "unresolved_questions": [],
                "active_objects": [],
                "active_relationships": [],
                "active_threads": [],
                "recent_claims": [],
                "recent_user_points": [],
                "recent_assistant_points": [],
                "summary": {},
                "state": {"ephemeral": True, "durable_memory": False, "turn_count": 0},
            }
        return scene

    def _add_turn(self, scene: Dict[str, Any], *, text: str, role: str, event: Event, now: float) -> None:
        max_turns = int(scene.get("max_turns", self.DEFAULT_MAX_TURNS) or self.DEFAULT_MAX_TURNS)
        role = "user" if role not in {"assistant", "system", "user"} else role
        turn = {
            "ts": now,
            "role": role,
            "text": text[:700],
            "tokens": _meaningful(_norm_tokens(text)),
            "correlation_id": event.correlation_id,
        }
        turns = list(scene.get("turns", []) or [])
        turns.append(turn)
        scene["turns"] = turns[-max_turns:]
        participants = list(scene.get("participants", []) or [])
        if role not in participants:
            participants.append(role)
        scene["participants"] = participants[-8:]
        if role == "user":
            points = list(scene.get("recent_user_points", []) or [])
            points.append(text[:220])
            scene["recent_user_points"] = points[-8:]
            if self._looks_claim(text):
                claims = list(scene.get("recent_claims", []) or [])
                claims.append(text[:240])
                scene["recent_claims"] = claims[-10:]
        elif role == "assistant":
            points = list(scene.get("recent_assistant_points", []) or [])
            points.append(text[:220])
            scene["recent_assistant_points"] = points[-8:]

    def _add_thought(self, scene: Dict[str, Any], *, text: str, kind: str, now: float) -> None:
        thoughts = list(scene.get("thoughts", []) or [])
        thoughts.append({"ts": now, "kind": kind, "text": text[:260], "tokens": _meaningful(_norm_tokens(text))})
        scene["thoughts"] = thoughts[-16:]

    def _add_unresolved(self, scene: Dict[str, Any], question: str, now: float) -> None:
        existing = list(scene.get("unresolved_questions", []) or [])
        norm = " ".join(_norm_tokens(question))
        existing = [q for q in existing if isinstance(q, Mapping) and q.get("norm") != norm]
        existing.append({"ts": now, "text": question[:260], "norm": norm})
        scene["unresolved_questions"] = existing[-12:]

    def _recompute(self, scene: Dict[str, Any], *, now: float) -> None:
        turns = list(scene.get("turns", []) or [])
        thoughts = list(scene.get("thoughts", []) or [])
        token_counter: Counter[str] = Counter()
        for item in turns[-16:] + thoughts[-8:]:
            if not isinstance(item, Mapping):
                continue
            for tok in list(item.get("tokens", []) or []):
                token_counter[str(tok)] += 1
        for marker in THREAD_MARKERS:
            if marker in token_counter:
                token_counter[marker] += 2
        active = [tok for tok, _ in token_counter.most_common(16)]
        threads = [tok for tok in active if tok in THREAD_MARKERS][:8]
        if not threads:
            threads = active[:6]

        relationships: List[str] = []
        active_set = set(active)
        relation_pairs = [
            ("reality", "prediction"), ("prediction", "pressure"), ("pressure", "question"),
            ("conversation", "scene"), ("scene", "expectation"), ("need", "pressure"),
            ("ddna", "field"), ("input", "interaction"), ("memory", "relationship"),
            ("learning", "relationship"),
        ]
        for left, right in relation_pairs:
            if left in active_set and right in active_set:
                relationships.append(f"{left}->{right}")

        topic = ", ".join(threads[:4]) if threads else (", ".join(active[:4]) if active else "conversation")
        last_user = ""
        last_assistant = ""
        for turn in reversed(turns):
            if not isinstance(turn, Mapping):
                continue
            if not last_user and turn.get("role") == "user":
                last_user = str(turn.get("text", "") or "")[:220]
            if not last_assistant and turn.get("role") == "assistant":
                last_assistant = str(turn.get("text", "") or "")[:220]
            if last_user and last_assistant:
                break

        recent_claims = [str(x) for x in list(scene.get("recent_claims", []) or []) if str(x or "").strip()]
        recent_user_points = [str(x) for x in list(scene.get("recent_user_points", []) or []) if str(x or "").strip()]
        recent_assistant_points = [str(x) for x in list(scene.get("recent_assistant_points", []) or []) if str(x or "").strip()]
        last_claim = recent_claims[-1] if recent_claims else ""

        scene["updated_at"] = now
        scene["active_objects"] = active
        scene["active_threads"] = threads
        scene["active_relationships"] = relationships[:12]
        scene["summary"] = {
            "topic": topic,
            "active_threads": threads[:8],
            "active_objects": active[:12],
            "active_relationships": relationships[:8],
            "last_user_point": last_user,
            "last_assistant_point": last_assistant,
            "last_claim": last_claim,
            "recent_user_points": recent_user_points[-8:],
            "recent_assistant_points": recent_assistant_points[-8:],
            "recent_claims": recent_claims[-8:],
            "unresolved_count": len(scene.get("unresolved_questions", []) or []),
            "turn_count": len(turns),
            "continuity_hint": self._continuity_hint(topic, threads, relationships),
        }
        state = dict(scene.get("state", {}) or {})
        state.update({"ephemeral": True, "durable_memory": False, "turn_count": len(turns)})
        scene["state"] = state

    def _continuity_hint(self, topic: str, threads: List[str], relationships: List[str]) -> str:
        if relationships:
            return f"Continue the verbal scene around {topic}; preserve relationship links: {', '.join(relationships[:3])}."
        if threads:
            return f"Continue the verbal scene around {topic}; avoid unrelated fallback replies."
        return "Continue the current verbal scene; ask for clarification rather than drifting."

    def _looks_claim(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered or lowered.endswith("?"):
            return False
        return any(word in lowered for word in (" is ", " are ", " means ", " becomes ", " creates ", " allows ", " should ", " = ", "->"))

    def _is_control_plane(self, event: Event) -> bool:
        if str(event.topic or "").startswith(("ui/", "control/")):
            return True
        meta = dict(event.meta or {})
        if meta.get("control") is True:
            return True
        if event.topic == "thought/internal" and meta.get("cognitive_visible") is False:
            return True
        return False


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/text", "act/speech", "thought/internal", "question/unresolved"],
        output_topics=["conversation/scene"],
        priority=3,
        cooldown_sec=0.0,
    )
    return [ConversationSceneNeuron(cfg)]
