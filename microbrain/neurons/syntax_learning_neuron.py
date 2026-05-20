from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem

CONTROL_WORDS = {"IF", "USER", "THEN", "ELSE", "CLASSIFY", "REPLY", "NOT", "AND"}
DDNA_TARGETS = {
    "warmth",
    "friendly",
    "gentle",
    "playful",
    "curious",
    "supportive",
    "direct",
    "terse",
}

IF_USER_SAYS_RE = re.compile(r"^IF\s+USER\s+says\s+(.+?)\s+THEN\s+(.+)$")
AND_SPLIT_RE = re.compile(r"\s+AND\s+")
ELSE_SPLIT_RE = re.compile(r"\s+ELSE\s+")
TOKEN_RE = re.compile(r"[a-z0-9_']+")


def _norm_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower())).strip()


def _strip_outer_quotes(text: str) -> str:
    text = str(text or "").strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        return text[1:-1].strip()
    return text


def _classifier_list(raw: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        name = _norm_text(part).replace(" ", "_")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


class SyntaxLearningNeuron(BaseNeuron):
    """
    Parses structured /r teaching notes into connected memory objects.

    This organ only listens to control/reinforce. It does not treat the quoted
    teaching text as normal user speech, and it writes rule-like memory with
    classifier links and DDNA/personality targets.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        if event.topic != "control/reinforce" or not isinstance(event.payload, dict):
            return []

        note = str(event.payload.get("teaching_note", "") or "").strip()
        if not note:
            return []

        parsed = self._parse_teaching_note(note)
        if not parsed:
            return [
                self._status(
                    "Teaching note ignored: expected all-caps syntax like IF USER says moin THEN CLASSIFY social_greeting AND REPLY good morning.",
                    event,
                )
            ]

        target = event.payload.get("target") if isinstance(event.payload.get("target"), dict) else {}
        target_text = str(target.get("text", "") or "").strip()
        try:
            weight = int(event.payload.get("weight", 0) or 0)
        except Exception:
            weight = 0
        weight = max(-5, min(5, weight))

        parsed["reinforce_weight"] = weight
        parsed["target_text"] = target_text
        parsed["target_role"] = str(event.payload.get("target_role", "") or "")
        parsed["target_hrm_idx"] = target.get("hrm_idx")
        parsed["nonce"] = str(event.payload.get("nonce", "") or "")
        parsed["teaching_note"] = note
        parsed["ts"] = time.time()

        store = await self._mem_cell_store(ctx)
        if store is None:
            return [self._status("Teaching note parsed, but mem-cell store is unavailable.", event)]

        try:
            saved = self._store_rule(store, parsed)
            await ctx.set_kv("syntax:last_rule", parsed)
            return [
                self._status(
                    f"Learned rule: IF USER says {parsed['condition_text']} THEN "
                    f"CLASSIFY {', '.join(parsed['classifiers']) or '(none)'}"
                    + (f" AND REPLY {parsed['reply_text']}" if parsed.get("reply_text") else "")
                    + (f" AND NOT REPLY {', '.join(parsed['avoid_replies'])}" if parsed.get("avoid_replies") else "")
                    + f". saved={saved}",
                    event,
                )
            ]
        except Exception as exc:
            await ctx.log_warn(f"[{self.name}] failed to store syntax rule", error=repr(exc))
            return [self._status(f"Teaching note parsed, but storing failed: {exc!r}", event)]

    async def _mem_cell_store(self, ctx) -> Optional[MemCellStore]:
        raw = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(raw, MemCellStore):
            return raw
        try:
            memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
            store = MemCellStore(memdir)
            await ctx.set_kv("memory:mem_cell_store", store)
            return store
        except Exception:
            return None

    def _parse_teaching_note(self, note: str) -> Dict[str, Any]:
        # Require all grammar words that are present to be ALL CAPS. This keeps
        # normal conversational language from accidentally becoming syntax.
        for word in CONTROL_WORDS:
            if re.search(rf"\b{word.lower()}\b", note):
                return {}

        m = IF_USER_SAYS_RE.match(note.strip())
        if not m:
            return {}

        condition_text = _strip_outer_quotes(m.group(1).strip())
        action_text = m.group(2).strip()
        then_text, *else_parts = ELSE_SPLIT_RE.split(action_text, maxsplit=1)
        else_text = else_parts[0].strip() if else_parts else ""

        classifiers: List[str] = []
        reply_text = ""
        avoid_replies: List[str] = []

        for action in AND_SPLIT_RE.split(then_text):
            action = action.strip()
            if not action:
                continue
            if action.startswith("CLASSIFY "):
                classifiers.extend(_classifier_list(action[len("CLASSIFY "):]))
            elif action.startswith("NOT REPLY "):
                avoid = _strip_outer_quotes(action[len("NOT REPLY "):].strip())
                if avoid:
                    avoid_replies.append(avoid)
            elif action.startswith("REPLY "):
                reply_text = _strip_outer_quotes(action[len("REPLY "):].strip())

        # de-dupe after multiple CLASSIFY clauses
        classifiers = list(dict.fromkeys(classifiers))
        ddna_targets = [name for name in classifiers if name in DDNA_TARGETS]
        concept_classifiers = [name for name in classifiers if name not in DDNA_TARGETS]

        return {
            "condition_actor": "USER",
            "condition_operator": "says",
            "condition_text": condition_text,
            "condition_norm": _norm_text(condition_text),
            "classifiers": classifiers,
            "concept_classifiers": concept_classifiers,
            "ddna_targets": ddna_targets,
            "reply_text": reply_text,
            "avoid_replies": avoid_replies,
            "else_text": else_text,
        }

    def _store_rule(self, store: MemCellStore, rule: Dict[str, Any]) -> int:
        condition = str(rule.get("condition_text", "") or "").strip()
        if not condition:
            return 0

        weight = int(rule.get("reinforce_weight", 0) or 0)
        ddna_strength = max(1, abs(weight))
        ddna_map = {name: ddna_strength for name in list(rule.get("ddna_targets", []) or [])}

        saved = 0
        reply_text = str(rule.get("reply_text", "") or "").strip()
        if reply_text:
            store.ingest_trainer_alignment(
                desired_text=reply_text,
                context_query=condition,
                bad_utterance=str(rule.get("target_text", "") or "") if weight < 0 else "",
                need="interaction",
                style="direct_simple",
                source="syntax_learning",
                meta={
                    "syntax_rule": True,
                    "syntax_classifiers": list(rule.get("classifiers", []) or []),
                    "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
                    "ddna_targets": ddna_map,
                    "condition_text": condition,
                    "reinforce_weight": weight,
                },
                tier="learned",
            )
            saved += 1

        digest = hashlib.blake2b(
            jsonish(rule).encode("utf-8", errors="ignore"),
            digest_size=8,
        ).hexdigest()
        now_ts = time.time()
        refs: List[Dict[str, Any]] = [
            {"kind": "condition", "value": condition},
        ]
        for c in list(rule.get("concept_classifiers", []) or []):
            refs.append({"kind": "classifier", "value": c, "target_type": "concept"})
        for d in list(rule.get("ddna_targets", []) or []):
            refs.append({"kind": "classifier", "value": d, "target_type": "ddna", "strength": ddna_strength})
        if reply_text:
            refs.append({"kind": "reply", "value": reply_text})
        for avoid in list(rule.get("avoid_replies", []) or []):
            refs.append({"kind": "avoid_reply", "value": avoid})

        cell = {
            "id": f"sr{digest}",
            "kind": "syntax_rule",
            "tier": "learned",
            "anchor": {"kind": "syntax/condition", "ref": condition[:200], "norm": _norm_text(condition)[:200]},
            "refs": refs,
            "modalities": ["text", "teaching"],
            "links_explicit": [],
            "activation": 1.0,
            "promotion": 0.52,
            "decay": 1.0,
            "trust": 0.94 if weight >= 0 else 0.80,
            "meta": {
                "role": "system",
                "kind": "syntax_rule",
                "condition_text": condition,
                "condition_norm": _norm_text(condition),
                "syntax_classifiers": list(rule.get("classifiers", []) or []),
                "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
                "ddna_targets": ddna_map,
                "reply_text": reply_text,
                "avoid_replies": list(rule.get("avoid_replies", []) or []),
                "reinforce_weight": weight,
                "target_text": str(rule.get("target_text", "") or ""),
            },
            "ts": now_ts,
            "last_seen": now_ts,
            "encounter_count": 1,
            "revision": 0,
        }
        store.upsert_cell(cell, tier="learned")
        saved += 1
        return saved

    def _status(self, text: str, event: Event) -> Event:
        return Event(
            topic="act/speech",
            payload={"text": text, "style": "system", "channel": "default"},
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"control": True, "kind": "syntax_learning_status", "store_in_memory": False},
        )


def jsonish(value: Any) -> str:
    import json

    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(value)


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["control/reinforce"],
        output_topics=["act/speech"],
        priority=-4,
    )
    yield SyntaxLearningNeuron(cfg)
