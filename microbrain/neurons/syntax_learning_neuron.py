from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem

CONTROL_WORDS = {"IF", "USER", "THEN", "ELSE", "CLASSIFY", "REPLY", "NOT", "AND", "SUPPRESS"}
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

IF_RULE_RE = re.compile(r"^IF\s+(.+?)\s+THEN\s+(.+)$")
IF_USER_SAYS_CONDITION_RE = re.compile(r"^USER\s+says\s+(.+)$")
CONDITION_OPERATOR_RE = re.compile(
    r"^(.+?)\s+"
    r"(differs\s+from|plus\s+time|is|exists|detected|changes|restored|resolved|unresolved|"
    r"persists|increases|decreases|occurs|begins|completes|fails|succeeds|helps|harms)"
    r"(?:\s+(.+))?$",
    re.IGNORECASE,
)
AND_SPLIT_RE = re.compile(r"\s+AND\s+")
ELSE_SPLIT_RE = re.compile(r"\s+ELSE\s+")
TOKEN_RE = re.compile(r"[a-z0-9_']+")
SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _norm_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower())).strip()


def _strip_outer_quotes(text: str) -> str:
    text = str(text or "").strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        return text[1:-1].strip()
    return text


def strip_slearn_inline_comment(text: str) -> str:
    """Strip a trailing # comment while preserving # characters inside quotes."""
    raw = str(text or "")
    out: List[str] = []
    quote = ""
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in {"\"", "'"}:
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _mask_quoted_text(text: str) -> str:
    """Mask quoted data before checking whether control rails are lowercase."""
    raw = str(text or "")
    out: List[str] = []
    quote = ""
    escaped = False
    for ch in raw:
        if escaped:
            out.append(" ")
            escaped = False
            continue
        if ch == "\\" and quote:
            out.append(" ")
            escaped = True
            continue
        if quote:
            out.append(" ")
            if ch == quote:
                quote = ""
            continue
        if ch in {"\"", "'"}:
            quote = ch
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


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


def _template_slots(text: str) -> List[str]:
    """Return ordered, unique {slot} names used by a SLEARN template."""
    out: List[str] = []
    seen: set[str] = set()
    for match in SLOT_RE.finditer(str(text or "")):
        name = str(match.group(1) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


class SyntaxLearningNeuron(BaseNeuron):
    """
    Parses structured /r teaching notes into connected memory objects.

    This organ listens to manual reinforcement (control/reinforce) and
    document curriculum ingestion (control/slearn). It does not treat the
    quoted teaching text as normal user speech, and it writes rule-like memory
    with classifier links and DDNA/personality targets.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        if event.topic not in {"control/reinforce", "control/slearn"} or not isinstance(event.payload, dict):
            return []

        note = str(event.payload.get("teaching_note", "") or event.payload.get("rule", "") or "").strip()
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
        parsed["source_mode"] = "slearn" if event.topic == "control/slearn" else "reinforcement"
        parsed["source_name"] = str(event.payload.get("source_name", "") or "")
        parsed["source_path"] = str(event.payload.get("source_path", "") or "")
        parsed["source_line"] = event.payload.get("source_line")
        parsed["ts"] = time.time()

        store = await self._mem_cell_store(ctx)
        if store is None:
            return [self._status("Teaching note parsed, but mem-cell store is unavailable.", event)]

        try:
            saved = self._store_rule(store, parsed)
            await ctx.set_kv("syntax:last_rule", parsed)
            if parsed.get("source_mode") == "slearn":
                await self._record_slearn_apply(ctx, parsed, saved)
            return [
                self._status(
                    f"Learned {parsed.get('rule_kind', 'rule')}: IF {parsed.get('condition_raw', parsed['condition_text'])} THEN "
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

    async def _record_slearn_apply(self, ctx, rule: Dict[str, Any], saved: int) -> None:
        total = int(await ctx.get_kv("slearn:rules_applied_total", 0) or 0) + max(1, int(saved or 0))
        await ctx.set_kv("slearn:rules_applied_total", total)
        last = {
            "ts": time.time(),
            "saved": saved,
            "rules_applied_total": total,
            "rule_kind": str(rule.get("rule_kind", "") or ""),
            "condition_text": str(rule.get("condition_text", "") or ""),
            "source_name": str(rule.get("source_name", "") or ""),
            "source_path": str(rule.get("source_path", "") or ""),
            "source_line": rule.get("source_line"),
        }
        await ctx.set_kv("slearn:last_applied_rule", last)
        try:
            memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
            path = Path(memdir) / "slearn" / "slearn_audit.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "rule_applied", **last}, ensure_ascii=False, sort_keys=True) + "\n")
            await ctx.set_kv("slearn:audit_path", str(path))
        except Exception:
            # Audit visibility is helpful, but learning should not fail if the
            # audit file is locked or the memdir is temporarily unavailable.
            pass

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
        # Comments and quoted domain text are data, not control rails.  Mask them
        # before the lowercase-rail check so a gloss containing "and" or a term
        # such as "rock and roll" cannot invalidate an otherwise safe CAPS rule.
        note = strip_slearn_inline_comment(note).strip()
        rail_scan = _mask_quoted_text(note)
        for word in CONTROL_WORDS:
            if re.search(rf"\b{word.lower()}\b", rail_scan):
                return {}

        m = IF_RULE_RE.match(note)
        if not m:
            return {}

        condition_raw = m.group(1).strip()
        action_text = m.group(2).strip()
        then_text, *else_parts = ELSE_SPLIT_RE.split(action_text, maxsplit=1)
        else_text = else_parts[0].strip() if else_parts else ""

        condition = self._parse_condition(condition_raw)
        classifiers: List[str] = []
        reply_text = ""
        avoid_replies: List[str] = []
        suppress_targets: List[str] = []

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
            elif action.startswith("SUPPRESS "):
                suppress_targets.extend(_classifier_list(action[len("SUPPRESS "):]))

        # de-dupe after multiple CLASSIFY / SUPPRESS clauses
        classifiers = list(dict.fromkeys(classifiers))
        suppress_targets = list(dict.fromkeys(suppress_targets))
        ddna_targets = [name for name in classifiers if name in DDNA_TARGETS]
        concept_classifiers = [name for name in classifiers if name not in DDNA_TARGETS]
        rule_kind = self._rule_kind_for(condition)
        condition_text_for_slots = str(condition.get("condition_text", "") or "")
        condition_slots = _template_slots(condition_text_for_slots)
        reply_slots = _template_slots(reply_text)

        # Refuse an unanchored catch-all such as USER says "{payload}". A learned
        # speech template must contain at least one literal word/number so it can
        # compete as a specific language rule instead of swallowing every input.
        if condition_slots and not re.search(r"[A-Za-z0-9]", SLOT_RE.sub("", condition_text_for_slots)):
            return {}

        # A templated reply may only use values captured by the condition.
        # This keeps SLEARN declarative: the sheet can route bound input data,
        # but it cannot conjure unbound placeholders at runtime.
        if any(slot not in condition_slots for slot in reply_slots):
            return {}

        return {
            **condition,
            "rule_kind": rule_kind,
            "condition_slots": condition_slots,
            "reply_slots": reply_slots,
            "is_template_rule": bool(condition_slots),
            "classifiers": classifiers,
            "concept_classifiers": concept_classifiers,
            "ddna_targets": ddna_targets,
            "reply_text": reply_text,
            "avoid_replies": avoid_replies,
            "suppress_targets": suppress_targets,
            "else_text": else_text,
        }

    def _parse_condition(self, condition_raw: str) -> Dict[str, Any]:
        condition_raw = str(condition_raw or "").strip()

        user_match = IF_USER_SAYS_CONDITION_RE.match(condition_raw)
        if user_match:
            condition_text = _strip_outer_quotes(user_match.group(1).strip())
            return {
                "condition_actor": "USER",
                "condition_operator": "says",
                "condition_domain": "user_speech",
                "condition_text": condition_text,
                "condition_norm": _norm_text(condition_text),
                "condition_raw": condition_raw,
            }

        op_match = CONDITION_OPERATOR_RE.match(condition_raw)
        if op_match:
            subject = _strip_outer_quotes(op_match.group(1).strip())
            operator = _norm_text(op_match.group(2).strip()).replace(" ", "_")
            obj = _strip_outer_quotes((op_match.group(3) or "").strip())
            condition_text = " ".join(part for part in (subject, operator.replace("_", " "), obj) if part).strip()
            subject_norm = _norm_text(subject).replace(" ", "_")
            return {
                "condition_actor": subject_norm.upper() if subject_norm else "SYSTEM",
                "condition_operator": operator,
                "condition_domain": subject_norm or "system",
                "condition_subject": subject,
                "condition_object": obj,
                "condition_text": condition_text,
                "condition_norm": _norm_text(condition_text),
                "condition_raw": condition_raw,
            }

        # Generic CAPS curriculum condition. This keeps the parser extensible for
        # object/scene/reasoning sheets without pretending every domain already
        # has a dedicated organ-level command.
        domain = _norm_text(condition_raw).split(" ", 1)[0] if _norm_text(condition_raw) else "system"
        return {
            "condition_actor": domain.upper(),
            "condition_operator": "state",
            "condition_domain": domain,
            "condition_subject": condition_raw,
            "condition_object": "",
            "condition_text": condition_raw,
            "condition_norm": _norm_text(condition_raw),
            "condition_raw": condition_raw,
        }

    def _rule_kind_for(self, condition: Dict[str, Any]) -> str:
        if condition.get("condition_actor") == "USER" and condition.get("condition_operator") == "says":
            return "syntax_rule"
        domain = str(condition.get("condition_domain", "") or "").lower()
        if domain in {"power", "maintenance", "boredom", "social", "safety", "curiosity", "need", "uplift", "rest"}:
            return "drive_rule"
        if domain in {"object", "entity", "state", "action", "relationship", "scene", "visual", "auditory", "touch", "feedback"}:
            return "object_rule"
        if domain in {"expectation", "supposition", "question", "gap", "memory", "thought", "reasoning", "abstraction"}:
            return "reasoning_rule"
        return "curriculum_rule"

    def apply_slearn_batch(
        self,
        store: MemCellStore,
        items: Sequence[Dict[str, Any]],
        *,
        weight: int = 3,
    ) -> Dict[str, Any]:
        """Parse/store a SLEARN batch without emitting one event per rule.

        The read sidecar runs this method in a worker thread.  Pure classifier
        rules are built as deterministic cells and staged in one composer file;
        reply-bearing rules keep the richer trainer-alignment path but defer its
        flush until the end of the batch.
        """
        bounded_weight = max(1, min(5, int(weight or 3)))
        accepted = 0
        rejected = 0
        saved_cells = 0
        direct_cells: List[Dict[str, Any]] = []
        errors: List[str] = []

        for item in items or []:
            note = str(item.get("teaching_note", "") or item.get("rule", "") or "").strip()
            if not note:
                rejected += 1
                continue
            parsed = self._parse_teaching_note(note)
            if not parsed:
                rejected += 1
                continue

            parsed["reinforce_weight"] = bounded_weight
            parsed["target_text"] = str(item.get("target_text", "") or "")
            parsed["target_role"] = str(item.get("target_role", "") or "")
            parsed["target_hrm_idx"] = item.get("target_hrm_idx")
            parsed["nonce"] = str(item.get("nonce", "") or "")
            parsed["teaching_note"] = note
            parsed["source_mode"] = "slearn"
            parsed["source_name"] = str(item.get("source_name", "") or "")
            parsed["source_path"] = str(item.get("source_path", "") or "")
            parsed["source_line"] = item.get("source_line")
            parsed["ts"] = time.time()

            try:
                if str(parsed.get("reply_text", "") or "").strip():
                    saved_cells += self._store_rule(store, parsed, flush=False)
                else:
                    direct_cells.append(self._build_rule_cell(parsed))
                    saved_cells += 1
                accepted += 1
            except Exception as exc:
                rejected += 1
                errors.append(repr(exc))

        if direct_cells:
            store.stage_cells(direct_cells, tier="learned", touch=True)
        if store.dirty_count("learned"):
            store.flush_tier("learned")

        return {
            "accepted": accepted,
            "rejected": rejected,
            "saved_cells": saved_cells,
            "staged_paths": store.take_staged_paths("learned"),
            "errors": errors[:8],
        }

    def _store_rule(self, store: MemCellStore, rule: Dict[str, Any], *, flush: bool = True) -> int:
        condition = str(rule.get("condition_text", "") or "").strip()
        if not condition:
            return 0

        weight = int(rule.get("reinforce_weight", 0) or 0)
        ddna_strength = max(1, abs(weight))
        ddna_map = {name: ddna_strength for name in list(rule.get("ddna_targets", []) or [])}

        saved = 0
        reply_text = str(rule.get("reply_text", "") or "").strip()
        is_template_rule = bool(rule.get("is_template_rule", False) or rule.get("condition_slots"))
        if reply_text and not is_template_rule:
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
                    "source_mode": str(rule.get("source_mode", "reinforcement") or "reinforcement"),
                    "source_name": str(rule.get("source_name", "") or ""),
                    "source_path": str(rule.get("source_path", "") or ""),
                    "source_line": rule.get("source_line"),
                },
                tier="learned",
                flush=flush,
            )
            saved += 1

        store.upsert_cell(self._build_rule_cell(rule), tier="learned", flush=flush)
        saved += 1
        return saved

    def _build_rule_cell(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        condition = str(rule.get("condition_text", "") or "").strip()
        weight = int(rule.get("reinforce_weight", 0) or 0)
        ddna_strength = max(1, abs(weight))
        ddna_map = {name: ddna_strength for name in list(rule.get("ddna_targets", []) or [])}
        reply_text = str(rule.get("reply_text", "") or "").strip()

        # Rule identity is semantic and restart-stable.  Volatile fields such as
        # timestamps, source line numbers, and job nonces must not mint a new
        # memory cell when a resumable SLEARN job sees the same rule again.
        identity = {
            "rule_kind": str(rule.get("rule_kind", "") or ""),
            "condition_raw": str(rule.get("condition_raw", condition) or condition),
            "classifiers": list(rule.get("classifiers", []) or []),
            "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
            "ddna_targets": list(rule.get("ddna_targets", []) or []),
            "reply_text": reply_text,
            "condition_slots": list(rule.get("condition_slots", []) or []),
            "reply_slots": list(rule.get("reply_slots", []) or []),
            "avoid_replies": list(rule.get("avoid_replies", []) or []),
            "suppress_targets": list(rule.get("suppress_targets", []) or []),
            "reinforce_weight": weight,
        }
        digest = hashlib.blake2b(
            jsonish(identity).encode("utf-8", errors="ignore"),
            digest_size=8,
        ).hexdigest()
        now_ts = time.time()
        refs: List[Dict[str, Any]] = [
            {"kind": "condition", "value": condition},
            {"kind": "condition_domain", "value": str(rule.get("condition_domain", "") or "")},
            {"kind": "condition_operator", "value": str(rule.get("condition_operator", "") or "")},
        ]
        for c in list(rule.get("concept_classifiers", []) or []):
            refs.append({"kind": "classifier", "value": c, "target_type": "concept"})
        for d in list(rule.get("ddna_targets", []) or []):
            refs.append({"kind": "classifier", "value": d, "target_type": "ddna", "strength": ddna_strength})
        if reply_text:
            refs.append({"kind": "reply", "value": reply_text})
        for avoid in list(rule.get("avoid_replies", []) or []):
            refs.append({"kind": "avoid_reply", "value": avoid})
        for target in list(rule.get("suppress_targets", []) or []):
            refs.append({"kind": "suppress", "value": target})

        source_mode = str(rule.get("source_mode", "reinforcement") or "reinforcement")
        rule_kind = str(rule.get("rule_kind", "syntax_rule") or "syntax_rule")
        trust = 0.94 if weight >= 0 else 0.80
        if source_mode == "slearn":
            trust = 0.86 if weight >= 0 else 0.72

        return {
            "id": f"sr{digest}",
            "kind": rule_kind,
            "tier": "learned",
            "anchor": {"kind": f"slearn/{rule_kind}", "ref": condition[:200], "norm": _norm_text(condition)[:200]},
            "refs": refs,
            "modalities": ["text", "teaching"],
            "links_explicit": [],
            "activation": 1.0,
            "promotion": 0.52,
            "decay": 1.0,
            "trust": trust,
            "meta": {
                "role": "system",
                "kind": rule_kind,
                "condition_text": condition,
                "condition_raw": str(rule.get("condition_raw", "") or ""),
                "condition_actor": str(rule.get("condition_actor", "") or ""),
                "condition_operator": str(rule.get("condition_operator", "") or ""),
                "condition_domain": str(rule.get("condition_domain", "") or ""),
                "condition_subject": str(rule.get("condition_subject", "") or ""),
                "condition_object": str(rule.get("condition_object", "") or ""),
                "condition_norm": _norm_text(condition),
                "condition_slots": list(rule.get("condition_slots", []) or []),
                "reply_slots": list(rule.get("reply_slots", []) or []),
                "is_template_rule": bool(rule.get("is_template_rule", False) or rule.get("condition_slots")),
                "syntax_classifiers": list(rule.get("classifiers", []) or []),
                "concept_classifiers": list(rule.get("concept_classifiers", []) or []),
                "ddna_targets": ddna_map,
                "reply_text": reply_text,
                "avoid_replies": list(rule.get("avoid_replies", []) or []),
                "suppress_targets": list(rule.get("suppress_targets", []) or []),
                "reinforce_weight": weight,
                "source_decay_bias": 0.85 if source_mode == "slearn" else 1.0,
                "lived_experience_can_override": True,
                "target_text": str(rule.get("target_text", "") or ""),
                "source_mode": source_mode,
                "source_name": str(rule.get("source_name", "") or ""),
                "source_path": str(rule.get("source_path", "") or ""),
                "source_line": rule.get("source_line"),
            },
            "ts": now_ts,
            "last_seen": now_ts,
            "encounter_count": 1,
            "revision": 0,
        }

    def _status(self, text: str, event: Event) -> Event:
        return Event(
            topic="ui/status",
            payload={"text": text, "style": "system", "channel": "default"},
            source=self.name,
            correlation_id=event.correlation_id,
            meta={"control": True, "kind": "syntax_learning_status", "store_in_memory": False, "reinforcement_eligible": False, "self_output_track": False, "cognitive_visible": False},
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
        subscribed_topics=["control/reinforce", "control/slearn"],
        output_topics=["ui/status"],
        priority=-4,
    )
    yield SyntaxLearningNeuron(cfg)
