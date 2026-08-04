from __future__ import annotations

import re
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping

from microbrain.hormone import derive_rosehip_state
from microbrain.memory.cross_modal_answer import gather_support, compose_answer
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem
SLEARN_SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _strip_outer_quotes(text: str) -> str:
    raw = str(text or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
        return raw[1:-1].strip()
    return raw


def _flex_literal_regex(text: str) -> str:
    parts = re.split(r"(\s+)", str(text or ""))
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts if part)


def _match_slearn_template(pattern: str, text: str) -> Dict[str, str] | None:
    """Bind {slot} placeholders in a learned USER-speech condition."""
    pattern = str(pattern or "").strip()
    raw_text = str(text or "").strip()
    matches = list(SLEARN_SLOT_RE.finditer(pattern))
    if not pattern or not raw_text or not matches:
        return None

    # Do not allow a bare {anything} rule to become a catch-all speech rule.
    literal_text = SLEARN_SLOT_RE.sub("", pattern)
    if not re.search(r"[A-Za-z0-9]", literal_text):
        return None

    parts: List[str] = [r"^\s*"]
    seen: set[str] = set()
    pos = 0
    for match in matches:
        parts.append(_flex_literal_regex(pattern[pos:match.start()]))
        name = str(match.group(1) or "").strip()
        if name in seen:
            parts.append(rf"(?P={name})")
        else:
            parts.append(rf"(?P<{name}>.+?)")
            seen.add(name)
        pos = match.end()
    parts.append(_flex_literal_regex(pattern[pos:]))
    parts.append(r"\s*$")

    try:
        matched = re.match("".join(parts), raw_text, flags=re.IGNORECASE | re.DOTALL)
    except re.error:
        return None
    if not matched:
        return None

    bindings: Dict[str, str] = {}
    for name, value in matched.groupdict().items():
        clean = _strip_outer_quotes(str(value or "").strip())
        if not clean:
            return None
        bindings[name] = clean
    return bindings


def _render_slearn_template(template: str, bindings: Mapping[str, str]) -> str:
    raw = str(template or "").strip()
    if not raw:
        return ""

    unresolved = False

    def replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = str(match.group(1) or "")
        if name not in bindings:
            unresolved = True
            return ""
        return str(bindings.get(name, "") or "")

    rendered = SLEARN_SLOT_RE.sub(replace, raw).strip()
    return "" if unresolved else rendered


def _looks_stock_reply(text: str) -> bool:
    """Return True for low-value canned replies that should not be reused as memory."""
    norm = _norm(text)
    if not norm:
        return False
    stock_exact = {
        "hey there whats up",
        "hey whats up",
        "whats up",
        "hello im here and listening",
        "hello im here",
        "im here and listening",
        "need one variable",
        "need a target",
        "need one target",
        "give me a concrete goal question or choice and ill respond directly",
        "give me the concrete target or missing variable and ill answer directly",
    }
    return (
        norm in stock_exact
        or norm.startswith("hey there whats up")
        or norm.startswith("i heard your question")
        or norm.startswith("i heard you")
        or norm.startswith("the question is about")
        or norm.startswith("the open question is about")
    )


def _scene_turn_count(summary: Mapping[str, Any] | None) -> int:
    if not isinstance(summary, Mapping):
        return 0
    try:
        return int(summary.get("turn_count", 0) or 0)
    except Exception:
        return 0


def _conversation_active(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    return _scene_turn_count(summary) >= 2 or bool(summary.get("topic") or summary.get("active_threads"))


def _sentence_list(value: Any, limit: int = 6) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _accent_from_meta(raw_meta: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    """Return signed /acc metadata.

    + values are positive emphasis / preference pull.
    - values are correction severity / preference push-away.
    Magnitude still records how strong the user marked the tone.
    """
    try:
        value = float(raw_meta.get("accent_value", 0.0) or 0.0)
    except Exception:
        value = 0.0
    value = max(-10.0, min(10.0, value))

    try:
        magnitude = float(raw_meta.get("accent_magnitude", raw_meta.get("accent_intensity", abs(value))) or abs(value))
    except Exception:
        magnitude = abs(value)
    magnitude = max(0.0, min(10.0, magnitude))

    positive = max(0.0, value)
    negative_severity = abs(min(0.0, value))
    label = str(raw_meta.get("tone_label", "") or "").strip()
    return value, magnitude, positive, negative_severity, label


class NativeResponderNeuron(BaseNeuron):
    """
    Default non-LLM responder.

    This is deliberately small and deterministic. It exists so MB can answer on
    its own legs while higher cognition / atomization / hormone shaping are
    brought online.

    Hormone / DDNA state modulates:
      - whether a statement gets a full reply or a minimal acknowledgement
      - warmth / terseness
      - whether to clarify before overcommitting
      - whether to externalize a thought at all when the request is weak
    """

    def _meta_ddna_targets(self, meta: Dict[str, Any]) -> Dict[str, float]:
        raw = meta.get("ddna_targets", {}) if isinstance(meta, Mapping) else {}
        out: Dict[str, float] = {}
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                name = _norm(str(key or "")).replace(" ", "_")
                if not name:
                    continue
                out[name] = max(out.get(name, 0.0), abs(_safe_float(value, 1.0)))
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                name = _norm(str(item or "")).replace(" ", "_")
                if name:
                    out[name] = max(out.get(name, 0.0), 1.0)
        return out

    def _ddna_bonus(self, meta: Dict[str, Any], *, warm: float = 0.0) -> float:
        targets = self._meta_ddna_targets(meta)
        if not targets:
            return 0.0
        bonus = 0.0
        if "warmth" in targets:
            bonus += min(0.12, targets["warmth"] * (0.025 + 0.015 * max(0.0, warm)))
        if "friendly" in targets:
            bonus += min(0.12, targets["friendly"] * 0.035)
        if "supportive" in targets:
            bonus += min(0.08, targets["supportive"] * 0.025)
        if "gentle" in targets:
            bonus += min(0.06, targets["gentle"] * 0.02)
        return round(min(0.28, bonus), 4)

    def _syntax_guidance(self, mem_store: Any, lookup_text: str) -> Dict[str, Any]:
        guidance: Dict[str, Any] = {"preferred_replies": [], "avoid_replies": [], "ddna_targets": {}, "classifiers": []}
        if not isinstance(mem_store, MemCellStore) or not str(lookup_text or "").strip():
            return guidance
        try:
            hits = mem_store.search_text_cells(lookup_text, limit=16, tiers=("learned", "long", "hot", "now", "short"))
        except Exception:
            return guidance

        seen_reply: set[str] = set()
        seen_avoid: set[str] = set()
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            meta = dict(hit.get("meta", {}) or {})
            kind = str(hit.get("kind", "") or meta.get("kind", "") or "")
            if kind not in {"syntax_rule", "trainer_alignment"}:
                continue

            template_bindings: Dict[str, str] = {}
            condition_text = str(meta.get("condition_text", "") or "").strip()
            condition_slots = [str(v or "").strip() for v in list(meta.get("condition_slots", []) or []) if str(v or "").strip()]
            if kind == "syntax_rule" and not condition_slots and condition_text:
                condition_slots = [str(m.group(1) or "").strip() for m in SLEARN_SLOT_RE.finditer(condition_text) if str(m.group(1) or "").strip()]
            if kind == "syntax_rule" and condition_slots:
                matched = _match_slearn_template(condition_text, lookup_text)
                if matched is None:
                    continue
                template_bindings = matched

            score = _safe_float(hit.get("score", 0.0), 0.0)
            if template_bindings:
                # A full learned template match is stronger evidence than fuzzy
                # token overlap used only to retrieve the candidate rule.
                score += 0.45
                meta["template_bindings"] = dict(template_bindings)

            for key, value in self._meta_ddna_targets(meta).items():
                guidance["ddna_targets"][key] = max(float(guidance["ddna_targets"].get(key, 0.0)), value)
            for classifier in list(meta.get("syntax_classifiers", []) or []):
                name = _norm(str(classifier or "")).replace(" ", "_")
                if name and name not in guidance["classifiers"]:
                    guidance["classifiers"].append(name)
            reply = str(meta.get("reply_text", "") or meta.get("desired_utterance", "") or "").strip()
            if template_bindings and reply:
                reply = _render_slearn_template(reply, template_bindings)
            elif reply and SLEARN_SLOT_RE.search(reply):
                # Never speak an unresolved learned placeholder literally.
                reply = ""
            if reply and not _looks_stock_reply(reply):
                norm = _norm(reply)
                if norm and norm not in seen_reply:
                    seen_reply.add(norm)
                    guidance["preferred_replies"].append({"text": reply, "score": score, "meta": meta})
            for avoid in list(meta.get("avoid_replies", []) or []):
                avoid_text = str(avoid or "").strip()
                norm = _norm(avoid_text)
                if avoid_text and norm and norm not in seen_avoid:
                    seen_avoid.add(norm)
                    guidance["avoid_replies"].append(avoid_text)
            bad = str(meta.get("bad_utterance", "") or meta.get("trainer_bad_utterance", "") or "").strip()
            norm_bad = _norm(bad)
            if bad and norm_bad and norm_bad not in seen_avoid:
                seen_avoid.add(norm_bad)
                guidance["avoid_replies"].append(bad)
        return guidance

    def _preferred_rule_reply(self, guidance: Dict[str, Any], *, warm: float) -> str:
        avoid_norms = {_norm(text) for text in list(guidance.get("avoid_replies", []) or []) if _norm(str(text or ""))}
        best_text = ""
        best_score = 0.0
        for item in list(guidance.get("preferred_replies", []) or []):
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            norm = _norm(text)
            if not norm or norm in avoid_norms or _looks_stock_reply(text):
                continue
            meta = dict(item.get("meta", {}) or {})
            score = _safe_float(item.get("score", 0.0), 0.0) + self._ddna_bonus(meta, warm=warm)
            if score > best_score:
                best_text = text
                best_score = score
        return best_text if best_score >= 0.35 else ""

    def _context_breaks_scene(self, candidate: str, input_norm: str, conversation_summary: Mapping[str, Any]) -> bool:
        """Block generic memory/fallback phrases once a verbal scene is active."""
        if not _conversation_active(conversation_summary):
            return False
        cand_norm = _norm(candidate)
        if not cand_norm:
            return False
        current_is_greeting = input_norm in {"hi", "hello", "hey", "yo", "howdy"}
        if _looks_stock_reply(candidate) and not current_is_greeting:
            return True
        if cand_norm.startswith(("hey there", "hey whats", "hello im here")) and not current_is_greeting:
            return True
        return False

    def _latest_gap_question(self, gap: Mapping[str, Any], *, avoid_norm: str = "") -> str:
        if not isinstance(gap, Mapping):
            return ""
        status = str(gap.get("status", "") or "")
        if status in {"resolved", "closed", "answered"}:
            return ""
        anchor_norm = _norm(str(gap.get("anchor", "") or ""))
        if avoid_norm and anchor_norm == avoid_norm:
            return ""
        question = str(gap.get("question", "") or "").strip()
        return question if question and not _looks_stock_reply(question) else ""

    def _scene_recent(self, summary: Mapping[str, Any], key: str, limit: int = 6) -> List[str]:
        return _sentence_list(summary.get(key, []), limit=limit) if isinstance(summary, Mapping) else []

    def _last_scene_text(self, summary: Mapping[str, Any], key: str) -> str:
        value = summary.get(key, "") if isinstance(summary, Mapping) else ""
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            for item in reversed(value):
                text = str(item or "").strip()
                if text:
                    return text
        return ""

    def _infer_recent_referent(self, summary: Mapping[str, Any], *, prefer_questions: bool = False) -> str:
        if not isinstance(summary, Mapping):
            return ""
        texts: List[str] = []
        texts.extend(self._scene_recent(summary, "recent_assistant_points", limit=4))
        texts.extend(self._scene_recent(summary, "recent_user_points", limit=4))
        texts.extend(self._scene_recent(summary, "recent_claims", limit=4))
        blob = " ".join(texts).lower()
        if prefer_questions and re.search(r"\b(question|questions|inquiry|inquiries)\b", blob):
            return "questions"
        for word in ("questions", "question", "inquiry", "gap", "relationship", "scene", "memory", "pressure", "curiosity"):
            if re.search(rf"\b{re.escape(word)}\b", blob):
                return "questions" if word in {"question", "questions", "inquiry"} else word
        active_objects = _sentence_list(summary.get("active_objects", []), limit=8)
        for obj in active_objects:
            if obj not in {"like", "any", "what", "have", "question"}:
                return obj
        return ""

    async def _conversation_pragmatic_reply(
        self,
        ctx,
        *,
        text: str,
        norm: str,
        conversation_summary: Mapping[str, Any],
        conversation_scene: Mapping[str, Any],
        terse: float,
        warm: float,
    ) -> str:
        """Canned pragmatic replies were removed after line-live testing.

        Short follow-ups now remain inside the thought/context pipeline unless
        memory, learned syntax, or a reasoning backend builds a response.
        """
        return ""

    def _asks_internal_status(self, text: str, norm: str) -> bool:
        if not norm:
            return False
        exact = {
            "status",
            "how are you",
            "how are you doing",
            "how do you feel",
            "how are things",
            "are you okay",
            "are you ok",
            "what are your internal scores",
            "internal scores",
            "internal status",
            "system status",
        }
        if norm in exact:
            return True
        is_question = str(text or "").strip().endswith("?")
        strong_phrases = (
            "your internal state",
            "your internal status",
            "your internal scores",
            "your scores",
            "your status",
            "your state",
            "your needs",
            "your drives",
            "your power",
            "your battery",
            "do you need to charge",
            "battery level",
            "power level",
            "maintenance status",
        )
        if any(phrase in norm for phrase in strong_phrases):
            return True
        if is_question and "need to charge" in norm:
            return True
        if is_question and any(token in norm.split() for token in {"status", "state", "scores", "needs", "drives", "battery", "charging", "maintenance"}):
            return True
        return False

    def _status_band(self, value: float, *, low: float = 0.33, high: float = 0.66) -> str:
        if value >= high:
            return "high"
        if value >= low:
            return "moderate"
        return "low"

    async def _internal_status_reply(self, ctx, *, text: str, norm: str, terse: float) -> str:
        if not self._asks_internal_status(text, norm):
            return ""

        power_state = await ctx.get_kv("power:state", {}) or {}
        if not isinstance(power_state, Mapping):
            power_state = {}
        power_vector = await ctx.get_kv("drive:power_vector", {}) or {}
        if not isinstance(power_vector, Mapping):
            power_vector = {}
        pressure = power_vector.get("pressure", {}) if isinstance(power_vector.get("pressure", {}), Mapping) else {}
        needs = await ctx.get_kv("drive:needs_stack", {}) or {}
        hormones = await ctx.get_kv("drive:hormones", {}) or {}
        boredom = await ctx.get_kv("drive:boredom", {}) or {}
        stress = await ctx.get_kv("drive:stress", {}) or {}
        maint = await ctx.get_kv("memory:last_sleep_maintenance", {}) or {}
        want_vector = await ctx.get_kv("drive:want_vector", {}) or {}

        pct = _safe_float(power_state.get("pct", await ctx.get_kv("power:battery_pct", 100.0)), 100.0)
        urgency = _safe_float(pressure.get("urgency", 0.0), 0.0)
        charging = bool(power_state.get("charging", await ctx.get_kv("power:charging", False)))
        sleeping = bool(power_state.get("sleep", await ctx.get_kv("power:sleep", False)))
        mode = str(power_state.get("mode", await ctx.get_kv("power:mode", "active")) or "active")
        stress_level = _safe_float(stress.get("level", 0.0), 0.0) if isinstance(stress, Mapping) else 0.0
        boredom_level = _safe_float(boredom.get("level", 0.0), 0.0) if isinstance(boredom, Mapping) else 0.0
        curiosity_boost = _safe_float(await ctx.get_kv("curiosity:boost", 0.0), 0.0)
        inquiry = _safe_float(hormones.get("inquiry", 0.0), 0.0) if isinstance(hormones, Mapping) else 0.0
        externalize = _safe_float(want_vector.get("externalize", 0.0), 0.0) if isinstance(want_vector, Mapping) else 0.0
        maint_state = "stable"
        if isinstance(maint, Mapping) and maint:
            skipped = int(_safe_float(maint.get("skipped", 0), 0))
            promoted = int(_safe_float(maint.get("promoted", 0), 0))
            written = int(_safe_float(maint.get("written", 0), 0))
            maint_state = f"stable; last sleep pass promoted {promoted}, wrote {written}, skipped {skipped}"

        if charging:
            power_phrase = f"power is {pct:.0f}% and charging"
        elif sleeping:
            power_phrase = f"power is {pct:.0f}% and sleep mode is active"
        elif urgency >= 0.85:
            power_phrase = f"power is {pct:.0f}% with critical charge pressure"
        elif urgency >= 0.55:
            power_phrase = f"power is {pct:.0f}% with elevated charge pressure"
        else:
            power_phrase = f"power is {pct:.0f}% and stable"

        return (
            "Internal status: "
            f"{power_phrase}; power urgency {urgency:.2f}; "
            f"stress {stress_level:.2f}; boredom {boredom_level:.2f}; "
            f"curiosity {max(curiosity_boost, inquiry):.2f}; externalize {externalize:.2f}; "
            f"mode {mode}; maintenance {maint_state}."
        )

    def _is_low_value_answer(self, answer: str, *, query_text: str, conversation_summary: Mapping[str, Any]) -> bool:
        norm_answer = _norm(answer)
        if not norm_answer:
            return True
        if _looks_stock_reply(answer):
            return True
        if _conversation_active(conversation_summary) and norm_answer.startswith("the question is about"):
            return True
        # Avoid answering short follow-ups with a lexical reflection of the helper word.
        norm_q = _norm(query_text)
        if norm_q in {"like what", "any questions", "do you have any", "what about it", "what about that"}:
            return True
        return False

    async def _mem_store(self, ctx) -> MemCellStore | None:
        store = await ctx.get_kv("memory:mem_cell_store", None)
        if isinstance(store, MemCellStore):
            return store
        memdir = await ctx.get_kv("cfg:memdir", None) or await ctx.get_kv("memdir", None)
        if not memdir:
            return None
        try:
            store = MemCellStore(str(memdir))
            await ctx.set_kv("memory:mem_cell_store", store)
            return store
        except Exception:
            return None

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "reason/request":
            return []

        # Native responder owns the default path only when backend reasoning is OFF.
        if bool(await ctx.get_kv("llm:enabled", False)):
            return []

        payload = event.payload if isinstance(event.payload, dict) else {"text": event.payload}
        text = str(payload.get("text", "") or "").strip()
        channel = str(payload.get("channel", "repl") or "repl")
        source = str(payload.get("source", "user") or "user")
        raw_meta = payload.get("raw_meta", {}) if isinstance(payload.get("raw_meta", {}), dict) else {}
        hypothesis = payload.get("hypothesis", {}) if isinstance(payload.get("hypothesis", {}), Mapping) else {}
        selected_action = str(payload.get("selected_action", "") or raw_meta.get("selected_action", "") or hypothesis.get("recommended_action", "") or "")
        transport_source = str(raw_meta.get("transport_source", source) or source)

        if not text:
            return []

        # Keep purely internal traffic internal.
        if channel in ("internal", "thought"):
            return []

        mem_store = await self._mem_store(ctx)
        syntax_guidance = self._syntax_guidance(mem_store, text)
        learned_direct_reply = bool(list(syntax_guidance.get("preferred_replies", []) or []))

        shape = await self._shape_reply(
            ctx,
            text=text,
            channel=channel,
            transport_source=transport_source,
            raw_meta=raw_meta,
            hypothesis=hypothesis,
            selected_action=selected_action,
            learned_direct_reply=learned_direct_reply,
        )
        if shape.get("suppress", False):
            await ctx.log_debug(
                f"[{self.name}] Suppressed outward native reply",
                reason=str(shape.get("reason", "withhold")),
                text_preview=text[:100],
            )
            return []

        reply = await self._build_response(
            ctx,
            text=text,
            shape=shape,
            payload=payload,
            mem_store=mem_store,
            syntax_guidance=syntax_guidance,
        )
        if not reply:
            return []

        memory_cell_ids: List[str] = []
        answer_bundle = await ctx.get_kv("composer:last_answer_bundle", {})
        if isinstance(answer_bundle, Mapping):
            bundle_query = str(answer_bundle.get("query_text", "") or "")
            bundle_answer = str(answer_bundle.get("answer", "") or "")
            bundle_ts = _safe_float(answer_bundle.get("ts", 0.0), 0.0)
            if (
                _norm(bundle_query) == _norm(text)
                and _norm(bundle_answer) == _norm(reply)
                and bundle_ts > 0.0
                and (time.time() - bundle_ts) <= 15.0
            ):
                memory_cell_ids = [
                    str(cell_id or "")
                    for cell_id in list(answer_bundle.get("selected_cell_ids", []) or [])
                    if str(cell_id or "")
                ][:12]

        await ctx.set_kv(
            "native_responder:last",
            {
                "ts": time.time(),
                "text": text,
                "reply": reply,
                "shape": shape,
                "channel": channel,
                "transport_source": transport_source,
            },
        )

        return [
            Event(
                topic="act/speech",
                payload={
                    "text": reply,
                    "channel": channel,
                    "style": "assistant",
                    "memory_cell_ids": memory_cell_ids,
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "native_responder_reply",
                    "transport_source": transport_source,
                    "shape": shape,
                    "memory_cell_ids": memory_cell_ids,
                },
            )
        ]

    async def _shape_reply(
        self,
        ctx,
        *,
        text: str,
        channel: str,
        transport_source: str,
        raw_meta: Mapping[str, Any] | None = None,
        hypothesis: Mapping[str, Any] | None = None,
        selected_action: str = "",
        learned_direct_reply: bool = False,
    ) -> Dict[str, Any]:
        hormones = await ctx.get_kv("drive:hormones", {}) or {}
        wants = await ctx.get_kv("drive:want_vector", {}) or {}
        ddna = await ctx.get_kv("drive:ddna_modulators", {}) or {}
        atomized = await ctx.get_kv("language:last_atomized", {}) or {}
        rosehip = await ctx.get_kv("drive:rosehip", {}) or {}
        needs = await ctx.get_kv("drive:needs_stack", {}) or {}
        social_interaction = await ctx.get_kv("drive:social_interaction", {}) or {}
        social_experimentation = await ctx.get_kv("drive:social_experimentation", {}) or {}
        thought_momentum = await ctx.get_kv("thought:momentum", {}) or {}
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        hypothesis = hypothesis if isinstance(hypothesis, Mapping) else {}
        hypothesis_pattern = hypothesis.get("pattern_analysis", {}) if isinstance(hypothesis.get("pattern_analysis", {}), Mapping) else {}
        hypothesis_response_demand = _safe_float(hypothesis.get("response_demand", 0.0), 0.0)
        hypothesis_should_respond = bool(hypothesis.get("should_respond", False))
        hypothesis_uncertainty = _safe_float(hypothesis_pattern.get("uncertainty", 0.0), 0.0)
        hypothesis_continuity = _safe_float(hypothesis_pattern.get("continuity", 0.0), 0.0)
        hypothesis_statement_kind = str(hypothesis_pattern.get("statement_kind", "statement") or "statement")
        selected_action = str(selected_action or hypothesis.get("recommended_action", "") or "")
        accent_value, accent_magnitude, accent_positive, accent_negative_severity, tone_label = _accent_from_meta(raw_meta)

        text_norm = _norm(text)
        is_question = text.strip().endswith("?")
        direct_response_request = any(
            key in text_norm for key in ("please respond", "respond", "reply", "speak up", "can you hear me")
        )
        parse_request = any(
            key in text_norm for key in ("what did you parse", "what do you see", "what did you get")
        )
        greeting = text_norm in ("hi", "hello", "hey", "yo", "howdy")

        externalize = _safe_float(wants.get("externalize", 0.0))
        withhold = _safe_float(wants.get("withhold", 0.0))
        inquire = _safe_float(wants.get("inquire", 0.0))
        connect = _safe_float(wants.get("connect", 0.0))
        social_level = _safe_float(social_interaction.get("level", 0.0), 0.0) if isinstance(social_interaction, Mapping) else 0.0
        social_experiment_pressure = _safe_float(social_experimentation.get("pressure", 0.0), 0.0) if isinstance(social_experimentation, Mapping) else 0.0
        momentum_pressure = _safe_float(thought_momentum.get("pressure", 0.0), 0.0) if isinstance(thought_momentum, Mapping) else 0.0
        momentum_intent = str(thought_momentum.get("dominant_intent", "") or "") if isinstance(thought_momentum, Mapping) else ""
        caution = _safe_float(hormones.get("caution", 0.0))
        affiliation = _safe_float(hormones.get("affiliation", 0.0))
        continuity = _safe_float(hormones.get("continuity", 0.0))
        inquiry_h = _safe_float(hormones.get("inquiry", 0.0))
        expression_bias = _safe_float(ddna.get("expression_bias", 1.0), 1.0)
        restraint_bias = _safe_float(ddna.get("restraint_bias", 1.0), 1.0)

        # Direct address should almost always get some answer. Statements can be held back more.
        direct_bonus = 0.0
        if is_question:
            direct_bonus += 0.28
        if direct_response_request:
            direct_bonus += 0.32
        if parse_request or learned_direct_reply or greeting:
            direct_bonus += 0.24
        if transport_source in ("textual", "cli", "ui", "mic"):
            direct_bonus += 0.12
        if accent_positive >= 7.0:
            # Positive /acc is a stronger emphasis/preference bid, not literal content.
            direct_bonus += min(0.14, accent_positive / 100.0)
        elif accent_negative_severity >= 7.0:
            # Negative /acc is important as correction, but it should not become
            # a positive salience/preference boost for repeating the content.
            direct_bonus -= min(0.10, accent_negative_severity / 120.0)
        elif accent_magnitude <= 0.25 and raw_meta.get("accent_source") == "acc_command":
            # /acc 0 = explicitly neutral tone; do not add salience.
            direct_bonus -= 0.02

        if momentum_pressure >= 0.20:
            direct_bonus += min(0.08, momentum_pressure * 0.07)
        if hypothesis_should_respond:
            direct_bonus += min(0.24, hypothesis_response_demand * 0.24)
        if selected_action in {"clarify", "ask_followup"}:
            direct_bonus += min(0.08, hypothesis_uncertainty * 0.08)
        elif selected_action in {"acknowledge", "acknowledge_revision", "continue_thread", "reflect"}:
            direct_bonus += min(0.08, hypothesis_continuity * 0.08)

        outward_urge = _clamp((externalize * expression_bias) + (0.18 * connect) + (0.10 * continuity) + (0.08 * social_level) + direct_bonus)
        brake = _clamp((withhold * restraint_bias) + (0.10 * caution))

        if not isinstance(rosehip, Mapping) or not rosehip:
            rosehip = derive_rosehip_state(
                hormones,
                needs=needs,
                ddna=ddna,
                context={
                    "interruption_cost": 0.0,
                    "redundancy": 0.0,
                    "confidence": 0.65,
                    "direct_address": 1.0 if (is_question or direct_response_request or hypothesis_should_respond) else 0.0,
                    "recent_user": 1.0,
                    "answered": 0.0,
                    "recent_reply": 0.0,
                    "repeated_direct": 0.0,
                    "sleeping": False,
                    "charging": False,
                },
            )

        expression_brake = _safe_float(rosehip.get("expression_brake", 0.0))
        social_brake = _safe_float(rosehip.get("social_brake", 0.0))
        redundancy_brake = _safe_float(rosehip.get("redundancy_brake", 0.0))
        interrupt_brake = _safe_float(rosehip.get("interrupt_brake", 0.0))
        sleep_quiet_brake = _safe_float(rosehip.get("sleep_quiet_brake", 0.0))
        confidence_brake = _safe_float(rosehip.get("confidence_brake", 0.0))
        clarify_bias = _safe_float(rosehip.get("clarify_bias", 0.0))
        outward_scale = max(0.05, _safe_float(rosehip.get("outward_scale", 1.0), 1.0))
        direct_reply_floor = _safe_float(rosehip.get("direct_reply_floor", 0.0))
        external_bias = _safe_float(rosehip.get("external_bias", 0.0))
        accent_negative_brake = min(0.18, accent_negative_severity / 70.0)
        accent_positive_push = min(0.12, accent_positive / 85.0)

        release_score = _clamp(
            (outward_urge * outward_scale)
            + (0.10 * external_bias)
            + accent_positive_push
            - accent_negative_brake
            - brake
            - (0.18 * expression_brake)
            - (0.10 * social_brake)
            - (0.12 * redundancy_brake)
            - (0.18 * interrupt_brake)
            - (0.24 * sleep_quiet_brake)
            - (0.10 * confidence_brake)
        )
        if is_question or direct_response_request or parse_request or learned_direct_reply or greeting:
            release_score = max(release_score, direct_reply_floor)
        elif hypothesis_should_respond:
            hypothesis_floor = min(0.38, 0.16 + (0.22 * hypothesis_response_demand))
            release_score = max(release_score, hypothesis_floor)

        terse = _clamp(
            (0.55 * restraint_bias)
            + (0.22 * caution)
            + (0.10 * accent_negative_severity / 10.0)
            + (0.24 * expression_brake)
            + (0.18 * redundancy_brake)
            + (0.14 * interrupt_brake)
            - (0.20 * expression_bias)
            - (0.15 * affiliation)
        )
        warm = _clamp(
            (0.40 * affiliation)
            + (0.18 * connect)
            + (0.10 * social_level)
            + (0.05 * social_experiment_pressure)
            + (0.04 * momentum_pressure if momentum_intent in {"social_continuity", "seek_social_contact", "social_experiment"} else 0.0)
            + (0.03 * accent_positive / 10.0)
            - (0.05 * accent_negative_severity / 10.0)
            + (0.12 * expression_bias)
            - (0.10 * restraint_bias)
            - (0.18 * social_brake)
            - (0.12 * redundancy_brake)
        )
        clarify_first = _clamp(
            (0.35 * inquiry_h)
            + (0.22 * caution)
            + (0.10 * withhold)
            - (0.08 * connect)
            + (0.22 * clarify_bias)
            + (0.05 * momentum_pressure if momentum_intent in {"understand_user", "resolve_thread", "curiosity"} else 0.0)
            + (0.06 * accent_negative_severity / 10.0)
            - (0.10 * redundancy_brake)
        )

        relation_count = len(atomized.get("relations", [])) if isinstance(atomized, Mapping) else 0
        noun_count = len(atomized.get("nouns", [])) if isinstance(atomized, Mapping) else 0
        parse_available = (relation_count + noun_count) > 0

        suppress = False
        reason = ""
        if sleep_quiet_brake >= 0.70:
            suppress = True
            reason = "rosehip_sleep_quiet"
        elif not (is_question or direct_response_request or parse_request or learned_direct_reply or greeting or hypothesis_should_respond):
            if release_score < 0.18:
                suppress = True
                reason = "low_release_score"

        mode = "direct"
        if selected_action in {"clarify", "ask_followup"}:
            mode = "clarify"
        elif selected_action in {"acknowledge", "acknowledge_revision"}:
            mode = "ack"
        elif clarify_first >= 0.52 and is_question and not parse_request and not learned_direct_reply:
            mode = "clarify"
        elif release_score < 0.28 and not (is_question or direct_response_request):
            mode = "ack"
        elif parse_request and parse_available:
            mode = "parse_reflect"
        elif parse_request:
            mode = "parse_empty"
        elif relation_count > 0 and is_question:
            mode = "relation_reflect"
        elif noun_count > 0 and not is_question and release_score >= 0.35:
            mode = "noun_reflect"

        return {
            "suppress": suppress,
            "reason": reason,
            "mode": mode,
            "release_score": round(release_score, 4),
            "outward_urge": round(outward_urge, 4),
            "brake": round(brake, 4),
            "terse": round(terse, 4),
            "warm": round(warm, 4),
            "clarify_first": round(clarify_first, 4),
            "expression_bias": round(expression_bias, 4),
            "restraint_bias": round(restraint_bias, 4),
            "social_level": round(social_level, 4),
            "social_experiment_pressure": round(social_experiment_pressure, 4),
            "thought_momentum_pressure": round(momentum_pressure, 4),
            "thought_momentum_intent": momentum_intent,
            "accent_value": round(accent_value, 4),
            "accent_magnitude": round(accent_magnitude, 4),
            "accent_positive": round(accent_positive, 4),
            "accent_negative_severity": round(accent_negative_severity, 4),
            "tone_label": tone_label,
            "hypothesis_should_respond": hypothesis_should_respond,
            "hypothesis_response_demand": round(hypothesis_response_demand, 4),
            "hypothesis_uncertainty": round(hypothesis_uncertainty, 4),
            "hypothesis_continuity": round(hypothesis_continuity, 4),
            "hypothesis_statement_kind": hypothesis_statement_kind,
            "selected_action": selected_action,
        }

    async def _build_response(
        self,
        ctx,
        *,
        text: str,
        shape: Dict[str, Any],
        payload: Dict[str, Any],
        mem_store: MemCellStore | None = None,
        syntax_guidance: Mapping[str, Any] | None = None,
    ) -> str:
        """Build only from learned/internal sources; no canned speech fallbacks.

        Prebuilt test-pulse lines proved the routes were live. This responder now
        speaks only when a learned syntax rule, internal status read, memory
        composer, recalled phrase, or explicit reasoning output can supply text.
        If none exists, it stays silent so the thought path can keep working.
        """
        norm = _norm(text)
        atomized = await ctx.get_kv("language:last_atomized", {}) or {}
        relations = atomized.get("relations", []) if isinstance(atomized, Mapping) else []
        if not isinstance(mem_store, MemCellStore):
            mem_store = await self._mem_store(ctx)

        thought_path_last = await ctx.get_kv("thought_path:last", {}) or {}
        power_state = await ctx.get_kv("power:state", {}) or {}
        needs = await ctx.get_kv("drive:needs_stack", {}) or {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        associations = list(context.get("associations", []) or []) if isinstance(context, Mapping) else []
        association_meta = dict(context.get("association_meta", {}) or {}) if isinstance(context, Mapping) else {}
        conversation_summary = dict(context.get("conversation_summary", {}) or {}) if isinstance(context, Mapping) and isinstance(context.get("conversation_summary", {}), Mapping) else {}

        warm = _safe_float(shape.get("warm", 0.0))
        mode = str(shape.get("mode", "direct") or "direct")
        if not isinstance(syntax_guidance, Mapping):
            syntax_guidance = self._syntax_guidance(mem_store, text)
        syntax_reply = self._preferred_rule_reply(dict(syntax_guidance), warm=warm)
        avoid_norms = {_norm(t) for t in list(syntax_guidance.get("avoid_replies", []) or []) if _norm(str(t or ""))}

        internal_status_reply = await self._internal_status_reply(ctx, text=text, norm=norm, terse=_safe_float(shape.get("terse", 0.0)))
        if internal_status_reply:
            return internal_status_reply

        def best_recalled_phrase() -> str:
            best_text = ""
            best_score = 0.0
            top_assoc_score = _safe_float(association_meta.get("top_score", 0.0), 0.0)
            if top_assoc_score >= 0.42:
                for assoc in associations[:4]:
                    if not isinstance(assoc, Mapping):
                        continue
                    candidate = str(assoc.get("text", "") or "").strip()
                    candidate_norm = _norm(candidate)
                    if (
                        not candidate_norm
                        or candidate_norm == norm
                        or candidate_norm in avoid_norms
                        or _looks_stock_reply(candidate)
                        or self._context_breaks_scene(candidate, norm, conversation_summary)
                    ):
                        continue
                    if candidate.startswith("/") or len(candidate.split()) > 18:
                        continue
                    score = _safe_float(assoc.get("score", top_assoc_score), top_assoc_score)
                    if score > best_score:
                        best_text = candidate
                        best_score = score

            if isinstance(mem_store, MemCellStore):
                try:
                    hits = mem_store.search_text_cells(text, limit=10)
                except Exception:
                    hits = []
                for hit in hits:
                    if not isinstance(hit, Mapping):
                        continue
                    meta = dict(hit.get("meta", {}) or {})
                    role = str(meta.get("role", "") or "")
                    if role and role not in ("assistant", "system"):
                        continue
                    refs = hit.get("refs", []) if isinstance(hit.get("refs", []), list) else []
                    candidate = str(refs[0] if refs else hit.get("anchor_text", "") or "").strip()
                    candidate_norm = _norm(candidate)
                    if (
                        not candidate_norm
                        or candidate_norm == norm
                        or candidate_norm in avoid_norms
                        or _looks_stock_reply(candidate)
                        or self._context_breaks_scene(candidate, norm, conversation_summary)
                    ):
                        continue
                    if candidate.startswith("/") or len(candidate.split()) > 18:
                        continue
                    score = (
                        _safe_float(hit.get("score", 0.0), 0.0)
                        + (0.06 if role in ("assistant", "system") else 0.0)
                        + self._ddna_bonus(meta, warm=warm)
                    )
                    if score > best_score:
                        best_text = candidate
                        best_score = score
            return best_text if best_score >= 0.48 else ""

        if syntax_reply:
            return syntax_reply

        if mode == "relation_reflect" and relations:
            rel = relations[0]
            subj = str(rel.get("subject", "") or "something")
            relation = str(rel.get("relation", "") or "related_to")
            obj = str(rel.get("object", "") or "")
            await ctx.emit(Event(
                topic="thought/internal",
                payload={
                    "kind": "relation_parse",
                    "subject": subj,
                    "relation": relation,
                    "object": obj,
                },
                source=self.name,
                meta={
                    "channel": "thought",
                    "kind": "relation_parse",
                    "store_in_memory": False,
                    "reinforcement_eligible": False,
                    "self_output_track": False,
                    "cognitive_visible": False,
                },
            ))
            return ""

        if text.strip().endswith("?"):
            bundle = gather_support(
                query_text=text,
                mem_cell_store=mem_store if isinstance(mem_store, MemCellStore) else None,
                power_state=power_state if isinstance(power_state, Mapping) else {},
                needs=needs if isinstance(needs, Mapping) else {},
                thought_path_last=thought_path_last if isinstance(thought_path_last, Mapping) else {},
            )
            answer, confidence, answer_meta = compose_answer(bundle)
            if answer:
                selected_cell_ids = [
                    str(cell_id or "")
                    for cell_id in list(answer_meta.get("selected_cell_ids", []) or [])
                    if str(cell_id or "")
                ]
                await ctx.set_kv(
                    "composer:last_answer_bundle",
                    {
                        "query_text": text,
                        "bundle": bundle,
                        "answer": answer,
                        "confidence": confidence,
                        "meta": answer_meta,
                        "selected_cell_ids": selected_cell_ids,
                        "ts": time.time(),
                    },
                )
                if not self._is_low_value_answer(answer, query_text=text, conversation_summary=conversation_summary):
                    return answer
            return best_recalled_phrase()

        return best_recalled_phrase()

    def _conversation_continuity_reply(self, summary: Mapping[str, Any], *, text: str, terse: float = 0.0) -> str:
        """Canned continuity speech removed; continuity stays as scene state."""
        return ""


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["reason/request"],
        output_topics=["act/speech"],
        priority=4,
        cooldown_sec=0.0,
    )
    yield NativeResponderNeuron(cfg)
