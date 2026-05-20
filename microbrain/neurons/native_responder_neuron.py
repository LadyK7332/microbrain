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


def _looks_stock_reply(text: str) -> bool:
    norm = _norm(text)
    return norm.startswith("i heard your question") or norm.startswith("i heard you")


def _accent_from_meta(raw_meta: Mapping[str, Any]) -> tuple[float, float, str]:
    try:
        value = float(raw_meta.get("accent_value", 0.0) or 0.0)
    except Exception:
        value = 0.0
    if value < -10.0:
        value = -10.0
    if value > 10.0:
        value = 10.0
    try:
        intensity = float(raw_meta.get("accent_intensity", abs(value)) or abs(value))
    except Exception:
        intensity = abs(value)
    intensity = max(0.0, min(10.0, intensity))
    label = str(raw_meta.get("tone_label", "") or "").strip()
    return value, intensity, label


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
            hits = mem_store.search_text_cells(lookup_text, limit=16, tiers=("learned", "long", "now", "short"))
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
            score = _safe_float(hit.get("score", 0.0), 0.0)
            for key, value in self._meta_ddna_targets(meta).items():
                guidance["ddna_targets"][key] = max(float(guidance["ddna_targets"].get(key, 0.0)), value)
            for classifier in list(meta.get("syntax_classifiers", []) or []):
                name = _norm(str(classifier or "")).replace(" ", "_")
                if name and name not in guidance["classifiers"]:
                    guidance["classifiers"].append(name)
            reply = str(meta.get("reply_text", "") or meta.get("desired_utterance", "") or "").strip()
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
        transport_source = str(raw_meta.get("transport_source", source) or source)

        if not text:
            return []

        # Keep purely internal traffic internal.
        if channel in ("internal", "thought"):
            return []

        shape = await self._shape_reply(ctx, text=text, channel=channel, transport_source=transport_source, raw_meta=raw_meta)
        if shape.get("suppress", False):
            await ctx.log_debug(
                f"[{self.name}] Suppressed outward native reply",
                reason=str(shape.get("reason", "withhold")),
                text_preview=text[:100],
            )
            return []

        reply = await self._build_response(ctx, text=text, shape=shape, payload=payload)
        if not reply:
            return []

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
                },
                source=self.name,
                correlation_id=event.correlation_id,
                meta={
                    "kind": "native_responder_reply",
                    "transport_source": transport_source,
                    "shape": shape,
                },
            )
        ]

    async def _shape_reply(self, ctx, *, text: str, channel: str, transport_source: str, raw_meta: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        hormones = await ctx.get_kv("drive:hormones", {}) or {}
        wants = await ctx.get_kv("drive:want_vector", {}) or {}
        ddna = await ctx.get_kv("drive:ddna_modulators", {}) or {}
        atomized = await ctx.get_kv("language:last_atomized", {}) or {}
        rosehip = await ctx.get_kv("drive:rosehip", {}) or {}
        needs = await ctx.get_kv("drive:needs_stack", {}) or {}
        social_interaction = await ctx.get_kv("drive:social_interaction", {}) or {}
        social_experimentation = await ctx.get_kv("drive:social_experimentation", {}) or {}
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        accent_value, accent_intensity, tone_label = _accent_from_meta(raw_meta)

        text_norm = _norm(text)
        is_question = text.strip().endswith("?")
        direct_response_request = any(
            key in text_norm for key in ("please respond", "respond", "reply", "speak up", "can you hear me")
        )
        parse_request = any(
            key in text_norm for key in ("what did you parse", "what do you see", "what did you get")
        )
        say_request = text_norm.startswith("say ")
        greeting = text_norm in ("hi", "hello", "hey", "yo", "howdy")

        externalize = _safe_float(wants.get("externalize", 0.0))
        withhold = _safe_float(wants.get("withhold", 0.0))
        inquire = _safe_float(wants.get("inquire", 0.0))
        connect = _safe_float(wants.get("connect", 0.0))
        social_level = _safe_float(social_interaction.get("level", 0.0), 0.0) if isinstance(social_interaction, Mapping) else 0.0
        social_experiment_pressure = _safe_float(social_experimentation.get("pressure", 0.0), 0.0) if isinstance(social_experimentation, Mapping) else 0.0
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
        if parse_request or say_request or greeting:
            direct_bonus += 0.24
        if transport_source in ("textual", "cli", "ui", "mic"):
            direct_bonus += 0.12
        if accent_intensity >= 7.0:
            # High explicit tone/intensity is a stronger bid for attention, not literal content.
            direct_bonus += min(0.14, accent_intensity / 100.0)
        elif accent_intensity <= 0.25 and raw_meta.get("accent_source") == "acc_command":
            # /acc 0 = intentionally flat/dead tone; treat as lower expressive pressure.
            direct_bonus -= 0.04

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
                    "direct_address": 1.0 if (is_question or direct_response_request) else 0.0,
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

        release_score = _clamp(
            (outward_urge * outward_scale)
            + (0.10 * external_bias)
            - brake
            - (0.18 * expression_brake)
            - (0.10 * social_brake)
            - (0.12 * redundancy_brake)
            - (0.18 * interrupt_brake)
            - (0.24 * sleep_quiet_brake)
            - (0.10 * confidence_brake)
        )
        if is_question or direct_response_request or parse_request or say_request or greeting:
            release_score = max(release_score, direct_reply_floor)

        terse = _clamp(
            (0.55 * restraint_bias)
            + (0.22 * caution)
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
            + (0.03 * max(0.0, accent_value) / 10.0)
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
        elif not (is_question or direct_response_request or parse_request or say_request or greeting):
            if release_score < 0.18:
                suppress = True
                reason = "low_release_score"

        mode = "direct"
        if clarify_first >= 0.52 and is_question and not parse_request and not say_request:
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
            "accent_value": round(accent_value, 4),
            "accent_intensity": round(accent_intensity, 4),
            "tone_label": tone_label,
        }

    async def _build_response(self, ctx, *, text: str, shape: Dict[str, Any], payload: Dict[str, Any]) -> str:
        norm = _norm(text)
        atomized = await ctx.get_kv("language:last_atomized", {}) or {}
        noun_candidates = atomized.get("nouns", []) if isinstance(atomized, Mapping) else []
        noun_chunks = atomized.get("noun_chunks", []) if isinstance(atomized, Mapping) else []
        relations = atomized.get("relations", []) if isinstance(atomized, Mapping) else []
        mem_store = await ctx.get_kv("memory:mem_cell_store", None)
        if mem_store is None:
            memdir = await ctx.get_kv("cfg:memdir", None) or await ctx.get_kv("memdir", None)
            if memdir:
                try:
                    mem_store = MemCellStore(str(memdir))
                    await ctx.set_kv("memory:mem_cell_store", mem_store)
                except Exception:
                    mem_store = None
        thought_path_last = await ctx.get_kv("thought_path:last", {}) or {}
        power_state = await ctx.get_kv("power:state", {}) or {}
        needs = await ctx.get_kv("drive:needs_stack", {}) or {}
        context = payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {}
        associations = list(context.get("associations", []) or []) if isinstance(context, Mapping) else []
        association_meta = dict(context.get("association_meta", {}) or {}) if isinstance(context, Mapping) else {}

        terse = _safe_float(shape.get("terse", 0.0))
        warm = _safe_float(shape.get("warm", 0.0))
        mode = str(shape.get("mode", "direct") or "direct")
        syntax_guidance = self._syntax_guidance(mem_store, text)
        syntax_reply = self._preferred_rule_reply(syntax_guidance, warm=warm)
        avoid_norms = {_norm(t) for t in list(syntax_guidance.get("avoid_replies", []) or []) if _norm(str(t or ""))}

        def choose(short: str, medium: str, warmish: str | None = None) -> str:
            if warm >= 0.55 and warmish:
                return warmish
            if terse >= 0.56:
                return short
            return medium

        def best_recalled_phrase() -> str:
            best_text = ""
            best_score = 0.0

            top_assoc_score = _safe_float(association_meta.get("top_score", 0.0), 0.0)
            if top_assoc_score >= 0.42:
                for assoc in associations[:4]:
                    if not isinstance(assoc, Mapping):
                        continue
                    candidate = str(assoc.get("text", "") or "").strip()
                    if not candidate:
                        continue
                    candidate_norm = _norm(candidate)
                    if not candidate_norm or candidate_norm == norm or candidate_norm in avoid_norms or _looks_stock_reply(candidate):
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
                    candidate = ""
                    refs = hit.get("refs", []) if isinstance(hit.get("refs", []), list) else []
                    if refs:
                        candidate = str(refs[0] or "").strip()
                    if not candidate:
                        candidate = str(hit.get("anchor_text", "") or "").strip()
                    candidate_norm = _norm(candidate)
                    if not candidate_norm or candidate_norm == norm or candidate_norm in avoid_norms or _looks_stock_reply(candidate):
                        continue
                    if candidate.startswith("/") or len(candidate.split()) > 18:
                        continue
                    if text.strip().endswith("?") and not candidate.strip().endswith(("?", ".")):
                        candidate = candidate.rstrip() + "."
                    score = (
                        _safe_float(hit.get("score", 0.0), 0.0)
                        + (0.06 if role in ("assistant", "system") else 0.0)
                        + self._ddna_bonus(meta, warm=warm)
                    )
                    if score > best_score:
                        best_text = candidate
                        best_score = score

            return best_text if best_score >= 0.48 else ""

        recalled_phrase = best_recalled_phrase()

        if syntax_reply:
            return syntax_reply

        if norm in ("hi", "hello", "hey", "yo", "howdy"):
            return choose(
                "Hello.",
                "Hello. I'm here and listening.",
                "Hello. I'm here and listening.",
            )

        if "can you hear me" in norm:
            return choose(
                "Yes.",
                "Yes. I hear you.",
                "Yes. I hear you.",
            )

        if "please respond" in norm or norm == "respond" or norm == "reply" or "speak up" in norm:
            return choose(
                "I'm here.",
                "I hear you. Give me a direct question, goal, or choice and I'll answer plainly.",
                "I hear you. Give me a direct question, goal, or choice and I'll answer plainly.",
            )

        if norm in ("thanks", "thank you"):
            return choose("You're welcome.", "You're welcome.", "You're welcome.")

        if norm in ("bye", "goodbye", "see you"):
            return choose(
                "Alright.",
                "Alright. I'll be here when you get back.",
                "Alright. I'll be here when you get back.",
            )

        if "what can you do" in norm:
            return choose(
                "Listen, track context, and reply.",
                "I can listen, track context, connect memory, and respond directly. Give me a concrete target and I'll work from there.",
                "I can listen, track context, connect memory, and respond directly. Give me a concrete target and I'll work from there.",
            )

        if norm.startswith("say "):
            say_text = text[4:].strip()
            if say_text:
                return say_text

        if mode == "parse_reflect":
            if noun_chunks:
                return choose(
                    f"Noun chunks: {', '.join(noun_chunks[:2])}.",
                    f"I parsed noun chunks as: {', '.join(noun_chunks[:4])}.",
                    f"I parsed noun chunks as: {', '.join(noun_chunks[:4])}.",
                )
            if noun_candidates:
                lemmas = [str(n.get("lemma", "")) for n in noun_candidates[:4] if str(n.get("lemma", ""))]
                if lemmas:
                    return choose(
                        f"Nouns: {', '.join(lemmas[:2])}.",
                        f"I extracted noun candidates: {', '.join(lemmas)}.",
                        f"I extracted noun candidates: {', '.join(lemmas)}.",
                    )

        if mode == "parse_empty":
            return choose(
                "Not much yet.",
                "I did not get strong noun chunks from that line.",
                "I did not get strong noun chunks from that line.",
            )

        if mode == "relation_reflect" and relations:
            rel = relations[0]
            subj = str(rel.get("subject", "") or "something")
            relation = str(rel.get("relation", "") or "related to")
            obj = str(rel.get("object", "") or "something")
            if obj:
                return choose(
                    f"I got: {subj} {relation} {obj}.",
                    f"I parsed a relation that looks like: {subj} {relation} {obj}.",
                    f"I parsed a relation that looks like: {subj} {relation} {obj}.",
                )
            return choose(
                f"I got: {subj} {relation}.",
                f"I parsed a relation centered on {subj} and {relation}.",
                f"I parsed a relation centered on {subj} and {relation}.",
            )

        if mode == "noun_reflect" and noun_candidates:
            lemmas = [str(n.get("lemma", "")) for n in noun_candidates[:3] if str(n.get("lemma", ""))]
            if lemmas:
                return choose(
                    f"I heard: {', '.join(lemmas[:2])}.",
                    f"I heard you. The strongest noun candidates are: {', '.join(lemmas)}.",
                    f"I heard you. The strongest noun candidates are: {', '.join(lemmas)}.",
                )

        if mode == "clarify":
            return choose(
                "Need one variable.",
                "I can answer, but I need one missing variable first. Do you want explanation, action, or analysis?",
                "I can answer, but I need one missing variable first. Do you want explanation, action, or analysis?",
            )

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
                if selected_cell_ids and isinstance(mem_store, MemCellStore):
                    for cell_id in selected_cell_ids[:6]:
                        try:
                            mem_store.note_cell_usage(cell_id, success=True)
                        except Exception:
                            pass
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
                return answer
            if recalled_phrase:
                return recalled_phrase
            return choose(
                "Need a target.",
                "Give me the concrete target or missing variable and I'll answer directly.",
                "Give me the concrete target or missing variable and I'll answer directly.",
            )

        if mode == "ack":
            if recalled_phrase:
                return recalled_phrase
            return choose("Noted.", "I heard you.", "I heard you.")

        if recalled_phrase:
            return recalled_phrase
        return choose(
            "Need one target.",
            "Give me a concrete goal, question, or choice and I'll respond directly.",
            "Give me a concrete goal, question, or choice and I'll respond directly.",
        )


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["reason/request"],
        output_topics=["act/speech"],
        priority=4,
        cooldown_sec=0.0,
    )
    yield NativeResponderNeuron(cfg)
