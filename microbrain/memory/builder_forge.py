from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

TOKEN_RE = re.compile(r"[a-z0-9']+")
COPULAS = {"is", "are", "was", "were", "be", "been", "being"}
DETERMINERS = {"a", "an", "the", "this", "that", "my", "your", "our", "their"}


def _norm(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower())).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t]


def _is_pluralish(word: str) -> bool:
    s = str(word or "").strip().lower()
    if not s:
        return False
    return s.endswith("s") and not s.endswith("ss")


def _article_for(word: str) -> str:
    s = str(word or "").strip().lower()
    if not s:
        return "a"
    return "an" if s[:1] in {"a", "e", "i", "o", "u"} else "a"


def _clean_value(value: Any) -> str:
    return " ".join(_tokens(str(value or "")))


def _add_slot(slots: MutableMapping[str, List[Dict[str, Any]]], name: str, value: Any, score: float, source_id: str = "") -> None:
    text = _clean_value(value)
    if not text:
        return
    bucket = slots.setdefault(name, [])
    key = text.lower()
    for existing in bucket:
        if str(existing.get("value", "")).lower() == key:
            existing["score"] = max(float(existing.get("score", 0.0) or 0.0), float(score or 0.0))
            if source_id:
                source_ids = list(existing.get("source_ids", []) or [])
                if source_id not in source_ids:
                    source_ids.append(source_id)
                existing["source_ids"] = source_ids[:8]
            return
    bucket.append({"value": text, "score": float(score or 0.0), "source_ids": [source_id] if source_id else []})
    bucket.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    del bucket[4:]


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    meta = dict(candidate.get("meta", {}) or {})
    kind = str(candidate.get("kind", "") or "")
    if kind == "trainer_alignment":
        desired = str(meta.get("desired_utterance", "") or "").strip()
        if desired:
            return desired
    return str(candidate.get("anchor_text", "") or "").strip()


def _parse_text_slots(text: str) -> Dict[str, str]:
    norm = _norm(text)
    if not norm:
        return {}
    toks = _tokens(norm)
    out: Dict[str, str] = {}
    if not toks:
        return out

    if toks[:1] == ["there"] and len(toks) >= 3 and toks[1] in COPULAS:
        out["copula"] = toks[1]
        if toks[2] in DETERMINERS and len(toks) >= 4:
            out["deixis"] = toks[2]
            out["entity"] = " ".join(toks[3:])
        else:
            out["entity"] = " ".join(toks[2:])
        return out

    if len(toks) >= 3:
        if toks[0] in DETERMINERS and len(toks) >= 4 and toks[2] in COPULAS:
            out["deixis"] = toks[0]
            out["subject"] = toks[1]
            out["copula"] = toks[2]
            out["attribute"] = " ".join(toks[3:])
            return out
        if toks[1] in COPULAS:
            out["subject"] = toks[0]
            out["copula"] = toks[1]
            out["attribute"] = " ".join(toks[2:])
            return out
        if toks[0] in DETERMINERS and len(toks) >= 3:
            out["deixis"] = toks[0]
            out["subject"] = toks[1]
            out["action"] = " ".join(toks[2:])
            return out
        out["subject"] = toks[0]
        out["action"] = " ".join(toks[1:])
    return out


def _intent_from_pattern(pattern_type: str) -> str:
    p = str(pattern_type or "").strip().lower()
    if p == "assert_attribute":
        return "assert_attribute"
    if p == "assert_existence":
        return "assert_existence"
    if p in {"social_redirect", "request_action", "need_action", "query_need_action", "preference_action", "action_relation"}:
        return "action_assertion"
    if p == "question_about":
        return "question_about"
    return ""


def build_forge_workspace(*, query_type: str, focus_tokens: Sequence[str], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    focus = [str(t or "").strip().lower() for t in (focus_tokens or []) if str(t or "").strip()]
    slots: Dict[str, List[Dict[str, Any]]] = {}
    intent_votes: Dict[str, float] = {}
    seen_texts: List[str] = []

    for candidate in list(candidates or [])[:8]:
        score = float(candidate.get("score", 0.0) or 0.0)
        cell_id = str(candidate.get("cell_id", "") or "")
        kind = str(candidate.get("kind", "") or "")
        meta = dict(candidate.get("meta", {}) or {})
        text = _candidate_text(candidate)
        if text:
            seen_texts.append(text)

        pattern_type = str(meta.get("pattern_type", "") or "")
        intent = _intent_from_pattern(pattern_type)
        if intent:
            intent_votes[intent] = intent_votes.get(intent, 0.0) + max(0.1, score)

        for slot_name, slot_value in dict(meta.get("slots", {}) or {}).items():
            if slot_name in {"subject", "attribute", "copula", "deixis", "entity", "action", "person", "location", "focus", "target", "object", "urgency", "need_type"}:
                _add_slot(slots, slot_name, slot_value, score, cell_id)

        parsed = _parse_text_slots(text)
        for slot_name, slot_value in parsed.items():
            _add_slot(slots, slot_name, slot_value, score * 0.92, cell_id)

        if kind == "trainer_alignment":
            intent_votes["assert_attribute"] = intent_votes.get("assert_attribute", 0.0) + 0.12

    if focus:
        if not slots.get("subject"):
            _add_slot(slots, "subject", focus[0], 0.32, "focus")
        if not slots.get("entity"):
            _add_slot(slots, "entity", focus[0], 0.28, "focus")
        if not slots.get("focus"):
            _add_slot(slots, "focus", " ".join(focus[:3]), 0.24, "focus")

    if not slots.get("copula"):
        _add_slot(slots, "copula", "is", 0.20, "default")

    if not intent_votes:
        if query_type == "what_does":
            intent_votes["action_assertion"] = 0.4
        elif query_type in {"what_is", "statement"}:
            if slots.get("attribute"):
                intent_votes["assert_attribute"] = 0.4
            else:
                intent_votes["assert_existence"] = 0.3
        else:
            intent_votes["assert_attribute"] = 0.2

    intent = max(intent_votes.items(), key=lambda item: item[1])[0] if intent_votes else "assert_attribute"
    return {
        "intent": intent,
        "query_type": str(query_type or "question_generic"),
        "focus_tokens": focus,
        "slots": slots,
        "source_texts": seen_texts[:8],
        "intent_votes": intent_votes,
    }


def _top_values(slots: Mapping[str, List[Dict[str, Any]]], name: str, default: Sequence[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    values = list(slots.get(name, []) or [])
    if values:
        return values[:2]
    return list(default or [])


def _make_candidate(text: str, score: float, parts: Mapping[str, Any], source_ids: Sequence[str]) -> Dict[str, Any]:
    cleaned = " ".join(str(text or "").strip().split())
    if cleaned:
        cleaned = cleaned[:1].upper() + cleaned[1:]
        if not cleaned.endswith("."):
            cleaned += "."
    return {
        "text": cleaned,
        "score": round(float(score or 0.0), 6),
        "parts": dict(parts or {}),
        "source_ids": [str(s or "") for s in source_ids if str(s or "")][:8],
    }


def forge_from_workspace(workspace: Mapping[str, Any]) -> Dict[str, Any]:
    intent = str(workspace.get("intent", "assert_attribute") or "assert_attribute")
    slots = dict(workspace.get("slots", {}) or {})
    focus = list(workspace.get("focus_tokens", []) or [])
    built: List[Dict[str, Any]] = []

    if intent == "assert_attribute":
        subjects = _top_values(slots, "subject", default=[{"value": focus[0], "score": 0.24, "source_ids": ["focus"]}] if focus else [])
        attributes = _top_values(slots, "attribute")
        copulas = _top_values(slots, "copula", default=[{"value": "is", "score": 0.2, "source_ids": ["default"]}])
        deixis_values = _top_values(slots, "deixis") + [{"value": "", "score": 0.0, "source_ids": []}]
        for subj in subjects:
            for attr in attributes[:2]:
                for cop in copulas[:1]:
                    for de in deixis_values[:2]:
                        subject = str(subj.get("value", "") or "").strip()
                        attribute = str(attr.get("value", "") or "").strip()
                        copula = str(cop.get("value", "is") or "is").strip() or "is"
                        deixis = str(de.get("value", "") or "").strip()
                        if not subject or not attribute:
                            continue
                        subject_phrase = " ".join(p for p in [deixis, subject] if p).strip()
                        base_score = float(subj.get("score", 0.0) or 0.0) + float(attr.get("score", 0.0) or 0.0) + float(cop.get("score", 0.0) or 0.0)
                        if not deixis:
                            article = "" if _is_pluralish(subject) else _article_for(subject)
                            if article:
                                built.append(_make_candidate(f"{article} {subject} {copula} {attribute}", base_score + 0.07, {
                                    "intent": intent, "subject": subject, "attribute": attribute, "copula": copula, "deixis": article
                                }, list(subj.get("source_ids", [])) + list(attr.get("source_ids", [])) + list(cop.get("source_ids", []))))
                        built.append(_make_candidate(f"{subject_phrase} {copula} {attribute}", base_score + 0.12 + (0.05 if deixis else 0.0), {
                            "intent": intent, "subject": subject, "attribute": attribute, "copula": copula, "deixis": deixis
                        }, list(subj.get("source_ids", [])) + list(attr.get("source_ids", [])) + list(cop.get("source_ids", [])) + list(de.get("source_ids", []))))

    elif intent == "assert_existence":
        entities = _top_values(slots, "entity") or _top_values(slots, "subject", default=[{"value": focus[0], "score": 0.24, "source_ids": ["focus"]}] if focus else [])
        copulas = _top_values(slots, "copula", default=[{"value": "is", "score": 0.2, "source_ids": ["default"]}])
        deixis_values = _top_values(slots, "deixis") + [{"value": "", "score": 0.0, "source_ids": []}]
        for ent in entities[:2]:
            for cop in copulas[:1]:
                for de in deixis_values[:2]:
                    entity = str(ent.get("value", "") or "").strip()
                    copula = str(cop.get("value", "is") or "is").strip() or "is"
                    deixis = str(de.get("value", "") or "").strip()
                    if not entity:
                        continue
                    entity_phrase = " ".join(p for p in [deixis, entity] if p).strip() if deixis else entity
                    built.append(_make_candidate(f"there {copula} {entity_phrase}", float(ent.get("score", 0.0) or 0.0) + float(cop.get("score", 0.0) or 0.0) + 0.18, {
                        "intent": intent, "entity": entity, "copula": copula, "deixis": deixis
                    }, list(ent.get("source_ids", [])) + list(cop.get("source_ids", [])) + list(de.get("source_ids", []))))

    elif intent == "action_assertion":
        subjects = _top_values(slots, "subject", default=[{"value": focus[0], "score": 0.24, "source_ids": ["focus"]}] if focus else [])
        actions = _top_values(slots, "action") or _top_values(slots, "attribute")
        deixis_values = _top_values(slots, "deixis") + [{"value": "", "score": 0.0, "source_ids": []}]
        for subj in subjects[:2]:
            for act in actions[:2]:
                for de in deixis_values[:2]:
                    subject = str(subj.get("value", "") or "").strip()
                    action = str(act.get("value", "") or "").strip()
                    deixis = str(de.get("value", "") or "").strip()
                    if not subject or not action:
                        continue
                    subject_phrase = " ".join(p for p in [deixis, subject] if p).strip() if deixis else subject
                    built.append(_make_candidate(f"{subject_phrase} {action}", float(subj.get("score", 0.0) or 0.0) + float(act.get("score", 0.0) or 0.0) + 0.14, {
                        "intent": intent, "subject": subject, "action": action, "deixis": deixis
                    }, list(subj.get("source_ids", [])) + list(act.get("source_ids", [])) + list(de.get("source_ids", []))))

    elif intent == "question_about":
        focuses = _top_values(slots, "focus", default=[{"value": " ".join(focus[:3]), "score": 0.24, "source_ids": ["focus"]}] if focus else [])
        for item in focuses[:2]:
            built.append(_make_candidate(f"the question is about {item.get('value', '')}", float(item.get("score", 0.0) or 0.0) + 0.12, {
                "intent": intent, "focus": str(item.get("value", "") or "")
            }, list(item.get("source_ids", []))))

    # dedupe and rank
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for cand in built:
        key = _norm(str(cand.get("text", "") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    deduped.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    chosen = deduped[0] if deduped else {}
    confidence = 0.0
    if chosen:
        raw_score = float(chosen.get("score", 0.0) or 0.0)
        confidence = max(0.52, min(0.91, 0.55 + (raw_score * 0.12)))
        chosen = dict(chosen)
        chosen["confidence"] = round(confidence, 6)
    return {
        "intent": intent,
        "slots": slots,
        "candidates": deduped[:6],
        "chosen": chosen,
    }
