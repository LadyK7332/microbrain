from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from microbrain.memory.builder_forge import build_forge_workspace, forge_from_workspace
from microbrain.memory.mem_cell_store import MemCellStore

STOPWORDS = {
    "what", "is", "are", "does", "do", "did", "the", "a", "an", "to", "of", "for", "and", "or",
    "tell", "me", "about", "please", "can", "you", "your", "it", "this", "that", "how", "why",
}
ACTION_HINTS = {
    "is", "are", "was", "were", "be", "being", "been", "does", "do", "did", "has", "have", "had",
    "makes", "make", "made", "moves", "move", "moved", "chops", "chop", "chopped", "cuts", "cut",
    "cutting", "using", "used", "use", "helps", "help", "works", "work", "working",
}
STRUCTURAL_KINDS = {"general_pattern", "compressed_general_pattern", "clause_frame", "learning_frame", "understanding_gap", "trainer_alignment"}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower())).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t]


def _focus_tokens(text: str) -> List[str]:
    toks = _tokens(text)
    return [t for t in toks if t not in STOPWORDS]



def _render_general_pattern(candidate: Mapping[str, Any]) -> str:
    meta = dict(candidate.get("meta", {}) or {})
    pattern_type = str(meta.get("pattern_type", "") or "").strip()
    slots = dict(meta.get("slots", {}) or {})
    if pattern_type == "assert_attribute":
        subject = str(slots.get("subject", "") or "").strip()
        attribute = str(slots.get("attribute", "") or "").strip()
        copula = str(slots.get("copula", "is") or "is").strip() or "is"
        deixis = str(slots.get("deixis", "") or "").strip()
        subject_text = " ".join([p for p in [deixis, subject] if p]).strip()
        if subject_text and attribute:
            return f"{subject_text} {copula} {attribute}".strip()
    if pattern_type == "assert_existence":
        entity = str(slots.get("entity", "") or "").strip()
        copula = str(slots.get("copula", "is") or "is").strip() or "is"
        deixis = str(slots.get("deixis", "") or "").strip()
        entity_text = " ".join([p for p in [deixis, entity] if p]).strip()
        if entity_text:
            return f"There {copula} {entity_text}".strip()
    if pattern_type == "question_about":
        focus = str(slots.get("focus", "") or "").strip()
        if focus:
            return f"The open question is about {focus}".strip()
    if pattern_type == "social_redirect":
        person = str(slots.get("person", "") or "").strip()
        location = str(slots.get("location", "") or "").strip()
        if person and location:
            return f"Ask {person} in {location}".strip()
        if person:
            return f"Ask {person}".strip()
    return str(candidate.get("anchor_text", "") or "").strip()


def _render_candidate(candidate: Mapping[str, Any]) -> str:
    kind = str(candidate.get("kind", "") or "")
    meta = dict(candidate.get("meta", {}) or {})
    if kind in {"general_pattern", "compressed_general_pattern"}:
        return _render_general_pattern(candidate)
    if kind == "clause_frame":
        slots = dict(meta.get("slots", {}) or {})
        subject = str(slots.get("subject", "") or "").strip()
        action = str(slots.get("action", "") or "").strip()
        obj = str(slots.get("object", "") or "").strip()
        complement = str(slots.get("complement", "") or "").strip()
        if str(meta.get("clause_type", "") or "") == "copular" and subject and complement:
            return f"{subject} is {complement}".strip()
        if subject and action:
            return " ".join([p for p in [subject, action, obj] if p]).strip()
    if kind in {"learning_frame", "understanding_gap"}:
        rendered = _render_learning_frame(candidate)
        if rendered:
            return rendered
    if kind == "trainer_alignment":
        desired = str(meta.get("desired_utterance", "") or "").strip()
        if desired:
            return desired
    return str(candidate.get("anchor_text", "") or "").strip()



def _render_learning_frame(candidate: Mapping[str, Any]) -> str:
    meta = dict(candidate.get("meta", {}) or {})
    slots = dict(meta.get("slots", {}) or {})
    pattern_type = str(meta.get("pattern_type", "") or "").strip()
    subject = str(slots.get("subject", "") or slots.get("query_target", "") or "").strip()
    designation = str(slots.get("designation", "") or slots.get("category", "") or "").strip()
    definition = str(slots.get("definition", "") or "").strip()
    subtype_of = str(slots.get("subtype_of", "") or "").strip()
    deictic = bool(slots.get("deictic", False))
    if pattern_type == "definition_question":
        target = str(slots.get("query_target", "") or subject).strip()
        return f"I need a definition for {target}".strip() if target else "I need a definition"
    if pattern_type == "contrast_claim" and subject and definition:
        return f"{subject} is not {definition}".strip()
    if pattern_type == "classification_claim" and subject and designation:
        return f"{subject} is a {designation}".strip()
    if pattern_type == "designation_claim" and subject:
        if definition:
            return f"{subject} is {definition}".strip()
        if designation:
            return f"{subject} is a {designation}".strip()
        if subtype_of:
            return f"{subject} is a kind of {subtype_of}".strip()
    if pattern_type == "definition_claim" and subject:
        if definition:
            return f"{subject} is {definition}".strip()
        if designation:
            return f"{subject} is a {designation}".strip()
    return str(candidate.get("anchor_text", "") or "").strip()


def _learning_frame_matches_focus(candidate: Mapping[str, Any], focus: Sequence[str]) -> bool:
    if not focus:
        return True
    meta = dict(candidate.get("meta", {}) or {})
    slots = dict(meta.get("slots", {}) or {})
    terms = {
        str(slots.get("subject", "") or "").strip().lower(),
        str(slots.get("query_target", "") or "").strip().lower(),
        str(slots.get("designation", "") or "").strip().lower(),
        str(slots.get("category", "") or "").strip().lower(),
    }
    return any(str(tok or "").strip().lower() in terms for tok in focus)


def _is_reading_quote(candidate: Mapping[str, Any]) -> bool:
    kind = str(candidate.get("kind", "") or "")
    transport_source = str(candidate.get("transport_source", "") or "")
    return transport_source == "reading" and kind == "utterance_anchor"


def classify_query(text: str) -> str:
    norm = _norm(text)
    if any(k in norm for k in ("power", "battery", "charge", "charging", "sleeping", "sleep", "maintenance")):
        return "maintenance_status"
    if norm.startswith("what does ") or norm.startswith("what do "):
        return "what_does"
    if norm.startswith("what is ") or norm.startswith("what are "):
        return "what_is"
    if text.strip().endswith("?"):
        return "question_generic"
    return "statement"


def gather_support(
    *,
    query_text: str,
    mem_cell_store: Optional[MemCellStore],
    power_state: Optional[Mapping[str, Any]] = None,
    needs: Optional[Mapping[str, Any]] = None,
    thought_path_last: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    focus = _focus_tokens(query_text)
    qtype = classify_query(query_text)
    candidates: List[Dict[str, Any]] = []
    modalities = set()

    if mem_cell_store is not None:
        queries = []
        if focus:
            queries.extend(focus[:3])
        queries.append(query_text)
        seen = set()
        for q in queries:
            for hit in mem_cell_store.search_text_cells(q, limit=12, tiers=("learned", "long", "hot", "now", "short")):
                hid = str(hit.get("cell_id", "") or "")
                if not hid or hid in seen:
                    continue
                seen.add(hid)
                score = float(hit.get("score", 0.0) or 0.0)
                anchor_text = str(hit.get("anchor_text", "") or "")
                mods = list(hit.get("modalities", []) or [])
                overlap = sum(1 for t in focus if t and t in _tokens(anchor_text))
                if overlap:
                    score += 0.12 * overlap
                kind = str(hit.get("kind", "") or "")
                tier = str(hit.get("tier", "") or "")
                meta = dict(hit.get("meta", {}) or {})
                transport_source = str(meta.get("transport_source", "") or "")
                channel = str(meta.get("channel", "") or "")

                if kind in {"general_pattern", "compressed_general_pattern", "clause_frame", "learning_frame", "understanding_gap"}:
                    pattern_type = str(meta.get("pattern_type", "") or "")
                    score += 0.18
                    if kind == "compressed_general_pattern" or tier == "derived":
                        score += 0.14
                    if kind == "learning_frame":
                        score += 0.18
                        if qtype == "what_is" and pattern_type in ("definition_claim", "classification_claim", "designation_claim"):
                            score += 0.30
                        if qtype == "what_is" and _learning_frame_matches_focus(hit, focus):
                            score += 0.22
                    if kind == "understanding_gap" and qtype == "what_is":
                        score -= 0.04
                    if qtype == "what_is" and pattern_type in ("assert_attribute", "assert_existence"):
                        score += 0.18
                    if qtype == "what_does" and pattern_type in ("assert_attribute", "social_redirect"):
                        score += 0.10
                    if qtype == "question_generic" and pattern_type == "question_about":
                        score += 0.06
                if kind == "trainer_alignment":
                    score += 0.16
                if qtype == "what_does" and ("pattern" in kind or "utterance" in kind):
                    score += 0.08
                    if any(tok in ACTION_HINTS for tok in _tokens(anchor_text)):
                        score += 0.10
                if qtype == "what_is" and kind in {"utterance_anchor", "general_pattern", "compressed_general_pattern", "clause_frame", "trainer_alignment"}:
                    score += 0.07

                # Reading should shape structure more than it quotes surface lines.
                if transport_source == "reading":
                    if kind in {"general_pattern", "compressed_general_pattern", "clause_frame"}:
                        score += 0.10
                    elif kind == "utterance_anchor":
                        score -= 0.18
                if transport_source == "trainer":
                    score += 0.08
                if channel == "reading" and kind == "utterance_anchor":
                    score -= 0.08

                source = "mem_cell_derived" if tier == "derived" else "mem_cell"
                candidates.append({
                    "source": source,
                    "cell_id": hid,
                    "kind": kind,
                    "tier": tier,
                    "anchor_text": anchor_text,
                    "refs": list(hit.get("refs", []) or []),
                    "modalities": mods,
                    "links_explicit": list(hit.get("links_explicit", []) or []),
                    "meta": meta,
                    "transport_source": transport_source,
                    "channel": channel,
                    "score": round(score, 6),
                })
                modalities.update(mods)

    if power_state and qtype == "maintenance_status":
        pct = float(power_state.get("pct", power_state.get("battery_pct", 0.0)) or 0.0)
        charging = bool(power_state.get("charging", False))
        sleep = bool(power_state.get("sleep", False))
        maint = 0.0
        if isinstance(needs, Mapping):
            try:
                maint = float(needs.get("maintenance", 0.0) or 0.0)
            except Exception:
                maint = 0.0
        candidates.append({
            "source": "power_state",
            "cell_id": "power:battery_state",
            "kind": "state_anchor",
            "tier": "state",
            "anchor_text": f"power {pct:.1f}% charging {'on' if charging else 'off'} sleep {'on' if sleep else 'off'} maintenance {maint:.2f}",
            "refs": [],
            "modalities": ["maintenance", "power"],
            "links_explicit": [],
            "meta": {"pct": pct, "charging": charging, "sleep": sleep, "maintenance": maint},
            "transport_source": "state",
            "channel": "internal",
            "score": 1.25,
        })
        modalities.update(["maintenance", "power"])

    if thought_path_last and isinstance(thought_path_last, Mapping):
        ans = str(thought_path_last.get("answer", "") or thought_path_last.get("reply", "") or "").strip()
        if ans:
            candidates.append({
                "source": "thought_path",
                "cell_id": str(thought_path_last.get("trace_id", "thought_path:last") or "thought_path:last"),
                "kind": "thought_path",
                "tier": "trace",
                "anchor_text": ans,
                "refs": [],
                "modalities": ["thought"],
                "links_explicit": list(thought_path_last.get("recalled_cells", []) or []),
                "meta": dict(thought_path_last),
                "transport_source": "thought",
                "channel": "internal",
                "score": 0.40,
            })
            modalities.add("thought")

    candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return {
        "query_text": query_text,
        "query_type": qtype,
        "focus_tokens": focus,
        "modalities": sorted(modalities),
        "candidates": candidates[:12],
    }


def _dedupe_preserve(items: Sequence[str], limit: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _compose_maintenance_answer(bundle: Mapping[str, Any]) -> Tuple[str, float]:
    power_candidate = next((c for c in bundle.get("candidates", []) if str(c.get("source", "")) == "power_state"), None)
    if not power_candidate:
        return "", 0.0
    meta = dict(power_candidate.get("meta", {}) or {})
    pct = float(meta.get("pct", 0.0) or 0.0)
    charging = bool(meta.get("charging", False))
    sleep = bool(meta.get("sleep", False))
    maintenance = float(meta.get("maintenance", 0.0) or 0.0)
    state_parts = [f"Power is at {pct:.1f}%"]
    state_parts.append("charging is on" if charging else "charging is off")
    state_parts.append("sleep is on" if sleep else "sleep is off")
    state_parts.append(f"maintenance pressure is {maintenance:.2f}")
    return ". ".join(state_parts) + ".", 0.92


def _answer_meta(selected: Sequence[Mapping[str, Any]], **extra: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "selected_sources": [str(c.get("source", "") or "") for c in selected],
        "selected_cell_ids": [str(c.get("cell_id", "") or "") for c in selected if str(c.get("cell_id", "") or "")],
        "selected_tiers": [str(c.get("tier", "") or "") for c in selected],
        "selected_transport_sources": [str(c.get("transport_source", "") or "") for c in selected],
    }
    meta.update(extra)
    return meta


def compose_answer(bundle: Mapping[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    qtype = str(bundle.get("query_type", "question_generic") or "question_generic")
    focus = list(bundle.get("focus_tokens", []) or [])
    candidates = list(bundle.get("candidates", []) or [])
    if qtype == "maintenance_status":
        answer, conf = _compose_maintenance_answer(bundle)
        if answer:
            return answer, conf, {"selected_sources": ["power_state"], "selected_cell_ids": ["power:battery_state"]}
    if not candidates:
        return "", 0.0, {"selected_sources": [], "selected_cell_ids": []}

    structural = [c for c in candidates if str(c.get("kind", "") or "") in STRUCTURAL_KINDS]
    non_reading_quotes = [c for c in candidates if not _is_reading_quote(c)]
    selected = (structural[:3] + [c for c in non_reading_quotes if c not in structural][:2])[:5]
    if not selected:
        selected = candidates[:5]
    best = selected[0]
    best_text = _render_candidate(best).strip()
    rendered_selected = [_render_candidate(c) for c in selected]
    candidate_texts = _dedupe_preserve(rendered_selected)

    if qtype == "what_is":
        learning_candidates = [
            c for c in selected
            if str(c.get("kind", "") or "") == "learning_frame"
            and str((dict(c.get("meta", {}) or {}).get("pattern_type", "") or "")) in {"definition_claim", "classification_claim", "designation_claim"}
            and _learning_frame_matches_focus(c, focus)
        ]
        if learning_candidates:
            chosen = learning_candidates[0]
            rendered = _render_learning_frame(chosen).strip().rstrip(" .")
            if rendered:
                return rendered[:1].upper() + rendered[1:] + ".", 0.86, _answer_meta(
                    [chosen],
                    used_learning_frame=True,
                    used_prebuilt_answer=False,
                )

    forge_workspace = build_forge_workspace(query_type=qtype, focus_tokens=focus, candidates=selected or candidates)
    forge_bundle = forge_from_workspace(forge_workspace)
    forge_choice = dict(forge_bundle.get("chosen", {}) or {})
    if forge_choice.get("text"):
        forge_sources = [sid for sid in list(forge_choice.get("source_ids", []) or []) if sid]
        forge_selected = [c for c in selected if str(c.get("cell_id", "") or "") in set(forge_sources)] or selected
        return str(forge_choice.get("text", "") or ""), float(forge_choice.get("confidence", 0.72) or 0.72), _answer_meta(
            forge_selected,
            used_forge=True,
            forge_intent=str(forge_bundle.get("intent", "") or ""),
            forge_workspace=forge_workspace,
            forge_choice=forge_choice,
            used_general_pattern=str(best.get("kind", "") or "") in {"general_pattern", "compressed_general_pattern", "clause_frame", "trainer_alignment"},
            used_compressed=str(best.get("kind", "") or "") == "compressed_general_pattern",
            used_trainer_alignment=str(best.get("kind", "") or "") == "trainer_alignment",
        )

    if str(best.get("kind", "") or "") in {"general_pattern", "compressed_general_pattern", "clause_frame", "trainer_alignment"}:
        rendered = _render_candidate(best)
        if rendered:
            cleaned = rendered.rstrip(" .")
            conf = 0.88 if str(best.get("kind", "") or "") == "compressed_general_pattern" else (0.84 if qtype in ("what_is", "what_does") else 0.74)
            return cleaned[:1].upper() + cleaned[1:] + ".", conf, _answer_meta(
                selected,
                used_general_pattern=True,
                used_compressed=str(best.get("kind", "") or "") == "compressed_general_pattern",
                used_trainer_alignment=str(best.get("kind", "") or "") == "trainer_alignment",
                forge_workspace=forge_workspace,
            )

    if qtype == "what_does":
        for text in candidate_texts:
            toks = _tokens(text)
            if not toks:
                continue
            if any(tok in ACTION_HINTS for tok in toks):
                cleaned = text.rstrip(" .")
                if focus:
                    subject = focus[0]
                    if subject in toks and toks.index(subject) == 1 and toks[0] in {"a", "an", "the", "this", "that"}:
                        pred = toks[2:]
                        if pred:
                            return f"A {subject} {' '.join(pred)}.", 0.76, _answer_meta(selected)
                return cleaned[:1].upper() + cleaned[1:] + ("." if not cleaned.endswith(".") else ""), 0.72, _answer_meta(selected)
        if best_text:
            return best_text[:1].upper() + best_text[1:] + ("." if not best_text.endswith(".") else ""), 0.52, _answer_meta(selected)

    if qtype == "what_is":
        if best_text:
            cleaned = best_text.rstrip(" .")
            return cleaned[:1].upper() + cleaned[1:] + ".", 0.68, _answer_meta(selected)

    if best_text:
        cleaned = best_text.rstrip(" .")
        return cleaned[:1].upper() + cleaned[1:] + ".", 0.55, _answer_meta(selected)
    return "", 0.0, {"selected_sources": [], "selected_cell_ids": []}
