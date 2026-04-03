from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower())).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t]


def _focus_tokens(text: str) -> List[str]:
    toks = _tokens(text)
    return [t for t in toks if t not in STOPWORDS]


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
            for hit in mem_cell_store.search_text_cells(q, limit=10, tiers=("long", "now", "short")):
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
                meta = dict(hit.get("meta", {}) or {})
                if qtype == "what_does" and ("pattern" in kind or "utterance" in kind):
                    score += 0.08
                    if any(tok in ACTION_HINTS for tok in _tokens(anchor_text)):
                        score += 0.10
                if qtype == "what_is" and kind == "utterance_anchor":
                    score += 0.07
                candidates.append({
                    "source": "mem_cell",
                    "cell_id": hid,
                    "kind": kind,
                    "anchor_text": anchor_text,
                    "refs": list(hit.get("refs", []) or []),
                    "modalities": mods,
                    "links_explicit": list(hit.get("links_explicit", []) or []),
                    "meta": meta,
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
            "anchor_text": f"power {pct:.1f}% charging {'on' if charging else 'off'} sleep {'on' if sleep else 'off'} maintenance {maint:.2f}",
            "refs": [],
            "modalities": ["maintenance", "power"],
            "links_explicit": [],
            "meta": {"pct": pct, "charging": charging, "sleep": sleep, "maintenance": maint},
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
                "anchor_text": ans,
                "refs": [],
                "modalities": ["thought"],
                "links_explicit": list(thought_path_last.get("recalled_cells", []) or []),
                "meta": dict(thought_path_last),
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


def compose_answer(bundle: Mapping[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    qtype = str(bundle.get("query_type", "question_generic") or "question_generic")
    focus = list(bundle.get("focus_tokens", []) or [])
    candidates = list(bundle.get("candidates", []) or [])
    if qtype == "maintenance_status":
        answer, conf = _compose_maintenance_answer(bundle)
        if answer:
            return answer, conf, {"selected_sources": ["power_state"]}
    if not candidates:
        return "", 0.0, {"selected_sources": []}

    selected = candidates[:5]
    best = selected[0]
    best_text = str(best.get("anchor_text", "") or "").strip()
    candidate_texts = _dedupe_preserve([str(c.get("anchor_text", "") or "").strip() for c in selected])

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
                            return f"A {subject} {' '.join(pred)}.", 0.76, {"selected_sources": [c.get("source") for c in selected]}
                return cleaned[:1].upper() + cleaned[1:] + ("." if not cleaned.endswith(".") else ""), 0.72, {"selected_sources": [c.get("source") for c in selected]}
        if best_text:
            return best_text[:1].upper() + best_text[1:] + ("." if not best_text.endswith(".") else ""), 0.52, {"selected_sources": [c.get("source") for c in selected]}

    if qtype == "what_is":
        if best_text:
            cleaned = best_text.rstrip(" .")
            return cleaned[:1].upper() + cleaned[1:] + ".", 0.68, {"selected_sources": [c.get("source") for c in selected]}

    if best_text:
        cleaned = best_text.rstrip(" .")
        return cleaned[:1].upper() + cleaned[1:] + ".", 0.55, {"selected_sources": [c.get("source") for c in selected]}
    return "", 0.0, {"selected_sources": []}
