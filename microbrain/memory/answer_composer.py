from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.utils.memdir import resolve_memdir_ctx

TOKEN_RE = re.compile(r"[a-z0-9']+")
QUESTION_STOPWORDS = {
    'what', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'a', 'an', 'the',
    'to', 'of', 'for', 'about', 'can', 'you', 'me', 'tell', 'give', 'explain',
    'why', 'how', 'when', 'where', 'who', 'whom', 'which'
}


def _norm(text: str) -> str:
    return ' '.join(TOKEN_RE.findall((text or '').lower())).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall((text or '').lower()) if t]


def _focus_tokens(text: str) -> List[str]:
    toks = _tokenize(text)
    focus = [t for t in toks if t not in QUESTION_STOPWORDS]
    return focus or toks


def _extract_body(hit: Dict[str, Any]) -> str:
    anchor = hit.get('anchor', {}) if isinstance(hit.get('anchor', {}), dict) else {}
    anchor_text = str(hit.get('anchor_text', '') or anchor.get('ref', '') or '').strip()
    refs = hit.get('refs', []) if isinstance(hit.get('refs', []), list) else []
    if anchor_text:
        return anchor_text
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return ''


def _dedupe_keep_order(items: Sequence[str], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = _norm(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


async def get_mem_cell_store(ctx) -> Optional[MemCellStore]:
    store = await ctx.get_kv('memory:mem_cell_store', None)
    if store is not None:
        return store
    try:
        memdir = await resolve_memdir_ctx(ctx, fallback=r'Z:\memory')
        store = MemCellStore(memdir)
        await ctx.set_kv('memory:mem_cell_store', store)
        return store
    except Exception:
        return None


async def compose_answer_from_memory(ctx, query_text: str) -> Dict[str, Any]:
    query_text = str(query_text or '').strip()
    norm_q = _norm(query_text)
    focus = _focus_tokens(query_text)
    store = await get_mem_cell_store(ctx)
    if store is None:
        return {'ok': False, 'reason': 'no_mem_cell_store', 'text': ''}

    search_query = ' '.join(focus) if focus else query_text
    hits = store.search_text_cells(search_query, limit=10, tiers=('long', 'now', 'short'))
    if not hits and search_query != query_text:
        hits = store.search_text_cells(query_text, limit=10, tiers=('long', 'now', 'short'))
    if not hits:
        return {'ok': False, 'reason': 'no_hits', 'text': ''}

    bodies = _dedupe_keep_order([_extract_body(h) for h in hits], limit=10)
    focus_set = set(focus)
    utterances = []
    patterns = []
    related = []
    for hit, body in zip(hits, bodies + [''] * max(0, len(hits) - len(bodies))):
        kind = str(hit.get('kind', '') or '')
        body = body.strip()
        if not body:
            continue
        body_norm = _norm(body)
        if any(tok in body_norm.split() for tok in focus_set):
            if kind == 'utterance_anchor':
                utterances.append(body)
            elif 'pattern' in kind:
                patterns.append(body)
            else:
                related.append(body)
        else:
            related.append(body)

    utterances = _dedupe_keep_order(utterances, limit=4)
    patterns = _dedupe_keep_order(patterns, limit=5)
    related = _dedupe_keep_order(related, limit=5)

    # Heuristic answer assembly
    text = ''
    reason = 'assembled'
    focus_phrase = ' '.join(focus[:3]).strip() or 'that'

    # Prefer direct utterance statements that mention the focus token.
    if utterances:
        chosen = ''
        for u in utterances:
            u_norm = _norm(u)
            if any(f" {tok} " in f" {u_norm} " for tok in focus_set):
                chosen = u.strip()
                break
        chosen = chosen or utterances[0].strip()
        if chosen.endswith('?'):
            chosen = chosen[:-1].strip()
        if chosen:
            lead = chosen[0].upper() + chosen[1:] if chosen else chosen
            if not lead.endswith('.'):
                lead += '.'
            text = f"I remember: {lead}"
            reason = 'utterance_match'

    if not text and patterns:
        joined = ', '.join(patterns[:3])
        text = f"For {focus_phrase}, I recall patterns like: {joined}."
        reason = 'pattern_match'

    if not text and related:
        joined = ', '.join(related[:3])
        text = f"For {focus_phrase}, I recall related anchors: {joined}."
        reason = 'related_match'

    if not text:
        return {'ok': False, 'reason': 'empty_assembly', 'text': ''}

    bundle = {
        'ts': time.time(),
        'query_text': query_text,
        'focus_tokens': focus,
        'hits': hits[:6],
        'utterances': utterances[:4],
        'patterns': patterns[:5],
        'related': related[:5],
        'reason': reason,
        'answer_text': text,
    }
    await ctx.set_kv('composer:last_answer', bundle)
    return {'ok': True, 'reason': reason, 'text': text, 'bundle': bundle}
