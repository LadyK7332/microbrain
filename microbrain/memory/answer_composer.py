from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from microbrain.memory.builder_forge import build_forge_workspace, forge_from_workspace
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.utils.memdir import resolve_memdir_ctx

TOKEN_RE = re.compile(r"[a-z0-9']+")
QUESTION_STOPWORDS = {
    'what', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'a', 'an', 'the',
    'to', 'of', 'for', 'about', 'can', 'you', 'me', 'tell', 'give', 'explain',
    'why', 'how', 'when', 'where', 'who', 'whom', 'which'
}
STRUCTURAL_KINDS = {'general_pattern', 'compressed_general_pattern', 'thought_template', 'clause_frame', 'trainer_alignment'}


def _norm(text: str) -> str:
    return ' '.join(TOKEN_RE.findall((text or '').lower())).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall((text or '').lower()) if t]


def _focus_tokens(text: str) -> List[str]:
    toks = _tokenize(text)
    focus = [t for t in toks if t not in QUESTION_STOPWORDS]
    return focus or toks


def _extract_body(hit: Dict[str, Any]) -> str:
    kind = str(hit.get('kind', '') or '')
    meta = dict(hit.get('meta', {}) or {})
    if kind == 'trainer_alignment':
        desired = str(meta.get('desired_utterance', '') or '').strip()
        if desired:
            return desired
    anchor = hit.get('anchor', {}) if isinstance(hit.get('anchor', {}), dict) else {}
    anchor_text = str(hit.get('anchor_text', '') or anchor.get('ref', '') or '').strip()
    refs = hit.get('refs', []) if isinstance(hit.get('refs', []), list) else []
    if anchor_text:
        return anchor_text
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return ''



def _render_learning_frame(hit: Mapping[str, Any]) -> str:
    meta = dict(hit.get('meta', {}) or {})
    slots = dict(meta.get('slots', {}) or {})
    pattern_type = str(meta.get('pattern_type', '') or '').strip()
    subject = str(slots.get('subject', '') or slots.get('query_target', '') or '').strip()
    designation = str(slots.get('designation', '') or slots.get('category', '') or '').strip()
    definition = str(slots.get('definition', '') or '').strip()
    subtype_of = str(slots.get('subtype_of', '') or '').strip()
    if pattern_type == 'definition_question':
        target = str(slots.get('query_target', '') or subject).strip()
        return f'I need a definition for {target}'.strip() if target else 'I need a definition'
    if pattern_type == 'contrast_claim' and subject and definition:
        return f'{subject} is not {definition}'.strip()
    if pattern_type == 'classification_claim' and subject and designation:
        return f'{subject} is a {designation}'.strip()
    if pattern_type == 'designation_claim' and subject:
        if definition:
            return f'{subject} is {definition}'.strip()
        if designation:
            return f'{subject} is a {designation}'.strip()
        if subtype_of:
            return f'{subject} is a kind of {subtype_of}'.strip()
    if pattern_type == 'definition_claim' and subject:
        if definition:
            return f'{subject} is {definition}'.strip()
        if designation:
            return f'{subject} is a {designation}'.strip()
    return str(hit.get('anchor_text', '') or '').strip()


def _render_pattern(hit: Dict[str, Any]) -> str:
    meta = dict(hit.get('meta', {}) or {})
    slots = dict(meta.get('slots', {}) or {})
    pattern_type = str(meta.get('pattern_type', '') or '').strip()
    if pattern_type == 'assert_attribute':
        subject = str(slots.get('subject', '') or '').strip()
        attribute = str(slots.get('attribute', '') or '').strip()
        copula = str(slots.get('copula', 'is') or 'is').strip() or 'is'
        deixis = str(slots.get('deixis', '') or '').strip()
        subject_text = ' '.join([p for p in [deixis, subject] if p]).strip()
        if subject_text and attribute:
            return f'{subject_text} {copula} {attribute}'.strip()
    if pattern_type == 'assert_existence':
        entity = str(slots.get('entity', '') or '').strip()
        copula = str(slots.get('copula', 'is') or 'is').strip() or 'is'
        deixis = str(slots.get('deixis', '') or '').strip()
        entity_text = ' '.join([p for p in [deixis, entity] if p]).strip()
        if entity_text:
            return f'There {copula} {entity_text}'.strip()
    if pattern_type in {'need_action', 'query_need_action'}:
        subject = str(slots.get('subject', '') or '').strip() or 'someone'
        action = str(slots.get('action', '') or '').strip()
        urgency = str(slots.get('urgency', '') or '').strip()
        if action:
            return ' '.join([subject, 'needs to', action, urgency]).strip()
    if pattern_type == 'request_action':
        action = str(slots.get('action', '') or '').strip()
        target = str(slots.get('target', '') or '').strip()
        if action:
            return ' '.join(['request to', action, target]).strip()
    if pattern_type == 'preference_action':
        subject = str(slots.get('subject', '') or '').strip() or 'someone'
        preference = str(slots.get('preference', 'likes') or 'likes').strip() or 'likes'
        action = str(slots.get('action', '') or '').strip()
        obj = str(slots.get('object', '') or '').strip()
        if action:
            return ' '.join([subject, preference, 'to', action, obj]).strip()
    if pattern_type == 'action_relation':
        subject = str(slots.get('subject', '') or '').strip()
        action = str(slots.get('action', '') or '').strip()
        obj = str(slots.get('object', '') or '').strip()
        if subject and action:
            return ' '.join([subject, action, obj]).strip()
    if str(hit.get('kind', '') or '') in {'learning_frame', 'understanding_gap'}:
        rendered = _render_learning_frame(hit)
        if rendered:
            return rendered
    if str(hit.get('kind', '') or '') == 'clause_frame':
        clause_type = str(meta.get('clause_type', '') or '')
        subject = str(slots.get('subject', '') or '').strip()
        action = str(slots.get('action', '') or '').strip()
        obj = str(slots.get('object', '') or '').strip()
        complement = str(slots.get('complement', '') or '').strip()
        if clause_type == 'copular' and subject and complement:
            return f'{subject} is {complement}'.strip()
        if subject and action:
            return ' '.join([subject, action, obj]).strip()
    return _extract_body(hit)


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
    hits = store.search_text_cells(search_query, limit=12, tiers=('learned', 'long', 'hot', 'now', 'short'))
    if not hits and search_query != query_text:
        hits = store.search_text_cells(query_text, limit=12, tiers=('learned', 'long', 'hot', 'now', 'short'))
    if not hits:
        return {'ok': False, 'reason': 'no_hits', 'text': ''}

    structural_hits = [h for h in hits if str(h.get('kind', '') or '') in STRUCTURAL_KINDS]
    direct_hits = [
        h for h in hits
        if str(h.get('kind', '') or '') == 'utterance_anchor'
        and str((h.get('meta', {}) if isinstance(h.get('meta', {}), dict) else {}).get('transport_source', '') or '') != 'reading'
    ]
    related_hits = [h for h in hits if h not in structural_hits and h not in direct_hits]

    structural = _dedupe_keep_order([_render_pattern(h) for h in structural_hits], limit=5)
    direct = _dedupe_keep_order([_extract_body(h) for h in direct_hits], limit=4)
    related = _dedupe_keep_order([_extract_body(h) for h in related_hits], limit=5)

    text = ''
    reason = 'assembled'
    focus_phrase = ' '.join(focus[:3]).strip() or 'that'
    selected_ids: List[str] = []

    forge_workspace = build_forge_workspace(query_type='what_is' if query_text.endswith('?') else 'statement', focus_tokens=focus, candidates=hits[:6])
    forge_bundle = forge_from_workspace(forge_workspace)
    forge_choice = dict(forge_bundle.get('chosen', {}) or {})

    if forge_choice.get('text'):
        text = str(forge_choice.get('text', '') or '').strip()
        reason = 'forge_match'
        selected_ids = [str(s or '') for s in list(forge_choice.get('source_ids', []) or []) if str(s or '')]

    if not text and structural:
        chosen = structural[0].strip().rstrip('?').rstrip(' .')
        if chosen:
            text = chosen[:1].upper() + chosen[1:] + '.'
            reason = 'structural_match'
            selected_ids = [str(h.get('cell_id', '') or '') for h in structural_hits[:3] if str(h.get('cell_id', '') or '')]

    if not text and direct:
        chosen = direct[0].strip().rstrip('?').rstrip(' .')
        if chosen:
            text = chosen[:1].upper() + chosen[1:] + "."
            reason = 'direct_match'
            selected_ids = [str(h.get('cell_id', '') or '') for h in direct_hits[:3] if str(h.get('cell_id', '') or '')]

    if not text and related:
        joined = ', '.join(related[:3])
        text = joined
        reason = 'related_match'
        selected_ids = [str(h.get('cell_id', '') or '') for h in related_hits[:3] if str(h.get('cell_id', '') or '')]

    if not text:
        return {'ok': False, 'reason': 'empty_assembly', 'text': ''}

    bundle = {
        'ts': time.time(),
        'query_text': query_text,
        'norm_query': norm_q,
        'focus_tokens': focus,
        'hits': hits[:8],
        'structural': structural[:5],
        'direct': direct[:4],
        'related': related[:5],
        'reason': reason,
        'answer_text': text,
        'selected_cell_ids': selected_ids,
        'forge_workspace': forge_workspace,
        'forge_choice': forge_choice,
    }
    await ctx.set_kv('composer:last_answer', bundle)
    return {'ok': True, 'reason': reason, 'text': text, 'bundle': bundle}
