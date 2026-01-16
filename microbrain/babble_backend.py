# microbrain/babble_backend.py
from __future__ import annotations

import random
import time
from typing import Any, Dict

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:]")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

def _join_tokens(tokens: list[str]) -> str:
    # Join tokens with sane spacing (no space before punctuation)
    out: list[str] = []
    for t in tokens:
        if t in ".,!?;:" and out:
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out)

def _weighted_choice(rng: random.Random, items: list[tuple[str, int]]) -> str:
    total = sum(max(0, w) for _, w in items)
    if total <= 0:
        return items[0][0]
    r = rng.randint(1, total)
    acc = 0
    for val, w in items:
        acc += max(0, w)
        if acc >= r:
            return val
    return items[-1][0]

_VOWELS = "aeiou"
_CONS = "bcdfghjklmnpqrstvwxyz"


def _syllable(rng: random.Random) -> str:
    # simple CVC-ish babble
    c1 = rng.choice(_CONS)
    v = rng.choice(_VOWELS)
    c2 = rng.choice(_CONS) if rng.random() < 0.6 else ""
    return c1 + v + c2


async def babble_generate(prompt: str, meta: Dict[str, Any]) -> str:
    """
    Drop-in replacement for llm_generate(prompt, meta).
    Produces short "babble" text so the cognition pipeline can be exercised
    without any external LLM.
    """
    # Gate babble strictly: only when boredom is active AND attention allows it
    boredom_active = bool(meta.get("boredom_active", False))
    allow_babble = bool(meta.get("allow_babble", False))
    if not (boredom_active and allow_babble):
        return ""


    # Make it stable-ish per event, but still lively.
    seed = int(time.time() * 1000) ^ hash(prompt[:120])
    rng = random.Random(seed)

    # If we can find the user's last line, bias toward "copying" it a little
    last_user = ""
    for line in reversed(prompt.splitlines()):
        if line.lower().startswith("user:"):
            last_user = line.split(":", 1)[-1].strip()
            break

    # Mimic corpus (if provided)
    mimic = meta.get("mimic", {}) if isinstance(meta, dict) else {}
    if isinstance(mimic, dict) and (mimic.get("unigrams") or mimic.get("bigrams") or mimic.get("last_user_text")):
        unigrams = mimic.get("unigrams") or {}
        bigrams = mimic.get("bigrams") or {}
        last_user_text = str(mimic.get("last_user_text", "") or "").strip()

        # Build a stable-ish RNG
        seed = int(time.time() * 1000) ^ hash(prompt[:120])
        rng = random.Random(seed)

        # 40%: direct mimic (small clipped imitation)
        if last_user_text and rng.random() < 0.4:
            toks = _tokenize(last_user_text)
            toks = toks[:12]
            if toks:
                # Tiny mutation: swap one token sometimes
                if isinstance(unigrams, dict) and unigrams and rng.random() < 0.25:
                    # pick a "word-ish" token
                    word_items = [(k, int(v)) for k, v in unigrams.items() if k.isalnum()]
                    if word_items:
                        repl = _weighted_choice(rng, word_items)
                        pos = rng.randrange(0, len(toks))
                        toks[pos] = repl
                return _join_tokens(toks)

        # 60%: recombine via bigram walk
        if isinstance(unigrams, dict) and isinstance(bigrams, dict) and unigrams and bigrams:
            # adjacency: t1 -> [(t2, count), ...]
            adj: dict[str, list[tuple[str, int]]] = {}
            for k, v in bigrams.items():
                if not isinstance(k, str):
                    continue
                if "|" not in k:
                    continue
                a, b = k.split("|", 1)
                adj.setdefault(a, []).append((b, int(v) if isinstance(v, (int, float)) else 1))

            start_items = [(k, int(v)) for k, v in unigrams.items() if isinstance(k, str) and k.isalnum()]
            if start_items:
                cur = _weighted_choice(rng, start_items)
                out = [cur]
                target_len = rng.randint(6, 12)
                for _ in range(target_len - 1):
                    nxt_items = adj.get(cur)
                    if not nxt_items:
                        break
                    cur = _weighted_choice(rng, nxt_items)
                    out.append(cur)
                if out:
                    return _join_tokens(out)

    # Build utterance
    n = rng.randint(2, 5)  # # of syllables
    babble = "".join(_syllable(rng) for _ in range(n))

    if last_user:
        # crude imitation: pick a couple letters from the user's line
        letters = [ch.lower() for ch in last_user if ch.isalpha()]
        if letters:
            take = "".join(rng.choice(letters) for _ in range(min(6, len(letters))))
            babble = f"{babble}… {take}?"

    return babble
