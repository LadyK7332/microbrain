# microbrain/babble_backend.py
from __future__ import annotations

import random
import time
from typing import Any, Dict

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
