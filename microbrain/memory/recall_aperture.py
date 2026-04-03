from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def make_recall_key(
    *,
    seed_kind: str,
    seed_id: str | None = None,
    tokens: Iterable[str] | None = None,
    noun_id: str | None = None,
) -> str:
    parts: list[str] = [str(seed_kind or 'unknown').strip().lower() or 'unknown']
    if seed_id:
        parts.append(str(seed_id).strip().lower())
    if noun_id:
        parts.append(str(noun_id).strip().lower())
    if tokens:
        toks = [str(t).strip().lower() for t in tokens if str(t).strip()]
        if toks:
            parts.append(','.join(toks[:6]))
    return '|'.join(parts)


def compute_match_quality(
    scores: Sequence[float],
    *,
    learned_hit_count: int = 0,
    fallback_only: bool = False,
) -> float:
    vals = [max(0.0, float(s)) for s in (scores or [])]
    if not vals:
        return 0.0

    vals = sorted(vals, reverse=True)
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0

    distinctness = top1 / max(top1 + top2, 1e-6)
    strength = top1 / max(top1 + 0.50, 0.50)
    quality = _clamp((0.55 * distinctness) + (0.45 * strength))

    if learned_hit_count <= 0:
        quality = min(quality, 0.18)
    if fallback_only:
        quality = min(quality, 0.15)

    return round(_clamp(quality), 4)


def advance_recall_tracker(
    tracker: Mapping[str, Any] | None,
    *,
    key: str,
    now: float,
    uncertainty: float,
    quality: float,
    revisit_window_s: float = 300.0,
    base_limit: int = 6,
    step: int = 2,
    max_extra: int = 12,
    uncertainty_boost: int = 6,
    failure_boost: int = 4,
    success_quality: float = 0.72,
    prune_limit: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(tracker or {})
    prev = out.get(key, {}) if isinstance(out.get(key, {}), Mapping) else {}

    prev_ts = float(prev.get('ts', 0.0) or 0.0)
    if prev_ts > 0.0 and (now - prev_ts) <= float(revisit_window_s):
        attempts = int(prev.get('attempts', 0) or 0) + 1
    else:
        attempts = 1

    prev_failures = int(prev.get('failures', 0) or 0)
    if float(quality) >= float(success_quality):
        failures = max(0, prev_failures - 1)
    else:
        failures = prev_failures + 1

    active_limit = int(base_limit)
    active_limit += min(int(max_extra), max(0, attempts - 1) * int(step))
    active_limit += int(round(_clamp(uncertainty) * int(uncertainty_boost)))
    active_limit += min(int(failure_boost), max(0, failures))
    active_limit = max(3, active_limit)

    entry = {
        'ts': float(now),
        'attempts': int(attempts),
        'failures': int(failures),
        'quality': round(float(quality), 4),
        'uncertainty': round(_clamp(uncertainty), 4),
        'active_limit': int(active_limit),
    }
    out[key] = entry

    if len(out) > int(prune_limit):
        ordered = sorted(
            ((k, v) for k, v in out.items() if isinstance(v, Mapping)),
            key=lambda kv: float(kv[1].get('ts', 0.0) or 0.0),
            reverse=True,
        )
        keep = {k for k, _ in ordered[: int(prune_limit)]}
        out = {k: v for k, v in out.items() if k in keep}

    return out, entry
