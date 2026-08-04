from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping

from microbrain.orchestrator.neuron_base import Event

SCHEMA_VER = "base.object.v1"
ROOT_TYPE = "base.object"

OBJECT_KINDS = {
    "base.object",
    "scene.object",
    "context.object",
    "event.object",
    "entity.object",
    "state.object",
    "action.object",
    "utterance.object",
    "visual.object",
    "auditory.object",
    "feedback.object",
    "internal_state.object",
    "drive.object",
    "hormone.object",
    "memory.object",
    "scene.exp",
    "question.object",
}

TEXT_STATE_HINTS = {
    "fast", "slow", "warm", "cold", "safe", "unsafe", "loud", "quiet",
    "low", "high", "good", "bad", "happy", "sad", "angry", "tired",
    "bored", "hungry", "full", "bright", "dark", "near", "far",
}

TEXT_ACTION_HINTS = {
    "go", "going", "went", "move", "moving", "moved", "run", "running",
    "look", "looking", "read", "reading", "sleep", "sleeping", "charge",
    "charging", "say", "says", "said", "reply", "ask", "asking", "want",
    "wants", "need", "needs", "touch", "feel", "feels", "grab", "hold",
}

TEXT_STOP_HINTS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "has", "have", "i", "if", "in", "is", "it", "me", "my", "of",
    "or", "our", "so", "that", "the", "this", "to", "was", "we", "what",
    "when", "where", "who", "why", "you", "your",
}

GREETING_HINTS = {"hi", "hello", "hey", "yo", "moin", "morning", "evening", "afternoon"}


def clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def stable_digest(data: Any, *, size: int = 12) -> str:
    try:
        raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        raw = repr(data)
    return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=size).hexdigest()


def make_object_id(prefix: str, data: Any) -> str:
    prefix = re.sub(r"[^a-z0-9_.:-]+", "_", str(prefix or "object").lower()).strip("_:") or "object"
    return f"{prefix}:{stable_digest(data)}"


def text_from_payload(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("text", "") or "").strip()
    if isinstance(payload, str):
        return payload.strip()
    return ""


def raw_meta_from_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("raw_meta"), Mapping):
        return dict(payload.get("raw_meta") or {})
    return {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def infer_grammar_roles(text: str) -> Dict[str, Any]:
    """Return lightweight but structure-aware language roles.

    The base object keeps only a compact slice of the richer parser output so
    scene objects can benefit from English word order without turning every
    utterance into a giant parse dump. Unknown words remain candidates rather
    than being forced into one permanent part of speech.
    """
    tokens = tokenize(text)
    try:
        from microbrain.language_scaffold import analyze_english_structure

        structure = analyze_english_structure(text)
        role_candidates = list(structure.get("role_candidates", []) or [])
        noun_like: list[str] = []
        verb_like: list[str] = []
        adjective_like: list[str] = []
        for candidate in role_candidates:
            if not isinstance(candidate, Mapping):
                continue
            norm = str(candidate.get("norm", "") or "").strip().lower()
            best_role = str(candidate.get("best_role", "") or "").strip().lower()
            if not norm:
                continue
            if best_role in {"noun", "proper_noun", "pronoun"} and norm not in noun_like:
                noun_like.append(norm)
            elif best_role == "verb" and norm not in verb_like:
                verb_like.append(norm)
            elif best_role == "adjective" and norm not in adjective_like:
                adjective_like.append(norm)

        compact_candidates = []
        for candidate in role_candidates[:16]:
            if not isinstance(candidate, Mapping):
                continue
            compact_candidates.append({
                "token": str(candidate.get("norm", "") or candidate.get("text", "") or ""),
                "best_role": str(candidate.get("best_role", "") or ""),
                "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                "alternatives": [
                    {"role": str(item.get("role", "") or ""), "score": float(item.get("score", 0.0) or 0.0)}
                    for item in list(candidate.get("candidates", []) or [])[:3]
                    if isinstance(item, Mapping)
                ],
            })

        return {
            "tokens": tokens,
            "noun_like": noun_like[:16],
            "verb_like": verb_like[:16],
            "adjective_like": adjective_like[:16],
            "role_candidates": compact_candidates,
            "best_clause": dict(structure.get("best_clause", {}) or {}),
        }
    except Exception:
        # Keep startup and event framing resilient if the richer language layer
        # is unavailable for any reason.
        state_words: list[str] = []
        action_words: list[str] = []
        entity_words: list[str] = []
        for tok in tokens:
            if tok in TEXT_STATE_HINTS or tok.endswith(("able", "ful", "less", "ous", "ive")):
                if tok not in state_words:
                    state_words.append(tok)
                continue
            if tok in TEXT_ACTION_HINTS or tok.endswith(("ing", "ed")):
                if tok not in action_words:
                    action_words.append(tok)
                continue
            if len(tok) >= 3 and tok not in TEXT_STOP_HINTS:
                if tok not in entity_words:
                    entity_words.append(tok)
        return {
            "tokens": tokens,
            "noun_like": entity_words[:16],
            "verb_like": action_words[:16],
            "adjective_like": state_words[:16],
            "role_candidates": [],
            "best_clause": {},
        }


def infer_text_classifiers(
    text: str,
    raw_meta: Mapping[str, Any] | None = None,
    grammar_roles: Mapping[str, Any] | None = None,
) -> list[str]:
    raw_meta = raw_meta or {}
    tokens = tokenize(text)
    lowered = " ".join(tokens)
    classifiers: list[str] = ["utterance"]
    if any(tok in GREETING_HINTS for tok in tokens) or lowered in {"good morning", "good evening", "good afternoon"}:
        classifiers.append("social_greeting")
    roles = dict(grammar_roles or infer_grammar_roles(text))
    for state in roles.get("adjective_like", []):
        classifiers.append(f"state.{state}")

    try:
        accent_value = float(raw_meta.get("accent_value", 0.0) or 0.0)
    except Exception:
        accent_value = 0.0
    if accent_value > 0:
        classifiers.append("tone.positive_emphasis")
    elif accent_value < 0:
        classifiers.append("tone.negative_correction")
    return dedupe(classifiers)


def dedupe(items: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


@dataclass
class BaseObjectFrame:
    object_id: str
    kind: str
    created_at: float
    updated_at: float
    root_type: str = ROOT_TYPE
    schema_ver: str = SCHEMA_VER
    time_window: Dict[str, Any] = field(default_factory=dict)
    source_event: Dict[str, Any] = field(default_factory=dict)
    modalities: Dict[str, Any] = field(default_factory=dict)
    classifiers: list[str] = field(default_factory=list)
    grammar_roles: Dict[str, Any] = field(default_factory=dict)
    links: Dict[str, Any] = field(default_factory=dict)
    salience: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = normalize_kind(data.get("kind", "base.object"))
        data["classifiers"] = dedupe(str(x) for x in data.get("classifiers", []) if str(x or "").strip())
        return data


def normalize_kind(kind: str) -> str:
    k = str(kind or "base.object").strip().lower()
    return k if k in OBJECT_KINDS else "base.object"


def event_source_packet(event: Event) -> Dict[str, Any]:
    return {
        "topic": event.topic,
        "source": event.source,
        "correlation_id": event.correlation_id,
        "event_ts": float(event.timestamp),
        "meta": dict(event.meta or {}),
    }


def build_event_object(event: Event, *, internal_state: Mapping[str, Any] | None = None) -> Dict[str, Any] | None:
    payload = event.payload if isinstance(event.payload, Mapping) else {"value": event.payload}
    raw_meta = raw_meta_from_payload(payload)
    now = time.time()

    # Do not create mind-objects for explicit control/UI/status traffic.
    meta = dict(event.meta or {})
    if meta.get("control") is True or meta.get("cognitive_visible") is False or meta.get("store_in_memory") is False:
        return None
    if str(event.topic or "").startswith("ui/") or str(event.topic or "").startswith("control/"):
        return None
    if isinstance(payload, Mapping) and (payload.get("cognitive_visible") is False or payload.get("store_in_memory") is False):
        return None

    kind = "event.object"
    modalities: Dict[str, Any] = {}
    classifiers: list[str] = []
    grammar_roles: Dict[str, Any] = {}
    salience: Dict[str, Any] = {}

    if event.topic in {"percept/text", "act/speech"}:
        text = text_from_payload(event.payload)
        if not text or text.lstrip().startswith("/"):
            return None
        kind = "utterance.object"
        role = "assistant" if event.topic == "act/speech" else str(payload.get("source", "user") or "user")
        modalities["text"] = {
            "text": text,
            "role": role,
            "channel": str(payload.get("channel", raw_meta.get("channel", "")) or ""),
            "transport_source": str(raw_meta.get("transport_source", payload.get("source", role)) or role),
            "accent": {
                "value": raw_meta.get("accent_value", None),
                "direction": raw_meta.get("accent_direction", None),
                "positive": raw_meta.get("accent_positive", None),
                "negative_severity": raw_meta.get("accent_negative_severity", None),
                "tone_label": raw_meta.get("tone_label", None),
            },
        }
        grammar_roles = infer_grammar_roles(text)
        classifiers.extend(infer_text_classifiers(text, raw_meta, grammar_roles))
        if raw_meta.get("salience_delta") is not None or raw_meta.get("preference_delta") is not None:
            salience["textual_accent"] = {
                "salience_delta": raw_meta.get("salience_delta"),
                "preference_delta": raw_meta.get("preference_delta"),
                "correction_severity": raw_meta.get("accent_negative_severity"),
            }

    elif event.topic == "percept/vision":
        kind = "visual.object"
        modalities["visual"] = {
            "description": str(payload.get("description", "") or ""),
            "objects": list(payload.get("objects", []) or []) if isinstance(payload.get("objects", []), list) else [payload.get("objects")],
            "data_ref": str(payload.get("data_ref", "") or ""),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "sensor": str(meta.get("sensor", payload.get("sensor", "vision")) or "vision"),
            "focus": dict(payload.get("focus", {}) or {}) if isinstance(payload.get("focus", {}), Mapping) else {},
        }
        classifiers.append("visual_percept")
        sensor = str(meta.get("sensor", "") or "").strip()
        if sensor:
            classifiers.append(f"sensor.{sensor}")

    elif event.topic == "vision/object_delta":
        kind = "visual.object"
        deltas = list(payload.get("deltas", []) or []) if isinstance(payload.get("deltas", []), list) else []
        modalities["visual_delta"] = {
            "schema": str(payload.get("schema", "vision.object_delta.v1") or "vision.object_delta.v1"),
            "scene_ref": str(payload.get("scene_ref", "") or ""),
            "text": str(payload.get("text", "") or ""),
            "delta_count": payload.get("delta_count", len(deltas)),
            "memory_candidate": bool(payload.get("memory_candidate", False)),
            "deltas": deltas[:8],
            "spatial": dict(payload.get("spatial", {}) or {}) if isinstance(payload.get("spatial", {}), Mapping) else None,
            "image_ref": str(payload.get("image_ref", "") or ""),
            "image_ref_policy": str(payload.get("image_ref_policy", "reference_only_do_not_hardsave_by_default") or ""),
        }
        classifiers.extend(["visual_delta", "object_delta"])
        if bool(payload.get("memory_candidate", False)):
            classifiers.append("memory_candidate")
        if isinstance(payload.get("spatial"), Mapping):
            classifiers.append("spatial_attached")

    elif event.topic == "percept/audio":
        kind = "auditory.object"
        modalities["auditory"] = dict(payload)
        classifiers.append("auditory_percept")

    elif event.topic == "percept/touch":
        kind = "feedback.object"
        modalities["touch"] = dict(payload)
        classifiers.append("touch_feedback")

    elif event.topic == "vision/proto_object":
        kind = "entity.object"
        modalities["visual"] = dict(payload)
        classifiers.extend(["visual_proto_object", str(payload.get("status", "unknown") or "unknown")])

    else:
        modalities["event"] = dict(payload)

    if internal_state:
        modalities["internal"] = dict(internal_state)

    id_basis = {
        "topic": event.topic,
        "correlation_id": event.correlation_id,
        "timestamp_bucket": int(float(event.timestamp) * 1000),
        "kind": kind,
        "payload_digest": stable_digest(payload, size=8),
    }
    frame = BaseObjectFrame(
        object_id=make_object_id(kind.replace(".object", ""), id_basis),
        kind=kind,
        created_at=now,
        updated_at=now,
        time_window={"start": float(event.timestamp), "end": now},
        source_event=event_source_packet(event),
        modalities=modalities,
        classifiers=classifiers,
        grammar_roles=grammar_roles,
        links={"source_event": event.correlation_id},
        salience=salience,
        state={},
        meta={"first_pass": True},
    )
    return frame.to_dict()


def build_scene_object(
    objects: list[Mapping[str, Any]],
    *,
    internal_state: Mapping[str, Any] | None = None,
    previous_scene_id: str = "",
) -> Dict[str, Any]:
    now = time.time()
    clean_objects = [dict(o) for o in objects if isinstance(o, Mapping) and str(o.get("object_id", "") or "")]
    object_ids = [str(o.get("object_id")) for o in clean_objects]
    kinds = dedupe(str(o.get("kind", "base.object") or "base.object") for o in clean_objects)
    modalities_present: list[str] = []
    classifiers: list[str] = ["current_context", "active_scene"]
    start = now
    for obj in clean_objects:
        tw = obj.get("time_window", {}) if isinstance(obj.get("time_window", {}), Mapping) else {}
        try:
            start = min(start, float(tw.get("start", obj.get("created_at", now)) or now))
        except Exception:
            pass
        for mod in (obj.get("modalities", {}) or {}).keys():
            if mod not in modalities_present:
                modalities_present.append(str(mod))
        for cls in list(obj.get("classifiers", []) or []):
            if str(cls or "").strip():
                classifiers.append(str(cls).strip())

    scene_basis = {"objects": object_ids[-16:], "previous": previous_scene_id, "time_bucket": int(now // 5)}
    return BaseObjectFrame(
        object_id=make_object_id("scene", scene_basis),
        kind="scene.object",
        created_at=start,
        updated_at=now,
        time_window={"start": start, "end": now},
        source_event={"topic": "object/scene", "source": "base_object_frame_neuron"},
        modalities={
            "scene": {
                "object_ids": object_ids[-32:],
                "object_kinds": kinds,
                "modalities_present": modalities_present,
            },
            "internal": dict(internal_state or {}),
        },
        classifiers=dedupe(classifiers),
        grammar_roles={},
        links={"contains": object_ids[-32:], "previous_scene": previous_scene_id} if previous_scene_id else {"contains": object_ids[-32:]},
        salience={},
        state={"active": True, "event_count": len(object_ids)},
        meta={"first_pass": True, "note": "context == active scene object"},
    ).to_dict()


def scene_signature(scene: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a compact comparison signature for a scene.

    This is not a memory record. It is a small parser/comparator shape used by
    expectation organs to ask: "does the current scene look like the expected
    scene?"  Object IDs are intentionally avoided because they are often
    event/time specific.
    """
    if not isinstance(scene, Mapping):
        return {"classifiers": [], "kinds": [], "modalities": [], "event_count": 0}

    mods = scene.get("modalities", {}) if isinstance(scene.get("modalities", {}), Mapping) else {}
    scene_mod = mods.get("scene", {}) if isinstance(mods.get("scene", {}), Mapping) else {}
    classifiers = [
        str(c).strip()
        for c in list(scene.get("classifiers", []) or [])
        if str(c or "").strip() and str(c).strip() not in {"current_context", "active_scene"}
    ]
    kinds = [str(k).strip() for k in list(scene_mod.get("object_kinds", []) or []) if str(k or "").strip()]
    modalities = [str(m).strip() for m in list(scene_mod.get("modalities_present", []) or []) if str(m or "").strip()]
    try:
        event_count = int((scene.get("state", {}) or {}).get("event_count", len(scene_mod.get("object_ids", []) or [])) or 0)
    except Exception:
        event_count = 0

    return {
        "classifiers": sorted(dedupe(classifiers))[:64],
        "kinds": sorted(dedupe(kinds))[:32],
        "modalities": sorted(dedupe(modalities))[:16],
        "event_count": event_count,
    }


def diff_scene_signatures(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare two scene signatures and return a small delta packet."""
    exp_cls = set(str(x) for x in expected.get("classifiers", []) or [])
    obs_cls = set(str(x) for x in observed.get("classifiers", []) or [])
    exp_kinds = set(str(x) for x in expected.get("kinds", []) or [])
    obs_kinds = set(str(x) for x in observed.get("kinds", []) or [])
    exp_mods = set(str(x) for x in expected.get("modalities", []) or [])
    obs_mods = set(str(x) for x in observed.get("modalities", []) or [])

    missing_cls = sorted(exp_cls - obs_cls)
    added_cls = sorted(obs_cls - exp_cls)
    missing_kinds = sorted(exp_kinds - obs_kinds)
    added_kinds = sorted(obs_kinds - exp_kinds)
    missing_mods = sorted(exp_mods - obs_mods)
    added_mods = sorted(obs_mods - exp_mods)

    exp_count = int(expected.get("event_count", 0) or 0)
    obs_count = int(observed.get("event_count", 0) or 0)
    count_delta = obs_count - exp_count

    base = (len(missing_cls) * 0.10) + (len(added_cls) * 0.08)
    base += (len(missing_kinds) + len(added_kinds)) * 0.08
    base += (len(missing_mods) + len(added_mods)) * 0.06
    if exp_count or obs_count:
        base += min(0.22, abs(count_delta) / max(4.0, float(max(exp_count, obs_count, 1))) * 0.22)
    magnitude = clamp_float(base, 0.0, 1.0)

    return {
        "magnitude": round(magnitude, 4),
        "missing_classifiers": missing_cls[:16],
        "added_classifiers": added_cls[:16],
        "missing_kinds": missing_kinds[:8],
        "added_kinds": added_kinds[:8],
        "missing_modalities": missing_mods[:8],
        "added_modalities": added_mods[:8],
        "event_count_delta": count_delta,
        "changed": magnitude >= 0.18,
    }


def build_scene_expectation_object(
    prior_scene: Mapping[str, Any],
    *,
    observed_at: float | None = None,
    place_key: str = "default",
) -> Dict[str, Any]:
    """Build an ephemeral scene.exp from a previous scene plus current time.

    `scene.exp` is a parser/prediction artifact. It should not be written as
    durable memory by default.
    """
    now = float(observed_at or time.time())
    prior_id = str(prior_scene.get("object_id", "") or "") if isinstance(prior_scene, Mapping) else ""
    prior_sig = scene_signature(prior_scene)
    basis = {"prior_scene_id": prior_id, "place_key": place_key, "time_bucket": int(now // 60)}
    return BaseObjectFrame(
        object_id=make_object_id("scene.exp", basis),
        kind="scene.exp",
        created_at=now,
        updated_at=now,
        time_window={"start": now, "end": now},
        source_event={"topic": "scene/expectation", "source": "scene_expectation_neuron"},
        modalities={
            "expectation": {
                "basis_scene_id": prior_id,
                "place_key": place_key,
                "expected_signature": prior_sig,
            }
        },
        classifiers=["scene_expectation", "ephemeral_prediction"],
        links={"basis_scene": prior_id} if prior_id else {},
        salience={},
        state={"ephemeral": True, "durable_memory": False},
        meta={"note": "scene.obj + time = scene.exp; compare, then discard"},
    ).to_dict()


def build_unresolved_question_object(
    *,
    question: str,
    expected_scene_id: str = "",
    observed_scene_id: str = "",
    place_key: str = "default",
    delta: Mapping[str, Any] | None = None,
    salience: float = 0.0,
    expires_at: float | None = None,
) -> Dict[str, Any]:
    now = time.time()
    basis = {
        "question": question,
        "expected": expected_scene_id,
        "observed": observed_scene_id,
        "place_key": place_key,
        "day": int(now // 86400),
    }
    return BaseObjectFrame(
        object_id=make_object_id("question", basis),
        kind="question.object",
        created_at=now,
        updated_at=now,
        time_window={"start": now, "end": float(expires_at or (now + 86400.0))},
        source_event={"topic": "question/unresolved", "source": "scene_expectation_neuron"},
        modalities={
            "question": {
                "text": question,
                "place_key": place_key,
                "delta": dict(delta or {}),
            }
        },
        classifiers=["unresolved_question", "scene_delta", "why_question"],
        links={
            "expected_scene": expected_scene_id,
            "observed_scene": observed_scene_id,
        },
        salience={"importance": clamp_float(salience), "novelty": clamp_float(salience)},
        state={"resolved": False, "parked": True, "expires_at": float(expires_at or (now + 86400.0))},
        meta={"note": "parked briefly; promote only if answer matters"},
    ).to_dict()
