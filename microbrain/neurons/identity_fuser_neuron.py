from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from microbrain.orchestrator.debug_utils import is_debug_enabled
from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


def _cos_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    v = float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)) / (na * nb))
    # clamp
    return max(-1.0, min(1.0, v))


class IdentityFuserNeuron(BaseNeuron):
    """
    Cross-modal identity belief: vision proto + audio fingerprint.

    Listens:
      - memory/recall_context   (from ProtoRecallNeuron; contains vision assets with proto_id)
      - memory/audio_engram     (from AudioEngramNeuron; contains fp32)

    Stores:
      - memdir/identity/profiles.json

    Publishes nothing by default; in --debug it prints a live confidence line.
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._memdir: Optional[Path] = None
        self._profiles_path: Optional[Path] = None
        self._profiles: Dict[str, Any] = {}

        self._last_vision_proto: Optional[str] = None
        self._last_audio_fp: Optional[list[float]] = None
        self._last_update_ts: float = 0.0

    async def _ensure_loaded(self, ctx) -> None:
        if self._memdir is None:
            self._memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
            (self._memdir / "identity").mkdir(parents=True, exist_ok=True)
            self._profiles_path = self._memdir / "identity" / "profiles.json"

            if self._profiles_path.exists():
                try:
                    self._profiles = json.loads(self._profiles_path.read_text(encoding="utf-8"))
                except Exception:
                    self._profiles = {}
            else:
                # default "owner" profile
                self._profiles = {
                    "person:owner": {
                        "display": "owner",
                        "enrolled": False,
                        "vision_proto": None,
                        "voice_fp32": None,
                        "seen_count": 0,
                        "last_seen_ts": 0.0,
                    }
                }
                self._save_profiles()

    def _save_profiles(self) -> None:
        if self._profiles_path is None:
            return
        try:
            self._profiles_path.write_text(json.dumps(self._profiles, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _extract_vision_proto(self, bundle: dict) -> Optional[str]:
        assets = bundle.get("assets", None)
        if not isinstance(assets, list):
            return None
        for a in assets:
            if not isinstance(a, dict):
                continue
            proto_id = str(a.get("proto_id", "") or "").strip()
            if proto_id:
                return proto_id
        return None

    async def _maybe_auto_enroll(self, ctx, person_id: str) -> None:
        p = self._profiles.get(person_id, {})
        auto = bool(await ctx.get_kv("identity:auto_enroll", True))
        if not auto:
            return

        if not p.get("vision_proto") and self._last_vision_proto:
            p["vision_proto"] = self._last_vision_proto
            p["enrolled"] = True

        if p.get("voice_fp32") is None and self._last_audio_fp:
            p["voice_fp32"] = self._last_audio_fp
            p["enrolled"] = True

        self._profiles[person_id] = p
        self._save_profiles()

    def _compute_confidence(self, p: dict) -> Tuple[float, float, float]:
        # Vision confidence
        v_conf = 0.0
        if self._last_vision_proto and p.get("vision_proto"):
            v_conf = 1.0 if self._last_vision_proto == p.get("vision_proto") else 0.0

        # Audio confidence
        a_conf = 0.0
        if self._last_audio_fp is not None and isinstance(p.get("voice_fp32"), list):
            cs = _cos_sim(self._last_audio_fp, p["voice_fp32"])
            # map to 0..1 (cos is typically 0..1 for similar)
            a_conf = max(0.0, min(1.0, (cs + 1.0) / 2.0))

        # Noisy-OR fusion (weights)
        wv = 0.60
        wa = 0.50
        fused = 1.0 - ((1.0 - v_conf) ** wv) * ((1.0 - a_conf) ** wa)

        return fused, v_conf, a_conf

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, payload=event.payload, source=event.source, meta=event.meta)

        await self._ensure_loaded(ctx)

        # Update last vision proto from recall bundle
        if event.topic == "memory/recall_context":
            bundle = event.payload if isinstance(event.payload, dict) else {}
            proto = self._extract_vision_proto(bundle)
            if proto:
                self._last_vision_proto = proto
                self._last_update_ts = time.time()

        # Update last audio fingerprint from audio engram
        if event.topic == "memory/audio_engram":
            row = event.payload if isinstance(event.payload, dict) else {}
            fp = row.get("fp32", None)
            if isinstance(fp, list) and fp:
                self._last_audio_fp = [float(x) for x in fp]
                self._last_update_ts = time.time()

        # Only compute if we have at least one modality updated
        if self._last_update_ts <= 0.0:
            return []

        person_id = str(await ctx.get_kv("identity:person_id", "person:owner") or "person:owner")
        if person_id not in self._profiles:
            self._profiles[person_id] = {
                "display": person_id.replace("person:", ""),
                "enrolled": False,
                "vision_proto": None,
                "voice_fp32": None,
                "seen_count": 0,
                "last_seen_ts": 0.0,
            }
            self._save_profiles()

        await self._maybe_auto_enroll(ctx, person_id)

        p = self._profiles.get(person_id, {})
        conf, v_conf, a_conf = self._compute_confidence(p)

        # Hysteresis / smoothing
        prev = await ctx.get_kv("identity:last", {}) or {}
        prev_c = float(prev.get("confidence", 0.0) or 0.0)
        alpha = float(await ctx.get_kv("identity:smoothing", 0.35) or 0.35)
        smoothed = (1.0 - alpha) * prev_c + alpha * conf

        out = {
            "person_id": person_id,
            "confidence": float(smoothed),
            "vision_conf": float(v_conf),
            "audio_conf": float(a_conf),
            "vision_proto": self._last_vision_proto,
            "ts": time.time(),
        }
        await ctx.set_kv("identity:last", out)

        # Update profile stats
        p["seen_count"] = int(p.get("seen_count", 0) or 0) + 1
        p["last_seen_ts"] = out["ts"]
        self._profiles[person_id] = p
        self._save_profiles()

        if is_debug_enabled():
            print(
                f"[IDENTITY] {person_id} conf={out['confidence']:.3f} "
                f"(vision={out['vision_conf']:.2f} audio={out['audio_conf']:.2f}) "
                f"proto={out['vision_proto']}"
            )

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["memory/recall_context", "memory/audio_engram"],
        output_topics=[],
        priority=3,
        cooldown_sec=0.0,
    )
    yield IdentityFuserNeuron(cfg)
