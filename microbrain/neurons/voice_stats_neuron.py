from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from microbrain.orchestrator.debug_utils import is_debug_enabled
from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx

NEURON_NAME = Path(__file__).stem


def _dbfs_rms(x: np.ndarray) -> float:
    # x is float32 in [-1, 1]
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return 20.0 * math.log10(max(1e-9, rms))


def _frame_view(x: np.ndarray, frame_n: int, hop_n: int) -> np.ndarray:
    """Return framed view of x: shape (n_frames, frame_n). Copies minimal."""
    if x.size < frame_n:
        return np.empty((0, frame_n), dtype=x.dtype)
    n_frames = 1 + (x.size - frame_n) // hop_n
    # Stride trick
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_n),
        strides=(hop_n * stride, stride),
        writeable=False,
    )


def _autocorr_pitch_hz(frame: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 350.0) -> Optional[float]:
    """
    Very cheap pitch estimate:
      - remove DC
      - compute autocorrelation via FFT
      - find peak in lag range corresponding to [fmin, fmax]
    Returns Hz or None.
    """
    if frame.size == 0:
        return None

    x = frame.astype(np.float32)
    x = x - float(np.mean(x))
    # Window to reduce edge effects
    w = np.hanning(x.size).astype(np.float32)
    xw = x * w

    # FFT autocorrelation
    n = int(1 << (int(xw.size - 1).bit_length() + 1))  # next power-of-two *2
    X = np.fft.rfft(xw, n=n)
    ac = np.fft.irfft(X * np.conj(X), n=n).astype(np.float32)
    ac = ac[: xw.size]

    # Normalize
    ac0 = float(ac[0]) + 1e-12
    ac = ac / ac0

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag <= min_lag + 2:
        return None
    if max_lag >= ac.size:
        max_lag = ac.size - 1
    if max_lag <= min_lag + 2:
        return None

    seg = ac[min_lag:max_lag]
    idx = int(np.argmax(seg))
    peak = float(seg[idx])

    # Require a decent periodicity peak
    if peak < 0.20:
        return None

    lag = min_lag + idx
    if lag <= 0:
        return None
    return float(sr / lag)


def _spectral_centroid_hz(frame: np.ndarray, sr: int) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32)
    x = x - float(np.mean(x))
    w = np.hanning(x.size).astype(np.float32)
    xw = x * w
    spec = np.fft.rfft(xw)
    mag = np.abs(spec).astype(np.float32) + 1e-12
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / sr).astype(np.float32)
    return float(np.sum(freqs * mag) / np.sum(mag))


def _voice_stats_from_pcm(pcm_bytes: bytes, sr: int) -> Dict[str, Any]:
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return {
            "rms_dbfs": -120.0,
            "voiced_ratio": 0.0,
            "f0_median_hz": None,
            "f0_p10_hz": None,
            "f0_p90_hz": None,
            "centroid_hz": 0.0,
        }

    # Normalize to [-1, 1]
    x = x / 32768.0

    frame_ms = 30.0
    hop_ms = 15.0
    frame_n = max(64, int(sr * frame_ms / 1000.0))
    hop_n = max(32, int(sr * hop_ms / 1000.0))

    frames = _frame_view(x, frame_n, hop_n)
    if frames.shape[0] == 0:
        return {
            "rms_dbfs": _dbfs_rms(x),
            "voiced_ratio": 0.0,
            "f0_median_hz": None,
            "f0_p10_hz": None,
            "f0_p90_hz": None,
            "centroid_hz": _spectral_centroid_hz(x[: min(x.size, sr)], sr),
        }

    # Energy per frame
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12).astype(np.float32)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9)).astype(np.float32)

    # Voiced frames heuristic: above -35 dBFS and not too peaky
    voiced_mask = rms_db > -35.0
    voiced_idx = np.nonzero(voiced_mask)[0]
    voiced_ratio = float(voiced_idx.size / max(1, frames.shape[0]))

    # Pitch estimate on up to N voiced frames (spread out)
    f0s: list[float] = []
    if voiced_idx.size > 0:
        # sample up to 18 frames evenly across utterance
        take = min(18, int(voiced_idx.size))
        pick = np.linspace(0, voiced_idx.size - 1, take).round().astype(int)
        for j in pick:
            fr = frames[int(voiced_idx[int(j)])]
            hz = _autocorr_pitch_hz(fr, sr)
            if hz is not None and 50.0 <= hz <= 450.0:
                f0s.append(float(hz))

    if f0s:
        arr = np.array(f0s, dtype=np.float32)
        f0_median = float(np.median(arr))
        f0_p10 = float(np.percentile(arr, 10))
        f0_p90 = float(np.percentile(arr, 90))
    else:
        f0_median = None
        f0_p10 = None
        f0_p90 = None

    # Centroid: use a representative 1s slice around the middle
    mid = x.size // 2
    win = min(x.size, sr)
    start = max(0, mid - win // 2)
    centroid = _spectral_centroid_hz(x[start:start + win], sr)

    return {
        "rms_dbfs": float(np.median(rms_db)),
        "voiced_ratio": voiced_ratio,
        "f0_median_hz": f0_median,
        "f0_p10_hz": f0_p10,
        "f0_p90_hz": f0_p90,
        "centroid_hz": centroid,
    }


def _ema(old: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
    if new is None:
        return old
    if old is None:
        return float(new)
    return float((1.0 - alpha) * old + alpha * float(new))


class VoiceStatsNeuron(BaseNeuron):
    """
    Compute lightweight voice stats + maintain per-person voiceprints.

    Listens:
      - percept/audio_utterance (expects: text, pcm_bytes, sample_rate, speaker, raw_meta)

    Writes:
      - memdir/identity/voice_stats.jsonl   (append-only telemetry)
      - memdir/identity/voiceprints.json    (rolling per-person stats)

    Emits:
      - memory/voice_stats   (payload includes person_id + stats)
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._memdir: Optional[Path] = None
        self._stats_path: Optional[Path] = None
        self._voiceprints_path: Optional[Path] = None
        self._voiceprints: Dict[str, Any] = {}

    async def _ensure_paths(self, ctx) -> None:
        if self._memdir is not None:
            return
        memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
        self._memdir = memdir
        id_dir = memdir / "identity"
        id_dir.mkdir(parents=True, exist_ok=True)
        self._stats_path = id_dir / "voice_stats.jsonl"
        self._voiceprints_path = id_dir / "voiceprints.json"

        # Load voiceprints if present
        try:
            if self._voiceprints_path.exists():
                self._voiceprints = json.loads(self._voiceprints_path.read_text(encoding="utf-8", errors="ignore"))
                if not isinstance(self._voiceprints, dict):
                    self._voiceprints = {}
        except Exception:
            self._voiceprints = {}

    def _save_voiceprints(self) -> None:
        if not self._voiceprints_path:
            return
        try:
            self._voiceprints_path.write_text(json.dumps(self._voiceprints, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != "percept/audio_utterance":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        text = str(payload.get("text", "") or "").strip()
        pcm = payload.get("pcm_bytes", b"") or b""
        sr = int(payload.get("sample_rate", 16000) or 16000)
        raw_meta = payload.get("raw_meta", {}) or {}
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        if not isinstance(pcm, (bytes, bytearray)) or len(pcm) < 32:
            return []

        await self._ensure_paths(ctx)

        # Determine person_id from identity fusion if confident enough
        person_id = "person:unknown"
        try:
            forced = await ctx.get_kv("voice_stats:force_person_id", None)
            if forced:
                person_id = str(forced)
            else:
                ident = await ctx.get_kv("identity:last", {}) or {}
                if isinstance(ident, dict):
                    conf = float(ident.get("confidence", 0.0) or 0.0)
                    if conf >= float(await ctx.get_kv("voice_stats:use_identity_conf_min", 0.75) or 0.75):
                        person_id = str(ident.get("person_id", person_id) or person_id)
        except Exception:
            pass

        # Self-echo detection (possible/confirmed)
        self_echo = False
        self_echo_confirmed = False
        try:
            mute_until = float(await ctx.get_kv("ears:mute_until", 0.0) or 0.0)
            last_spoken = await ctx.get_kv("tts:last_spoken", {}) or {}
            now = time.time()
            if mute_until and now < mute_until:
                self_echo = True
            if isinstance(last_spoken, dict):
                lt = str(last_spoken.get("text", "") or "").strip().lower()
                lts = float(last_spoken.get("ts", 0.0) or 0.0)
                if lt and text and lt == text.strip().lower() and (now - lts) <= 6.0:
                    self_echo = True
                    self_echo_confirmed = True
        except Exception:
            pass

        stats = _voice_stats_from_pcm(bytes(pcm), sr)
        sha1_audio = hashlib.sha1(bytes(pcm)).hexdigest()
        row = {
            "ts": time.time(),
            "person_id": person_id,
            "text": text,
            "sha1_audio": sha1_audio,
            "sample_rate": sr,
            "self_echo": bool(self_echo),
            "self_echo_confirmed": bool(self_echo_confirmed),
            "stats": stats,
            "raw_meta": {"input_modality": raw_meta.get("input_modality"), "device_index": raw_meta.get("device_index")},
        }

        # Append telemetry
        try:
            assert self._stats_path is not None
            with self._stats_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Update per-person voiceprint only if not self-echo
        if (not self_echo) and person_id != "person:unknown":
            vp = self._voiceprints.get(person_id, {})
            if not isinstance(vp, dict):
                vp = {}

            alpha = float(await ctx.get_kv("voice_stats:ema_alpha", 0.15) or 0.15)
            vp["count"] = int(vp.get("count", 0) or 0) + 1
            vp["last_ts"] = float(row["ts"])
            vp["last_sha1_audio"] = sha1_audio

            # EMA fields
            vp["rms_dbfs_ema"] = _ema(vp.get("rms_dbfs_ema", None), stats.get("rms_dbfs", None), alpha)
            vp["voiced_ratio_ema"] = _ema(vp.get("voiced_ratio_ema", None), stats.get("voiced_ratio", None), alpha)
            vp["centroid_hz_ema"] = _ema(vp.get("centroid_hz_ema", None), stats.get("centroid_hz", None), alpha)
            vp["f0_median_hz_ema"] = _ema(vp.get("f0_median_hz_ema", None), stats.get("f0_median_hz", None), alpha)
            vp["f0_p10_hz_ema"] = _ema(vp.get("f0_p10_hz_ema", None), stats.get("f0_p10_hz", None), alpha)
            vp["f0_p90_hz_ema"] = _ema(vp.get("f0_p90_hz_ema", None), stats.get("f0_p90_hz", None), alpha)

            self._voiceprints[person_id] = vp
            self._save_voiceprints()

        if is_debug_enabled():
            self.debug(
                "voice_stats",
                person_id=person_id,
                self_echo=self_echo,
                f0=stats.get("f0_median_hz"),
                voiced=stats.get("voiced_ratio"),
                rms_dbfs=stats.get("rms_dbfs"),
            )

        return [
            Event(
                topic="memory/voice_stats",
                payload=row,
                source=NEURON_NAME,
                correlation_id=event.correlation_id,
                meta={"kind": "voice_stats"},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio_utterance"],
        output_topics=["memory/voice_stats"],
        priority=3,
        cooldown_sec=0.0,
    )
    yield VoiceStatsNeuron(cfg)
