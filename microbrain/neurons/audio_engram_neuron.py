from __future__ import annotations

import hashlib
import json
import re
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx
from microbrain.voice.tts import TTS

NEURON_NAME = Path(__file__).stem

_JSONL_LOCK = None  # lazily created threading.Lock if needed (avoid import unless used)


def _ensure_lock():
    global _JSONL_LOCK
    if _JSONL_LOCK is None:
        import threading
        _JSONL_LOCK = threading.Lock()
    return _JSONL_LOCK


def _sanitize_label(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = s.replace(" ", "_")
    s = s.strip("_")
    if not s:
        return "unknown"
    return s[:48]


def _write_wav_mono_i16(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)


def _rms_peak_i16(pcm_bytes: bytes) -> tuple[float, float]:
    if not pcm_bytes:
        return 0.0, 0.0
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    return rms, peak


def _spectral_fingerprint(pcm_bytes: bytes, sample_rate: int, bands: int = 32) -> List[float]:
    """
    Cheap, dependency-light "audio embedding":
      - take up to ~1.0s of audio (centered)
      - rFFT
      - average magnitudes into N bands
    Useful for rough similarity checks later.
    """
    if not pcm_bytes:
        return [0.0] * bands

    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return [0.0] * bands

    # take up to 1.0s (or full if shorter)
    target_n = int(max(256, min(x.size, int(sample_rate * 1.0))))
    if x.size > target_n:
        start = (x.size - target_n) // 2
        x = x[start : start + target_n]

    # window
    w = np.hanning(x.size).astype(np.float32)
    xw = x * w

    # FFT length: next power of two up to 16384
    nfft = 1
    while nfft < xw.size:
        nfft <<= 1
    nfft = int(min(nfft, 16384))
    if nfft < xw.size:
        xw = xw[:nfft]
    elif nfft > xw.size:
        xw = np.pad(xw, (0, nfft - xw.size), mode="constant")

    spec = np.abs(np.fft.rfft(xw))
    spec = np.log1p(spec).astype(np.float32)

    # banding
    if spec.size <= bands:
        vec = np.pad(spec, (0, bands - spec.size), mode="constant")[:bands]
        return [float(v) for v in vec]

    edges = np.linspace(0, spec.size, num=bands + 1, dtype=np.int32)
    vec = []
    for i in range(bands):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            vec.append(0.0)
        else:
            vec.append(float(np.mean(spec[a:b])))
    return vec


class AudioEngramNeuron(BaseNeuron):
    """
    Stores labeled audio snippets ("engrams") for later similarity checks.

    Input:
      - percept/audio_utterance:
          {
            "text": "...",           # transcript label
            "pcm_bytes": <bytes>,    # int16 mono PCM
            "sample_rate": 16000,
            "channels": 1,
            "speaker": "user",
            ...
          }

    Output:
      - memory/audio_engram (metadata record)
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._tts_ref: Optional[TTS] = None
        self._tts_cfg: tuple[str | None, int, float] | None = None

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        lock = _ensure_lock()
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _get_tts(self, voice: str | None, rate: int, volume: float) -> TTS:
        cfg = (voice, int(rate), float(volume))
        if self._tts_ref is None or self._tts_cfg != cfg:
            self._tts_ref = TTS(rate=int(rate), volume=float(volume), preferred=voice or "")
            self._tts_cfg = cfg
        return self._tts_ref

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, source=event.source, payload=event.payload)

        if event.topic != "percept/audio_utterance":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        text = str(payload.get("text", "") or "").strip()
        pcm_bytes = payload.get("pcm_bytes", None)
        sr = int(payload.get("sample_rate", 0) or 0)

        if not text or not isinstance(pcm_bytes, (bytes, bytearray)) or sr <= 0:
            return []

        label = _sanitize_label(text)
        sha1 = hashlib.sha1(bytes(pcm_bytes)).hexdigest()
        rms, peak = _rms_peak_i16(bytes(pcm_bytes))
        vec = _spectral_fingerprint(bytes(pcm_bytes), sample_rate=sr, bands=32)

        memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
        engrams_jsonl = memdir / "audio" / "engrams.jsonl"

        # Write human snippet WAV
        human_path = memdir / "audio" / "snippets" / "human" / sha1[:2] / f"{sha1}.wav"
        try:
            _write_wav_mono_i16(human_path, bytes(pcm_bytes), sr)
        except Exception as e:
            await ctx.log_warn(f"[{self.name}] Failed to write human wav", err=str(e))
            human_path = Path("")

        rec_human: Dict[str, Any] = {
            "ts": time.time(),
            "kind": "human",
            "label": label,
            "text": text,
            "sha1": sha1,
            "path": str(human_path) if human_path else "",
            "sample_rate": sr,
            "rms": rms,
            "peak": peak,
            "fingerprint": vec,
        }
        self._append_jsonl(engrams_jsonl, rec_human)

        out: List[Event] = [
            Event(
                topic="memory/audio_engram",
                payload=rec_human,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"kind": "human"},
            )
        ]

        # Optional: store a TTS "reference" clip for the same label (babysitter voice).
        tts_ref_enabled = bool(await ctx.get_kv("audio:tts_ref_enabled", True))
        if tts_ref_enabled:
            try:
                voice = await ctx.get_kv("tts:voice", "Zira")
                rate = int(await ctx.get_kv("tts:rate", 155))
                volume = float(await ctx.get_kv("tts:volume", 0.9))

                # Keep references short-ish (avoid saving long paragraphs)
                if len(text) <= 48 and len(text.split()) <= 6:
                    tts = self._get_tts(str(voice) if voice else None, rate=rate, volume=volume)
                    tts_sha = hashlib.sha1(f"{label}|{voice}|{rate}|{volume}".encode("utf-8")).hexdigest()
                    tts_path = memdir / "audio" / "snippets" / "tts" / label / f"{tts_sha}.wav"

                    # Save-to-file blocks; run in a thread to avoid stalling the event loop.
                    import asyncio
                    await asyncio.to_thread(tts.save_to_file, text, str(tts_path))

                    # Record metadata (we do not re-read the wav; we trust pyttsx3)
                    rec_tts: Dict[str, Any] = {
                        "ts": time.time(),
                        "kind": "tts_ref",
                        "label": label,
                        "text": text,
                        "sha1": tts_sha,
                        "path": str(tts_path),
                        "sample_rate": None,
                        "voice": voice,
                        "rate": rate,
                        "volume": volume,
                    }
                    self._append_jsonl(engrams_jsonl, rec_tts)

                    out.append(
                        Event(
                            topic="memory/audio_engram",
                            payload=rec_tts,
                            source=self.name,
                            correlation_id=event.correlation_id,
                            meta={"kind": "tts_ref"},
                        )
                    )
            except Exception as e:
                await ctx.log_warn(f"[{self.name}] TTS ref failed", err=str(e))

        return out


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio_utterance"],
        output_topics=["memory/audio_engram"],
        priority=5,
    )
    yield AudioEngramNeuron(cfg)
