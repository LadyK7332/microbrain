from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from microbrain.orchestrator.debug_utils import is_debug_enabled
from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.utils.memdir import resolve_memdir_ctx
from microbrain.voice.tts import TTS

NEURON_NAME = Path(__file__).stem


def _safe_label(text: str) -> str:
    t = " ".join(str(text or "").strip().split()).lower()
    # keep simple: alnum and spaces only
    out = []
    for ch in t:
        if ch.isalnum() or ch in (" ", "-", "_"):
            out.append(ch)
    t2 = "".join(out).strip()
    return t2[:64]


def _write_wav_i16_mono(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)


def _fingerprint_vec(pcm_bytes: bytes, sample_rate: int, bands: int = 32) -> list[float]:
    """
    Cheap spectral fingerprint:
      - take up to first 1.2s
      - rfft magnitude
      - sum energy into N linear bands
      - log + L2 normalize
    """
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return [0.0] * bands

    max_n = int(min(x.size, int(sample_rate * 1.2)))
    x = x[:max_n] / 32768.0

    # window
    w = np.hanning(x.size).astype(np.float32)
    xw = x * w

    spec = np.fft.rfft(xw)
    mag = np.abs(spec).astype(np.float32)

    # ignore DC bin
    mag = mag[1:]

    if mag.size < bands:
        # pad
        mag = np.pad(mag, (0, bands - mag.size), mode="constant")

    chunks = np.array_split(mag, bands)
    e = np.array([float(np.sum(c * c)) for c in chunks], dtype=np.float32)

    e = np.log10(1e-7 + e)
    # normalize
    n = float(np.linalg.norm(e) + 1e-9)
    e = e / n
    return [float(v) for v in e.tolist()]


class AudioEngramNeuron(BaseNeuron):
    """
    Stores labeled audio exemplars ("engrams") for robust hearing + identity.

    Listens:
      - percept/audio_utterance  (text + pcm_bytes + sample_rate)

    Writes:
      - memdir/audio/engrams.jsonl
      - memdir/audio/snippets/human/<2hex>/<sha1>.wav
      - (optional) memdir/audio/snippets/tts/<label>/<sha1>.wav
    Emits:
      - memory/audio_engram
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._memdir: Optional[Path] = None
        self._engrams_path: Optional[Path] = None

    async def _ensure_paths(self, ctx) -> Path:
        if self._memdir is None:
            self._memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
            (self._memdir / "audio" / "snippets" / "human").mkdir(parents=True, exist_ok=True)
            (self._memdir / "audio" / "snippets" / "tts").mkdir(parents=True, exist_ok=True)
            self._engrams_path = self._memdir / "audio" / "engrams.jsonl"
        return self._memdir

    def _append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        dbg_payload = event.payload
        if isinstance(dbg_payload, dict) and "pcm_bytes" in dbg_payload:
            dbg_payload = {**dbg_payload, "pcm_bytes": f"<bytes {len(dbg_payload.get('pcm_bytes', b''))}>"}
        self.debug("received", topic=event.topic, payload=dbg_payload, source=event.source, meta=event.meta)

        if event.topic != "percept/audio_utterance":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        text = str(payload.get("text", "") or "").strip()
        pcm_bytes = payload.get("pcm_bytes", b"")
        sample_rate = int(payload.get("sample_rate", 16000) or 16000)

        if not text or not isinstance(pcm_bytes, (bytes, bytearray)) or len(pcm_bytes) < 2000:
            return []

        await self._ensure_paths(ctx)
        assert self._memdir is not None and self._engrams_path is not None

        sha1 = hashlib.sha1(pcm_bytes).hexdigest()
        sub = sha1[:2]
        human_wav = self._memdir / "audio" / "snippets" / "human" / sub / f"{sha1}.wav"

        if not human_wav.exists():
            try:
                _write_wav_i16_mono(human_wav, bytes(pcm_bytes), sample_rate)
            except Exception as exc:
                if is_debug_enabled():
                    print(f"[AUDIO_ENGRAM] failed to write wav: {exc!r}")
                return []

        fp = _fingerprint_vec(bytes(pcm_bytes), sample_rate, bands=32)

        # label mode: short utterances become "designations" (car, hello, etc.)
        words = text.split()
        label = _safe_label(text) if (len(words) <= 3 and len(text) <= 24) else ""

        # Detect likely self-echo: MB hears its own TTS through the mic.
        self_echo = False
        salience = 0.15
        try:
            mute_until = float(await ctx.get_kv("ears:mute_until", 0.0) or 0.0)
            last_spoken = await ctx.get_kv("tts:last_spoken", {}) or {}
            if mute_until and time.time() < mute_until:
                self_echo = True
            if isinstance(last_spoken, dict):
                lt = str(last_spoken.get("text", "") or "").strip().lower()
                lts = float(last_spoken.get("ts", 0.0) or 0.0)
                if lt and lt == text.strip().lower() and (time.time() - lts) <= 6.0:
                    self_echo = True
        except Exception:
            pass
        if self_echo:
            salience = -0.95

        tts_wav = ""
        tts_fp: list[float] | None = None
        # Optional babysitter reference clip (won't play out loud; saved to file)
        tts_enabled = bool(await ctx.get_kv("audio:tts_reference_enabled", True))
        if label and tts_enabled:
            try:
                tts_dir = self._memdir / "audio" / "snippets" / "tts" / label
                tts_dir.mkdir(parents=True, exist_ok=True)
                tts_sha = hashlib.sha1(label.encode("utf-8")).hexdigest()
                tts_wav_path = tts_dir / f"{tts_sha}.wav"
                if not tts_wav_path.exists():
                    voice = str(await ctx.get_kv("tts:voice", "Zira") or "Zira")
                    rate = int(await ctx.get_kv("tts:rate", 155) or 155)
                    volume = float(await ctx.get_kv("tts:volume", 0.9) or 0.9)
                    tts = TTS(rate=rate, volume=volume, preferred=voice)
                    tts.save_to_file(label, str(tts_wav_path))
                # Compute fp from saved wav (read back)
                try:
                    import wave
                    with wave.open(str(tts_wav_path), "rb") as wf:
                        if wf.getsampwidth() == 2 and wf.getnchannels() == 1:
                            rb = wf.readframes(wf.getnframes())
                            tts_fp = _fingerprint_vec(rb, wf.getframerate(), bands=32)
                            tts_wav = str(tts_wav_path)
                except Exception:
                    pass
            except Exception as exc:
                if is_debug_enabled():
                    print(f"[AUDIO_ENGRAM] tts reference failed: {exc!r}")

        row = {
            "kind": "audio_engram",
            "ts": time.time(),
            "text": text,
            "label": label,
            "sha1": sha1,
            "sample_rate": sample_rate,
            "human_wav": str(human_wav),
            "fp32": fp,
            "salience": salience,
            "self_echo": bool(self_echo),
        }
        if tts_wav:
            row["tts_wav"] = tts_wav
        if tts_fp is not None:
            row["tts_fp32"] = tts_fp

        self._append_jsonl(self._engrams_path, row)

        return [
            Event(
                topic="memory/audio_engram",
                payload=row,
                source=self.name,
                correlation_id=event.correlation_id,
                meta={"schema_ver": 1},
            )
        ]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio_utterance"],
        output_topics=["memory/audio_engram"],
        priority=2,
        cooldown_sec=0.0,
    )
    yield AudioEngramNeuron(cfg)
