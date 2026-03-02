from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event

NEURON_NAME = Path(__file__).stem


def _dc_blocker(x: np.ndarray, r: float = 0.995) -> np.ndarray:
    """
    Simple DC-block / high-pass-ish filter:
        y[n] = x[n] - x[n-1] + r * y[n-1]
    Good enough to reduce rumble / breath boom / adapter DC bias.
    """
    if x.size == 0:
        return x

    y = np.empty_like(x, dtype=np.float32)
    y0 = float(x[0])
    y[0] = y0
    prev_x = float(x[0])
    prev_y = y0
    for i in range(1, x.size):
        cur_x = float(x[i])
        cur_y = cur_x - prev_x + r * prev_y
        y[i] = cur_y
        prev_x = cur_x
        prev_y = cur_y
    return y


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Fast, dependency-light resampler (linear interpolation).
    Input x must be float32 mono in [-1..1].
    """
    if src_sr == dst_sr or x.size == 0:
        return x

    ratio = float(dst_sr) / float(src_sr)
    out_len = int(np.round(x.size * ratio))
    out_len = max(out_len, 1)

    # Map output sample positions back into input indices
    in_pos = np.linspace(0.0, x.size - 1.0, num=out_len, dtype=np.float32)
    in_idx = np.arange(x.size, dtype=np.float32)

    # np.interp expects float64; convert back to float32
    y = np.interp(in_pos, in_idx, x.astype(np.float64)).astype(np.float32)
    return y


def _rms_peak(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    xx = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(xx * xx)) + 1e-12)
    peak = float(np.max(np.abs(xx)) + 1e-12)
    return rms, peak


class CochlearNeuron(BaseNeuron):
    """
    Cochlear neuron: hardware audio -> canonical brain audio.

    Expects input payload on `percept/audio_raw` like:
      {
        "pcm_bytes": <bytes>,      # int16 mono PCM
        "sample_rate": 44100,
        "channels": 1,
        "device": <optional>,
        "source": "mic",
        ...
      }

    Emits:
      - `percept/audio_pcm` with canonical 16kHz mono int16 PCM
      - `affect/audio_energy` with RMS/peak/clipping + rates
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        if event.topic != "percept/audio_raw":
            return []

        payload = event.payload if isinstance(event.payload, dict) else {}
        pcm_bytes = payload.get("pcm_bytes", None)
        src_sr = int(payload.get("sample_rate", 0) or 0)
        channels = int(payload.get("channels", 1) or 1)

        # Config defaults
        dst_sr = int(getattr(self.config, "audio_target_sr", 16000) or 16000)
        dc_block = bool(getattr(self.config, "audio_dc_block", True))
        dc_r = float(getattr(self.config, "audio_dc_r", 0.995))

        if pcm_bytes is None or not isinstance(pcm_bytes, (bytes, bytearray)):
            self.debug("audio_raw missing pcm_bytes")
            return []

        if src_sr <= 0:
            self.debug("audio_raw missing/invalid sample_rate", sample_rate=src_sr)
            return []

        if channels != 1:
            # For now, we only support mono. Caller should downmix before emitting audio_raw.
            self.debug("audio_raw channels != 1 not supported", channels=channels)
            return []

        # Decode int16 -> float32 [-1..1]
        x_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        x = (x_i16.astype(np.float32) / 32768.0).astype(np.float32)

        # Optional DC blocker / rumble reduction
        if dc_block:
            x = _dc_blocker(x, r=dc_r)

        # Resample to canonical brain SR
        y = _resample_linear(x, src_sr=src_sr, dst_sr=dst_sr)

        # Metrics before quantization
        rms, peak = _rms_peak(y)

        # Clip + quantize back to int16
        clipped = bool(np.any(np.abs(y) > 1.0))
        y = np.clip(y, -1.0, 1.0)
        y_i16 = (y * 32767.0).astype(np.int16, copy=False)
        out_bytes = y_i16.tobytes()

        # Build outgoing events
        out_meta: Dict[str, Any] = dict(event.meta or {})
        out_meta.update(
            {
                "src_sample_rate": src_sr,
                "dst_sample_rate": dst_sr,
                "input_modality": "audio",
                "cochlear": True,
            }
        )

        # Canonical percept/audio_pcm for downstream VAD/Whisper/etc.
        e_audio = Event(
            topic="percept/audio_pcm",
            payload={
                "pcm_bytes": out_bytes,
                "sample_rate": dst_sr,
                "channels": 1,
                "rms": rms,
                "peak": peak,
                "clipped": clipped,
                "src_sample_rate": src_sr,
                "raw_meta": payload.get("raw_meta", {}),
            },
            source=NEURON_NAME,
            meta=out_meta,
        )

        # Energy signal for attention/salience gating / wake reflex
        e_energy = Event(
            topic="affect/audio_energy",
            payload={
                "rms": rms,
                "peak": peak,
                "clipped": clipped,
                "src_sample_rate": src_sr,
                "dst_sample_rate": dst_sr,
                "samples_in": int(x.size),
                "samples_out": int(y.size),
            },
            source=NEURON_NAME,
            meta=out_meta,
        )

        return [e_audio, e_energy]


def activate() -> Iterable[BaseNeuron]:
    # You can tweak these later via config if you add CLI/config wiring.
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio_raw"],
        output_topics=["percept/audio_pcm", "affect/audio_energy"],
        priority=15,
    )

    # Attach a few cochlear-specific tunables onto config
    # (Keeping it simple/flat so it works with your current NeuronConfig)
    setattr(cfg, "audio_target_sr", 16000)
    setattr(cfg, "audio_dc_block", True)
    setattr(cfg, "audio_dc_r", 0.995)

    yield CochlearNeuron(cfg)
