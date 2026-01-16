from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import sounddevice as sd


@dataclass
class MicProbeResult:
    device: Optional[int]
    device_name: str
    samplerate: int
    seconds: float
    rms: float
    peak: float


def list_input_devices() -> List[Dict[str, Any]]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0) or 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": str(d.get("name", "")),
                    "max_input_channels": int(d.get("max_input_channels", 0) or 0),
                    "default_samplerate": float(d.get("default_samplerate", 0.0) or 0.0),
                }
            )
    return out


def probe_rms(
    device: Optional[int],
    samplerate: int,
    seconds: float = 0.75,
    rms_threshold: float = 0.003,
) -> MicProbeResult:
    """
    Record a short clip and compute RMS/peak (float32 normalized [-1..1]).
    Raises RuntimeError if RMS < threshold.
    """
    frames = int(seconds * samplerate)
    kwargs = {}
    if device is not None:
        kwargs["device"] = device

    audio = sd.rec(
        frames,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        **kwargs,
    )
    sd.wait()

    x = np.asarray(audio[:, 0], dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)

    dev_name = "default"
    try:
        if device is not None:
            dev_name = sd.query_devices(device)["name"]
        else:
            dev_name = sd.query_devices(sd.default.device[0])["name"]
    except Exception:
        pass

    res = MicProbeResult(
        device=device,
        device_name=str(dev_name),
        samplerate=int(samplerate),
        seconds=float(seconds),
        rms=rms,
        peak=peak,
    )

    if res.rms < rms_threshold:
        raise RuntimeError(
            f"Mic RMS too low ({res.rms:.6f} < {rms_threshold}). "
            f"Device={res.device} ({res.device_name}) sr={res.samplerate}"
        )

    return res
