import time
import numpy as np
import sounddevice as sd

DURATION = 2.0
SAMPLE_RATE = 16000

def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

print("=== Devices ===")
print(sd.query_devices())
print("Default input:", sd.default.device)

print("\n=== Recording ===")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()

val = rms(audio[:, 0])
print(f"RMS={val:.6f} (threshold suggestion: ~0.002–0.01 depending on mic gain)")

if val < 0.002:
    raise SystemExit("FAIL: mic energy too low (device wrong, muted, or Windows app volume issue).")
print("OK: mic has energy.")
