# examples/gpu_dense_demo.py
# Minimal smoke test for DenseRegionGPU

import numpy as np
from microbrain2.regions.gpu_dense import DenseRegionGPU

if __name__ == "__main__":
    n_pre, n_post = 512, 256
    r = DenseRegionGPU(name="gpu_dense", n_pre=n_pre, n_post=n_post, activation="tanh")

    # Fake input
    x = np.random.randn(n_pre).astype(np.float32)

    # Engine-ish call pattern
    r.step(dt=0.01, inputs={"x": x})

    y = r.post_activity()["y"]
    print("y shape:", y.shape, " sample:", y[:8])
