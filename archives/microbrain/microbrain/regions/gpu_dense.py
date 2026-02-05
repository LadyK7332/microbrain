# microbrain2/regions/gpu_dense.py
# Python 3.13.7 — Kompute (Vulkan) + PyShader implementation of a Dense (matvec) region.
# Keeps host-side weights in numpy; ships matvec to GPU; activation done on CPU.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import kp
import numpy as np
from kp import Manager
from pyshader import Array, f32, ivec3, python2shader

# ruff B008: avoid calls in default args; use module-level singletons
_PY_BUF_F32 = Array(f32)


def make_dense_shader(n_pre: int):
    """Build a SPIR-V compute shader for y = W @ x where W is (n_post x n_pre).
    Each invocation computes one output neuron i.
    """

    # NOTE: We bake n_pre into the shader so the loop bound is a compile-time constant.
    @python2shader
    def _dense(
        index=("input", "GlobalInvocationId", ivec3),
        x=("buffer", 0, _PY_BUF_F32),  # len = n_pre
        W=("buffer", 1, _PY_BUF_F32),  # len = n_post * n_pre (row-major: i*n_pre + j)
        y=("buffer", 2, _PY_BUF_F32),
    ):  # len = n_post
        i = index.x
        acc = 0.0
        j = 0
        # while-loop to keep pyshader happy on dynamic unroll
        while j < n_pre:
            acc = acc + W[i * n_pre + j] * x[j]
            j = j + 1
        y[i] = acc

    return _dense


@dataclass
class DenseRegionGPU:
    name: str
    n_pre: int
    n_post: int
    W: np.ndarray | None = None  # shape (n_post, n_pre), float32
    activation: str = "tanh"  # "linear" | "relu" | "tanh"

    # Runtime fields
    _mgr: Manager = field(init=False, repr=False)
    _t_x: Any = field(init=False, repr=False)
    _t_W: Any = field(init=False, repr=False)
    _t_y: Any = field(init=False, repr=False)
    _algo: Any = field(init=False, repr=False)
    _sq: Any = field(init=False, repr=False)
    _dirty_W: bool = field(default=False, init=False, repr=False)

    # Host-side state
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.W = (
            self.W.astype(np.float32)
            if isinstance(self.W, np.ndarray) and self.W.size
            else (np.random.randn(self.n_post, self.n_pre).astype(np.float32) * 0.1)
        )
        self.x = np.zeros((self.n_pre,), dtype=np.float32)
        self.y = np.zeros((self.n_post,), dtype=np.float32)

        self._mgr = Manager()
        self._t_x = self._mgr.tensor(self.x.tolist())
        self._t_W = self._mgr.tensor(self.W.reshape(-1).tolist())
        self._t_y = self._mgr.tensor(self.y.tolist())

        shader = make_dense_shader(self.n_pre)
        self._algo = self._mgr.algorithm([self._t_x, self._t_W, self._t_y], shader.to_spirv())

        # Prime device buffers
        self._sq = self._mgr.sequence()
        self._sq.eval(kp.OpTensorSyncDevice([self._t_x, self._t_W, self._t_y]))

    # --- Public API expected by the engine ---
    def pre_activity(self) -> dict[str, np.ndarray]:
        return {"x": self.x}

    def post_activity(self) -> dict[str, np.ndarray]:
        return {"y": self.y}

    def weights(self) -> list[list[float]]:
        # Engine may mutate this; mark dirty so we re-upload next step.
        self._dirty_W = True
        return self.W.tolist()

    def step(self, dt: float, inputs: dict[str, np.ndarray]) -> None:
        # 1) Stage inputs
        xin = inputs.get("x")
        if xin is None:
            # graceful no-op
            return
        self.x[...] = np.asarray(xin, dtype=np.float32)

        # 2) Sync any host-updated weights
        if self._dirty_W:
            self._t_W.set_data(self.W.reshape(-1).tolist())
            self._sq.eval(kp.OpTensorSyncDevice([self._t_W]))
            self._dirty_W = False

        # 3) Upload x -> device & dispatch compute
        self._t_x.set_data(self.x.tolist())
        self._sq.eval(kp.OpTensorSyncDevice([self._t_x]))
        self._sq.eval(kp.OpAlgoDispatch(self._algo))
        self._sq.eval(kp.OpTensorSyncLocal([self._t_y]))

        # 4) Download y and run activation on CPU (cheap)
        self.y[...] = np.asarray(self._t_y.data(), dtype=np.float32)
        if self.activation == "relu":
            np.maximum(self.y, 0.0, out=self.y)
        elif self.activation == "tanh":
            self.y[...] = np.tanh(self.y)
        # else: linear

    # Called after Hebbian learning mutates host-side W
    def sync_weights_to_device(self) -> None:
        self._dirty_W = True
