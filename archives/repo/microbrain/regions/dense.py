from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DenseRegion:
    name: str
    n_pre: int
    n_post: int
    W: np.ndarray | None = None
    activation: str = "tanh"  # "linear" | "relu" | "tanh"

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

    def pre_activity(self) -> dict[str, np.ndarray]:
        return {"x": self.x}

    def post_activity(self) -> dict[str, np.ndarray]:
        return {"y": self.y}

    def weights(self) -> list[list[float]]:
        return self.W.tolist()

    def step(self, dt: float, inputs: dict[str, np.ndarray]) -> None:
        xin = inputs.get("x")
        if xin is None:
            return
        self.x[...] = np.asarray(xin, dtype=np.float32)
        self.y[...] = self.W @ self.x
        if self.activation == "relu":
            np.maximum(self.y, 0.0, out=self.y)
        elif self.activation == "tanh":
            self.y[...] = np.tanh(self.y)
