from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from microbrain.utils.memdir import resolve_memdir_ctx
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.memory.filters import classify_event_for_memory

try:
    # Reuse the same thread-safe JSONL writer used by MemoryStore
    from microbrain.memory.memory_store import JSONLStore
except Exception:  # pragma: no cover
    JSONLStore = None  # type: ignore

NEURON_NAME = Path(__file__).stem


@dataclass
class EpisodeLoggerConfig:
    """
    Episode logger: writes a unified, timestamped event stream to memdir/episodes/*.jsonl.

    This is meant to support "teaching by demonstration" later (vision + user inputs + feedback),
    without changing any cognition logic.
    """
    # Max frequency (Hz) for high-volume topics like vision. 0 means "no throttling".
    vision_hz: float = 2.0


class EpisodeLoggerNeuron(BaseNeuron):
    def __init__(self, cfg: NeuronConfig, elcfg: EpisodeLoggerConfig | None = None):
        super().__init__(cfg)
        self.elcfg = elcfg or EpisodeLoggerConfig()
        self._store: Any = None
        self._episode_path: Optional[Path] = None
        self._last_vision_ts: float = 0.0

    async def _ensure_store(self, ctx) -> None:
        if self._store is not None:
            return

        base = await resolve_memdir_ctx(ctx)
        
        episodes_dir = base / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        pid = os.getpid()
        self._episode_path = episodes_dir / f"episode-{ts}-pid{pid}.jsonl"

        if JSONLStore is None:
            raise RuntimeError("JSONLStore import failed; cannot write episode logs")

        self._store = JSONLStore(str(self._episode_path))
        self.debug("episode_logger_ready", path=str(self._episode_path))

    def _vision_allowed_now(self, ts: float) -> bool:
        hz = float(self.elcfg.vision_hz or 0.0)
        if hz <= 0.0:
            return True
        min_dt = 1.0 / hz
        if (ts - self._last_vision_ts) >= min_dt:
            self._last_vision_ts = ts
            return True
        return False

    async def process(self, ctx, event: Event) -> Iterable[Event]:
        await self._ensure_store(ctx)
        # Don't spam the episode file with ticks; we only use tick to bootstrap.
        if event.topic == "clock/tick":
            return []

        # --- debug roll call (only active when --debug is passed) ----
        self.debug("received", topic=event.topic, source=event.source, meta=event.meta)

        now = float(event.payload.get("ts")) if isinstance(event.payload, dict) and "ts" in event.payload else time.time()

        guard = classify_event_for_memory(event)
        if event.topic in ("percept/text", "act/speech") and not guard.get("allow_trace", False):
            return []

        # Throttle high-volume vision events to keep disk sane at first light.
        if event.topic == "percept/vision" and not self._vision_allowed_now(now):
            return []

        row: Dict[str, Any] = {
            "ts": now,
            "topic": event.topic,
            "source": event.source,
            "meta": event.meta or {},
            "payload": event.payload,
        }

        try:
            self._store.append(row)
        except Exception as e:
            self.debug("episode_logger_error", error=str(e), path=str(self._episode_path))

        return []


def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
    # Episodes are disabled by default for now.
    # The continuous memory stream is the source of truth until a real,
    # populated episodic layer is needed again.
    if False:
        cfg = NeuronConfig(
            name=NEURON_NAME,
            subscribed_topics=[
                "clock/tick",
                "percept/text",
                "act/speech",
                "percept/vision",
                "percept/input_mouse",
                "percept/input_key",
                "curiosity/adjust",
            ],
            output_topics=[],
        )
        yield EpisodeLoggerNeuron(cfg)
    return
