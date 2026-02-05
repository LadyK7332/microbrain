from __future__ import annotations

import time
from dataclasses import dataclass, field

from microbrain.orchestrator.neuron_base import Event


@dataclass
class AttentionController:
    """
    Minimal attention gate for external vs internal speech.

    Rule:
        If any external stimulus occurred recently, internal speech is forbidden.
    """

    external_hold_ms: int = 4000
    last_external_ts: float = field(default_factory=time.time)
    allow_babble: bool = False

    def observe_event(self, event: Event) -> None:
        source = self._extract_source(event)
        if source in {"cli", "mic"}:
            self.last_external_ts = time.time()

    def update_allow_babble(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        elapsed_ms = (now - self.last_external_ts) * 1000.0
        self.allow_babble = elapsed_ms >= float(self.external_hold_ms)
        return self.allow_babble

    @staticmethod
    def _extract_source(event: Event) -> str:
        if event.source:
            return str(event.source)
        if isinstance(event.payload, dict):
            payload_source = event.payload.get("source")
            if payload_source:
                return str(payload_source)
        if event.meta:
            meta_source = event.meta.get("source")
            if meta_source:
                return str(meta_source)
        return ""