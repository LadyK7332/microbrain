from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    topic: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


class EventBus:
    """
    Ultra-light pub/sub bus:
      - subscribe(topic, callback)
      - publish(topic, **payload) -> Event
      - history keeps the last N events for debugging/inspection
    """

    def __init__(self, history: int = 256) -> None:
        self._subs: defaultdict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history = deque(maxlen=history)

    def subscribe(self, topic: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subs[topic].append(callback)

    def publish(self, topic: str, **payload: Any) -> Event:
        ev = Event(topic=topic, payload=payload)
        with self._lock:
            self._history.append(ev)
            # copy to avoid mutation during callbacks
            callbacks = list(self._subs.get(topic, ()))
        for cb in callbacks:
            try:
                cb(ev)
            except Exception:
                # keep the bus resilient; individual handlers shouldn't crash the loop
                pass
        return ev

    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)
