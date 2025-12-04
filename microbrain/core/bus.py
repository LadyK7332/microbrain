# microbrain/core/bus.py
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any]
    ts: float


class EventBus:
    def __init__(self) -> None:
        self._subs: defaultdict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, fn: Callable[[Event], None]) -> None:
        with self._lock:
            self._subs[event_type].append(fn)

    def publish(self, event_type: str, **payload: Any) -> None:
        evt = Event(type=event_type, payload=payload, ts=time.time())
        with self._lock:
            subs = list(self._subs.get(event_type, [])) + self._subs.get("*", [])
        for fn in subs:
            try:
                fn(evt)
            except Exception as e:
                # Never let a subscriber kill the loop
                print(f"[bus] subscriber error on {event_type}: {e}")
