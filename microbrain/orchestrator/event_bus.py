# C:\aiproj\microbrain\orchestrator\event_bus.py

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .neuron_base import Event


# Type alias for handlers that can produce zero or more new events
EventHandler = Callable[[Event], Awaitable[Iterable[Event]]]


@dataclass
class Subscription:
    """
    A single subscription on the event bus.

    name:       human-friendly name (e.g. neuron name)
    topics:     list of topics this subscriber wants to receive
    priority:   higher priority subscribers see the event first
    handler:    async function(event) -> Iterable[Event]
    active:     can be toggled without unregistering
    created_at: timestamp for debugging/telemetry
    """
    name: str
    topics: Sequence[str]
    priority: int
    handler: EventHandler
    active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class BusMetrics:
    """
    Lightweight telemetry about bus usage.

    Not performance-critical; used for introspection and future UIs.
    """
    total_published: int = 0
    total_dispatched: int = 0
    total_handler_errors: int = 0
    last_error: Optional[str] = None
    last_error_subscriber: Optional[str] = None
    last_event_topic: Optional[str] = None
    last_event_time: Optional[float] = None


class EventBus:
    """
    Simple topic-based event bus for MicroBrain.

    Responsibilities:
    - maintain subscriptions (name + topics + priority + handler)
    - route a single Event to matching handlers in priority order
    - collect any Events returned by handlers and return them to caller

    IMPORTANT: This class is deliberately "dumb" about queues and main
    loops. The orchestrator will own the event queue and call:

        new_events = await bus.dispatch(event)

    and then re-enqueue those as needed.
    """

    def __init__(self) -> None:
        # subscriber_id -> Subscription
        self._subs_by_id: Dict[int, Subscription] = {}
        self._next_id: int = 1

        # topic -> list of subscriber_ids (kept sorted by priority)
        self._topic_index: Dict[str, List[int]] = {}

        # metrics
        self._metrics = BusMetrics()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> BusMetrics:
        return self._metrics

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _rebuild_topic_index(self) -> None:
        """
        Rebuild the topic index from scratch.

        This is fine for now; subscription counts will be small.
        """
        topic_index: Dict[str, List[int]] = {}
        for sub_id, sub in self._subs_by_id.items():
            if not sub.active:
                continue
            for t in sub.topics:
                topic_index.setdefault(t, []).append(sub_id)

        # Sort each topic list by subscriber priority (high -> low)
        for t, ids in topic_index.items():
            ids.sort(key=lambda sid: self._subs_by_id[sid].priority, reverse=True)

        self._topic_index = topic_index

    def subscribe(
        self,
        name: str,
        topics: Sequence[str],
        handler: EventHandler,
        priority: int = 0,
    ) -> int:
        """
        Register a new subscription and return its numeric ID.

        `topics` is a list of exact topic strings for now.
        """
        sub_id = self._next_id
        self._next_id += 1

        sub = Subscription(
            name=name,
            topics=list(topics),
            priority=priority,
            handler=handler,
        )
        self._subs_by_id[sub_id] = sub
        self._rebuild_topic_index()
        return sub_id

    def unsubscribe(self, sub_id: int) -> None:
        """
        Remove a subscription entirely.
        """
        if sub_id in self._subs_by_id:
            del self._subs_by_id[sub_id]
            self._rebuild_topic_index()

    def set_active(self, sub_id: int, active: bool) -> None:
        """
        Temporarily enable/disable a subscription without removing it.
        """
        sub = self._subs_by_id.get(sub_id)
        if sub is None:
            return
        sub.active = active
        self._rebuild_topic_index()

    def list_subscriptions(self) -> List[Tuple[int, Subscription]]:
        """
        Return a snapshot of all subscriptions (id, Subscription).
        """
        return list(self._subs_by_id.items())

    # ------------------------------------------------------------------
    # Topic resolution
    # ------------------------------------------------------------------

    def _matching_sub_ids(self, topic: str) -> List[int]:
        """
        Return subscriber IDs that want this topic, sorted by priority.

        Exact topic matching remains the neuron-routing default.  A subscriber
        may additionally subscribe to the special ``"*"`` topic to observe the
        complete bus.  This is intended for diagnostics/frontends and does not
        change the topic contract used by neurons.
        """
        exact = list(self._topic_index.get(topic, []))
        observers = list(self._topic_index.get("*", []))
        if not observers:
            return exact

        # De-duplicate a subscriber that intentionally registered both the exact
        # topic and the all-event observer route, then preserve priority order.
        merged = list(dict.fromkeys([*exact, *observers]))
        merged.sort(key=lambda sid: self._subs_by_id[sid].priority, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, event: Event) -> List[Event]:
        """
        Send an Event through the bus.

        For each matching subscriber:
        - await its handler(event)
        - collect any produced Events into a flat list
        - keep basic error metrics but do not stop dispatch on error

        Returns a list of all events produced by all handlers (in the
        order handlers ran, not sorted by topic).
        """
        self._metrics.total_published += 1
        self._metrics.last_event_topic = event.topic
        self._metrics.last_event_time = event.timestamp

        out_events: List[Event] = []
        sub_ids = self._matching_sub_ids(event.topic)

        if not sub_ids:
            # No subscribers; nothing to do.
            return out_events

        for sub_id in sub_ids:
            sub = self._subs_by_id.get(sub_id)
            if sub is None or not sub.active:
                continue

            try:
                produced = await sub.handler(event)
                if produced:
                    for ev in produced:
                        # sanity check: ensure it's an Event instance
                        if isinstance(ev, Event):
                            out_events.append(ev)
                        else:
                            # If someone returns non-Event, we just ignore it.
                            self._metrics.total_handler_errors += 1
                            self._metrics.last_error = (
                                f"Handler {sub.name} returned non-Event: {type(ev)!r}"
                            )
                            self._metrics.last_error_subscriber = sub.name
                self._metrics.total_dispatched += 1
            except Exception as exc:
                # Don't kill the whole system if one handler dies.
                self._metrics.total_handler_errors += 1
                self._metrics.last_error = f"{exc.__class__.__name__}: {exc}"
                self._metrics.last_error_subscriber = sub.name
                # TODO: integrate with a global logger/telemetry sink

        return out_events
