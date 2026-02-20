from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol, Callable

from .event_bus import EventBus
from .neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.core.attention_controller import AttentionController

# =====================================================================================
# Logging backend placeholder (upgradable later)
# =====================================================================================
_logger = logging.getLogger("microbrain.orchestrator")

async def _log_debug(msg: str, **kwargs: Any) -> None:
    _logger.debug("%s | %s", msg, kwargs)

async def _log_info(msg: str, **kwargs: Any) -> None:
    _logger.info("%s | %s", msg, kwargs)

async def _log_warn(msg: str, **kwargs: Any) -> None:
    _logger.warning("%s | %s", msg, kwargs)

async def _log_error(msg: str, **kwargs: Any) -> None:
    _logger.error("%s | %s", msg, kwargs)


# =====================================================================================
# Orchestrator Context (what neurons "see")
# =====================================================================================

class OrchestratorContext:
    """
    Concrete implementation of NeuronContext protocol.

    This is what neurons use to:
    - emit events
    - log
    - read/write KV state
    """

    def __init__(self, emit_callback: Callable[[Event], None], kv_store: Dict[str, Any]):
        self._emit_callback = emit_callback
        self._kv = kv_store

    # ---------------------- Logging ----------------------

    async def log_debug(self, msg: str, **kwargs: Any) -> None:
        await _log_debug(msg, **kwargs)

    async def log_info(self, msg: str, **kwargs: Any) -> None:
        await _log_info(msg, **kwargs)

    async def log_warn(self, msg: str, **kwargs: Any) -> None:
        await _log_warn(msg, **kwargs)

    async def log_error(self, msg: str, **kwargs: Any) -> None:
        await _log_error(msg, **kwargs)

    # ---------------------- Emit ----------------------

    async def emit(self, event: Event) -> None:
        """
        Called by neurons. This does NOT go through the bus directly —
        it just pushes into the orchestrator queue.
        """
        self._emit_callback(event)

    # ---------------------- KV Store ----------------------

    async def get_kv(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    async def set_kv(self, key: str, value: Any) -> None:
        self._kv[key] = value


# =====================================================================================
# Orchestrator: the central nervous system
# =====================================================================================

class Orchestrator:
    """
    The main MicroBrain orchestrator:

    - Owns the EventBus
    - Owns the async event queue
    - Owns the global KV store
    - Holds neuron instances
    - Dispatches events
    - Feeds new events back into the queue
    - Manages run loop, backpressure, and safe shutdown

    This is NOT the "app" — it is the system runtime.
    """

    def __init__(self):
        # Pipes & storage
        self.bus = EventBus()
        self.kv_store: Dict[str, Any] = {}

        # Attention gate for external vs internal speech
        self.attention = AttentionController()
        self.kv_store["attention:controller"] = self.attention

        # Async event queue
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()

        # All live neurons (name -> instance)
        self.neurons: Dict[str, BaseNeuron] = {}

        # Sub IDs mapped to neuron names
        self.subscription_ids: Dict[str, int] = {}

        # Let neurons emit by pushing into queue
        self.ctx = OrchestratorContext(
            emit_callback=self._queue_event,
            kv_store=self.kv_store,
        )

        # Run loop state
        self._running = False
        self._run_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Internal queue helper
    # ------------------------------------------------------------------

    def _queue_event(self, event: Event) -> None:
        """
        Safe push into queue from anywhere (neurons, inputs, etc.)
        """
        self.event_queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Neuron Registration
    # ------------------------------------------------------------------

    def register_neuron(self, neuron: BaseNeuron) -> None:
        """
        Attach a neuron to this runtime, subscribe it to the EventBus,
        sort by priority, etc.
        """
        name = neuron.name

        if name in self.neurons:
            raise RuntimeError(f"Neuron '{name}' already registered.")

        self.neurons[name] = neuron

        sub_id = self.bus.subscribe(
            name=name,
            topics=neuron.subscribed_topics,
            handler=lambda event, n=neuron: n.handle_event(event, self.ctx),
            priority=neuron.priority,
        )

        self.subscription_ids[name] = sub_id

    def unregister_neuron(self, name: str) -> None:
        """
        Remove neuron + its subscription fully.
        """
        if name not in self.neurons:
            return

        sub_id = self.subscription_ids.get(name)
        if sub_id is not None:
            self.bus.unsubscribe(sub_id)

        del self.neurons[name]
        self.subscription_ids.pop(name, None)

    # ------------------------------------------------------------------
    # Public API to feed external events (e.g. UI, Minecraft, sensors)
    # ------------------------------------------------------------------

    async def push_event(
        self,
        topic: str,
        payload: Any,
        meta: Dict[str, Any] | None = None,
        *,
        source: str = "",
        correlation_id: str | None = None,
    ) -> None:
        """External system entry point.

        Optional knobs:
        - source: set Event.source (helps attribution in logs/neurons)
        - correlation_id: propagate a request/response trace across the bus
        """
        if meta is None:
            meta = {}

        kwargs: Dict[str, Any] = {
            "topic": topic,
            "payload": payload,
            "source": source,
            "meta": meta,
        }
        if correlation_id:
            kwargs["correlation_id"] = correlation_id

        ev = Event(**kwargs)
        self._queue_event(ev)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """
        Single-threaded brainstem loop:

        - waits for event
        - dispatches via bus
        - enqueues resulting events
        - repeats
        """
        await self.ctx.log_info("MicroBrain orchestrator started")

        while self._running:
            try:
                # Block until an event arrives
                event = await self.event_queue.get()

                # Update attention gate based on external stimuli
                self.attention.observe_event(event)
                self.attention.update_allow_babble()

                # Dispatch through EventBus
                new_events = await self.bus.dispatch(event)

                # Feed results back into event queue
                for ev in new_events:
                    self._queue_event(ev)

            except Exception as exc:
                await self.ctx.log_error(
                    "Unhandled orchestrator loop exception",
                    exception=str(exc),
                )

        await self.ctx.log_info("MicroBrain orchestrator stopped")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._run_task = loop.create_task(self._run_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._run_task:
            await self._run_task

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def wait_for_idle(self, timeout: float = 1.0) -> bool:
        """
        Returns True if queue empties within timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.event_queue.empty():
                return True
            await asyncio.sleep(0.01)
        return False
