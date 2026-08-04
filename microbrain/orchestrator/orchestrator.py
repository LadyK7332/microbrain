from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from .event_bus import EventBus
from .neuron_base import BaseNeuron, Event
from microbrain.core.attention_controller import AttentionController
from microbrain.policy.policy_engine import PolicyEngine
from microbrain.utils.heartbeat_stream import (
    canonicalize_event_in_place,
    is_infrastructure_event,
    is_infrastructure_topic,
    service_target,
)

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
    """Concrete implementation of the NeuronContext protocol."""

    def __init__(self, emit_callback: Callable[[Event], None], kv_store: Dict[str, Any]):
        self._emit_callback = emit_callback
        self._kv = kv_store

    async def log_debug(self, msg: str, **kwargs: Any) -> None:
        await _log_debug(msg, **kwargs)

    async def log_info(self, msg: str, **kwargs: Any) -> None:
        await _log_info(msg, **kwargs)

    async def log_warn(self, msg: str, **kwargs: Any) -> None:
        await _log_warn(msg, **kwargs)

    async def log_error(self, msg: str, **kwargs: Any) -> None:
        await _log_error(msg, **kwargs)

    async def emit(self, event: Event) -> None:
        # The orchestrator chooses the body or cognitive queue from event class.
        self._emit_callback(event)

    async def get_kv(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    async def set_kv(self, key: str, value: Any) -> None:
        self._kv[key] = value


# =====================================================================================
# Orchestrator: central nervous system + isolated body infrastructure stream
# =====================================================================================


class Orchestrator:
    """MicroBrain runtime.

    Two buses deliberately exist:

    ``bus``
        Meaningful cognitive / perceptual / action events.  Attention and policy
        observe this stream.

    ``body_bus``
        Raw heartbeat and derived body-service cadence only.  It bypasses
        attention, policy, memory telemetry taps, and cognition-wide wildcard
        observers.  Body handlers may still emit a meaningful derived event; the
        queue router moves that result onto ``bus`` automatically.
    """

    def __init__(self):
        # Meaningful nervous-system bus and isolated body/infrastructure bus.
        self.bus = EventBus()
        self.body_bus = EventBus()
        self.kv_store: Dict[str, Any] = {}

        # Policy engine (hard veto / review gates) belongs to meaningful events.
        self.policy = PolicyEngine()
        self.kv_store["policy:engine"] = self.policy
        self.kv_store["policy:last_decision"] = None

        # Attention gate also belongs to meaningful events only.
        self.attention = AttentionController()
        self.kv_store["attention:controller"] = self.attention

        # Separate async queues prevent the raw 20-TPS body clock from becoming
        # part of the cognitive event queue or starving interactive traffic.
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.body_event_queue: asyncio.Queue[Event] = asyncio.Queue()

        self.neurons: Dict[str, BaseNeuron] = {}

        # name -> {"main": id, "body": id}; a neuron may participate in either
        # or both streams without duplicating the underlying heartbeat event.
        self.subscription_ids: Dict[str, Dict[str, int]] = {}

        self.ctx = OrchestratorContext(
            emit_callback=self._queue_event,
            kv_store=self.kv_store,
        )

        self._running = False
        self._run_task: Optional[asyncio.Task] = None
        self._body_run_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Internal queue helper
    # ------------------------------------------------------------------

    def _queue_event(self, event: Event) -> None:
        """Route events to body or cognitive queue from their event class."""
        if not isinstance(event, Event):
            return
        canonicalize_event_in_place(event)
        if is_infrastructure_event(event):
            self.body_event_queue.put_nowait(event)
        else:
            self.event_queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Neuron Registration
    # ------------------------------------------------------------------

    def register_neuron(self, neuron: BaseNeuron) -> None:
        """Attach a neuron, splitting its subscriptions by stream."""
        name = neuron.name
        if name in self.neurons:
            raise RuntimeError(f"Neuron '{name}' already registered.")

        self.neurons[name] = neuron
        main_topics = [topic for topic in neuron.subscribed_topics if not is_infrastructure_topic(topic)]
        body_topics = [topic for topic in neuron.subscribed_topics if is_infrastructure_topic(topic)]
        ids: Dict[str, int] = {}

        if main_topics:
            ids["main"] = self.bus.subscribe(
                name=name,
                topics=main_topics,
                handler=lambda event, n=neuron: n.handle_event(event, self.ctx),
                priority=neuron.priority,
            )

        if body_topics:
            ids["body"] = self.body_bus.subscribe(
                name=name,
                topics=body_topics,
                handler=lambda event, n=neuron: n.handle_event(event, self.ctx),
                priority=neuron.priority,
            )

        self.subscription_ids[name] = ids

        # The body scheduler can avoid emitting service topics nobody currently
        # consumes. This is runtime telemetry/state only, never durable memory.
        targets = set(self.kv_store.get("body:service_targets", []) or [])
        for topic in body_topics:
            target = service_target(topic)
            if target:
                targets.add(target)
        self.kv_store["body:service_targets"] = sorted(targets)

    def unregister_neuron(self, name: str) -> None:
        if name not in self.neurons:
            return

        ids = self.subscription_ids.get(name, {}) or {}
        if ids.get("main") is not None:
            self.bus.unsubscribe(ids["main"])
        if ids.get("body") is not None:
            self.body_bus.unsubscribe(ids["body"])

        del self.neurons[name]
        self.subscription_ids.pop(name, None)

        # Recompute active service targets from remaining neuron subscriptions.
        targets: set[str] = set()
        for neuron in self.neurons.values():
            for topic in neuron.subscribed_topics:
                target = service_target(topic)
                if target:
                    targets.add(target)
        self.kv_store["body:service_targets"] = sorted(targets)

    # ------------------------------------------------------------------
    # Public API
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
        """Entry point for external systems and the body pacemaker.

        Legacy ``clock/tick`` producers are canonicalized to ``body/heartbeat``
        and routed to the body bus exactly once.
        """
        kwargs: Dict[str, Any] = {
            "topic": topic,
            "payload": payload,
            "source": source,
            "meta": dict(meta or {}),
        }
        if correlation_id:
            kwargs["correlation_id"] = correlation_id
        self._queue_event(Event(**kwargs))

    async def push_body_event(
        self,
        topic: str,
        payload: Any,
        meta: Dict[str, Any] | None = None,
        *,
        source: str = "",
        correlation_id: str | None = None,
    ) -> None:
        """Explicit body-stream entry point; useful for pacemakers/tests."""
        await self.push_event(
            topic,
            payload,
            meta=meta,
            source=source,
            correlation_id=correlation_id,
        )

    # ------------------------------------------------------------------
    # Meaningful/cognitive run loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        await self.ctx.log_info("MicroBrain orchestrator started")

        while self._running:
            event: Event | None = None
            try:
                event = await self.event_queue.get()
                try:
                    # Hard stream guard: infrastructure never belongs in the head.
                    if is_infrastructure_event(event):
                        self.body_event_queue.put_nowait(event)
                        continue

                    self.attention.observe_event(event)
                    self.attention.update_allow_babble()

                    decision = self.policy.evaluate_event(event)
                    self.kv_store["policy:last_decision"] = decision.to_dict()

                    if decision.status == "veto":
                        await self.ctx.log_warn(
                            "Policy veto",
                            topic=event.topic,
                            rule_id=decision.rule_id,
                            reason=decision.reason,
                            source=event.source,
                        )
                        continue

                    if decision.status == "needs_review":
                        await self.ctx.log_info(
                            "Policy needs review",
                            topic=event.topic,
                            rule_id=decision.rule_id,
                            reason=decision.reason,
                            source=event.source,
                        )
                        if event.topic.startswith("act/") and event.topic != "act/speech":
                            self._queue_event(
                                Event(
                                    topic="act/speech",
                                    payload={
                                        "text": f"[policy] Paused: {decision.reason} ({decision.rule_id})",
                                        "style": "system",
                                    },
                                    source="policy_engine",
                                    correlation_id=event.correlation_id,
                                    meta={"kind": "policy_notice"},
                                )
                            )
                        continue

                    for produced in await self.bus.dispatch(event):
                        self._queue_event(produced)
                finally:
                    self.event_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                await self.ctx.log_error(
                    "Unhandled orchestrator loop exception",
                    exception=str(exc),
                )

        await self.ctx.log_info("MicroBrain orchestrator stopped")

    # ------------------------------------------------------------------
    # Body/infrastructure run loop
    # ------------------------------------------------------------------

    async def _run_body_loop(self) -> None:
        await self.ctx.log_info("MicroBrain body infrastructure bus started")

        while self._running:
            event: Event | None = None
            try:
                event = await self.body_event_queue.get()
                try:
                    canonicalize_event_in_place(event)

                    # Fail closed: if a meaningful event accidentally enters the body
                    # queue, return it to the normal nervous-system stream.
                    if not is_infrastructure_event(event):
                        self.event_queue.put_nowait(event)
                        continue

                    for produced in await self.body_bus.dispatch(event):
                        self._queue_event(produced)
                finally:
                    self.body_event_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                await self.ctx.log_error(
                    "Unhandled body bus exception",
                    exception=str(exc),
                )

        await self.ctx.log_info("MicroBrain body infrastructure bus stopped")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()
        self._run_task = loop.create_task(self._run_loop(), name="microbrain_cognitive_bus")
        self._body_run_task = loop.create_task(self._run_body_loop(), name="microbrain_body_bus")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        tasks = [task for task in (self._run_task, self._body_run_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._run_task = None
        self._body_run_task = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def wait_for_idle(self, timeout: float = 1.0) -> bool:
        """Return True after both streams finish all queued/in-flight work."""
        try:
            await asyncio.wait_for(
                asyncio.gather(self.event_queue.join(), self.body_event_queue.join()),
                timeout=max(0.01, float(timeout)),
            )
            return True
        except asyncio.TimeoutError:
            return False
