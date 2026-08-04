# C:\aiproj\microbrain\orchestrator\neuron_base.py

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
)
from abc import ABC, abstractmethod

from microbrain.utils.heartbeat_stream import (
    PRIMARY_HEARTBEAT_TOPIC,
    canonical_subscription_topic,
    canonical_topic,
    is_infrastructure_event,
)

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

# Infrastructure pulses are scheduler/metabolism triggers, not semantic input.
# They may wake explicitly subscribed organs, but they must never gain Hebbian
# significance merely because they occur frequently.
NON_SEMANTIC_INPUT_TOPICS = frozenset({"clock/tick", PRIMARY_HEARTBEAT_TOPIC})
NON_SEMANTIC_EVENT_CLASSES = frozenset({"infrastructure"})

# ---------------------------------------------------------------------------
# Core Event type
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    Generic event flowing through MicroBrain's event bus.

    topic:          routing key, e.g. "percept/text", "act/speech"
    payload:        arbitrary data; conventionally a dict or str
    timestamp:      event creation time (epoch seconds)
    source:         who/what emitted this event
    correlation_id: used to trace chains of related events
    meta:           extra metadata (confidence, tags, goal hints, etc.)
    """
    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Neuron runtime context (orchestrator-side capabilities)
# ---------------------------------------------------------------------------

class NeuronContext(Protocol):
    """
    Minimal interface that the orchestrator exposes to neurons.

    The actual orchestrator implementation will provide a concrete object
    that satisfies this Protocol.
    """

    async def emit(self, event: Event) -> None:
        """Publish a new event back onto the bus."""

    async def log_debug(self, msg: str, **kwargs: Any) -> None:
        ...

    async def log_info(self, msg: str, **kwargs: Any) -> None:
        ...

    async def log_warn(self, msg: str, **kwargs: Any) -> None:
        ...

    async def log_error(self, msg: str, **kwargs: Any) -> None:
        ...

    async def get_kv(self, key: str, default: Any = None) -> Any:
        """Lightweight key-value store (backed by memory subsystem)."""

    async def set_kv(self, key: str, value: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# Config & activation tracing
# ---------------------------------------------------------------------------

@dataclass
class NeuronConfig:
    """
    Tunable parameters shared by all neurons.

    Subclasses can extend this via composition or by adding fields
    on their own __init__.
    """
    name: str
    subscribed_topics: Sequence[str]
    output_topics: Sequence[str] = field(default_factory=list)

    # Scheduling / priority
    priority: int = 0                  # higher runs first when multiple fire
    cooldown_sec: float = 0.0          # minimum time between firings

    # Hebbian & PDNA knobs
    hebbian_learning_rate: float = 0.1
    hebbian_decay_rate: float = 0.001  # per-second passive decay
    pdna_bias: Dict[str, float] = field(default_factory=dict)

    # Goal homeostasis channels (e.g. ["power", "maintenance", "civ_dev"])
    goal_channels: Sequence[str] = field(default_factory=list)

    # Debug / tracing
    max_activation_history: int = 64   # how many activations to remember


@dataclass
class ActivationRecord:
    """
    Lightweight trace of a single neuron activation.

    Helps with debugging and future visualization.
    """
    timestamp: float
    topic: str
    input_meta: Dict[str, Any]
    outputs_count: int
    latency_ms: float
    hebbian_context: str
    hebbian_weight_after: float


# ---------------------------------------------------------------------------
# Base neuron
# ---------------------------------------------------------------------------

class BaseNeuron(ABC):
    def __init__(self, config: NeuronConfig):
        self.config = config
        name = config.name
        self._name = name or self.__class__.__name__

        # subscription & output bookkeeping
        self._subscribed_topics: set[str] = {
            canonical_subscription_topic(topic) for topic in (config.subscribed_topics or [])
        }
        self._output_topics: List[str] = list(config.output_topics or [])

        # Hebbian weights keyed by a simple context key (e.g. topic name)
        self._hebbian_weights: Dict[str, float] = {}

        # Last time this neuron fired (for cooldown)
        self._last_fire_time: float = 0.0

        # Recent activation history (for debugging / visualization)
        self._activation_history: Deque[ActivationRecord] = deque(
            maxlen=config.max_activation_history
        )

    def debug(self, message: str, **fields) -> None:
        """
        Lightweight, global-flag-gated debug helper.
        Safe to call from any neuron.
        """
        from microbrain.orchestrator.debug_utils import is_debug_enabled

        if not is_debug_enabled():
            return

        extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
        # You can swap this to logging if you prefer
        print(f"[NEURON-DEBUG][{self.name}] {message} {extra}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self.config.priority

    @property
    def subscribed_topics(self) -> tuple[str, ...]:
        return tuple(self._subscribed_topics)

    def subscribe(self, *topics: str) -> None:
        self._subscribed_topics.update(canonical_subscription_topic(topic) for topic in topics)

    @property
    def output_topics(self) -> Sequence[str]:
        return tuple(self._output_topics)

    def get_activation_history(self) -> Sequence[ActivationRecord]:
        """Return recent activation records (for debug/telemetry UIs)."""
        return tuple(self._activation_history)

    # ------------------------------------------------------------------
    # Topic matching & cooldown
    # ------------------------------------------------------------------

    def matches_topic(self, topic: str) -> bool:
        """
        Basic topic matcher.

        For now it's simple equality. Later we can extend to:
        - wildcard segments (e.g. "percept/*")
        - regex
        - tag-based routing
        """
        return canonical_topic(topic) in self._subscribed_topics

    def _is_in_cooldown(self, now: Optional[float] = None) -> bool:
        if self.config.cooldown_sec <= 0:
            return False
        if now is None:
            now = time.time()
        return (now - self._last_fire_time) < self.config.cooldown_sec

    # ------------------------------------------------------------------
    # Hebbian handling
    # ------------------------------------------------------------------

    def _is_semantic_input(self, event: Event) -> bool:
        """Return False for scheduler/telemetry events that are not cognition.

        These events can still be delivered to neurons that explicitly subscribe
        to them.  The distinction only prevents infrastructure frequency from
        masquerading as semantic/associative evidence.
        """
        meta = event.meta if isinstance(event.meta, dict) else {}
        if is_infrastructure_event(event):
            return False
        if event.topic in NON_SEMANTIC_INPUT_TOPICS:
            return False
        if meta.get("semantic_input") is False:
            return False
        event_class = str(meta.get("event_class", "") or "").strip().lower()
        if event_class in NON_SEMANTIC_EVENT_CLASSES:
            return False
        return True

    def _event_reinforcement_eligible(self, event: Event) -> bool:
        """Whether an input event may earn base Hebbian reinforcement."""
        if not self._is_semantic_input(event):
            return False
        meta = event.meta if isinstance(event.meta, dict) else {}
        return meta.get("reinforcement_eligible") is not False

    def _hebb_context_key(self, event: Event) -> str:
        """
        Derive a simple semantic context key from the event.

        Infrastructure triggers intentionally return an empty key so they remain
        schedulers rather than learned associations.  Subclasses can override this
        for richer semantic contexts, but should preserve this separation.
        """
        if not self._is_semantic_input(event):
            return ""
        return event.topic

    def get_hebbian_weight(self, key: str) -> float:
        return self._hebbian_weights.get(key, 0.0)

    def _apply_hebbian_decay(self, now: Optional[float] = None) -> None:
        """
        Passive linear-ish decay of weights over time.

        This keeps old associations from dominating forever.
        """
        if not self._hebbian_weights:
            return

        if now is None:
            now = time.time()

        decay_rate = self.config.hebbian_decay_rate
        if decay_rate <= 0:
            return

        to_delete: List[str] = []
        for key, weight in self._hebbian_weights.items():
            new_weight = weight - decay_rate
            if abs(new_weight) < 1e-6:
                to_delete.append(key)
            else:
                self._hebbian_weights[key] = new_weight

        for key in to_delete:
            del self._hebbian_weights[key]

    def reinforce_context(self, key: str, amount: Optional[float] = None) -> float:
        """
        Strengthen the association for this context key.

        Returns the new weight.
        """
        if amount is None:
            amount = self.config.hebbian_learning_rate

        current = self._hebbian_weights.get(key, 0.0)
        new_weight = current + amount
        self._hebbian_weights[key] = new_weight
        return new_weight

    # ------------------------------------------------------------------
    # PDNA helpers
    # ------------------------------------------------------------------

    def pdna_adjust_score(self, score: float, channels: Sequence[str]) -> float:
        """
        Adjust a scalar 'score' using PDNA bias values for provided channels.

        Example:
            score = neuron.pdna_adjust_score(score, ["excited", "curious"])

        PDNA is just a bias look-up for now; the higher-level PDNA lattice
        can be wired in later by updating config.pdna_bias dynamically.
        """
        bias = 0.0
        for ch in channels:
            bias += self.config.pdna_bias.get(ch, 0.0)
        return score + bias

    # ------------------------------------------------------------------
    # Memory / state helpers (KV-based, backed by ctx)
    # ------------------------------------------------------------------

    def _state_key(self, suffix: str) -> str:
        """
        Compose a namespaced state key so different neurons don't collide.

        Example: "neuron:EchoNeuron:echo:last_user"
        """
        return f"neuron:{self.__class__.__name__}:{self.name}:{suffix}"

    async def load_state(
        self,
        ctx: NeuronContext,
        suffix: str,
        default: Any = None,
    ) -> Any:
        """Convenience wrapper around ctx.get_kv with a namespaced key."""
        key = self._state_key(suffix)
        return await ctx.get_kv(key, default=default)

    async def save_state(
        self,
        ctx: NeuronContext,
        suffix: str,
        value: Any,
    ) -> None:
        """Convenience wrapper around ctx.set_kv with a namespaced key."""
        key = self._state_key(suffix)
        await ctx.set_kv(key, value)

    # ------------------------------------------------------------------
    # Goal / homeostasis hooks (stubbed for now)
    # ------------------------------------------------------------------

    async def evaluate_goals(
        self,
        event: Event,
        proposed_outputs: Sequence[Event],
        ctx: NeuronContext,
    ) -> Sequence[Event]:
        """
        Hook for Goal Homeostasis.

        For now this is a no-op pass-through, but the orchestrator can
        later provide shared goal state that this method consults via
        ctx.get_kv(...) and attenuates or drops outputs that conflict
        with global drives (power, maintenance, human civilization dev).
        """
        # Example future shape (pseudo):
        # goals = await ctx.get_kv("global:goals", default={})
        # adjust or filter proposed_outputs based on self.config.goal_channels
        return proposed_outputs

    # ------------------------------------------------------------------
    # Activation tracing
    # ------------------------------------------------------------------

    def _record_activation(
        self,
        timestamp: float,
        event: Event,
        outputs: Sequence[Event],
        latency_ms: float,
        hebbian_context: str,
        hebbian_weight_after: float,
    ) -> None:
        rec = ActivationRecord(
            timestamp=timestamp,
            topic=event.topic,
            input_meta=dict(event.meta),
            outputs_count=len(outputs),
            latency_ms=latency_ms,
            hebbian_context=hebbian_context,
            hebbian_weight_after=hebbian_weight_after,
        )
        self._activation_history.append(rec)

    # ------------------------------------------------------------------
    # Main entry point from orchestrator
    # ------------------------------------------------------------------

    async def handle_event(self, event: Event, ctx: NeuronContext) -> Iterable[Event]:
        """
        Orchestrator calls this to let the neuron react to an event.

        The default implementation:
        - checks topic subscription
        - enforces cooldown
        - applies Hebbian decay
        - invokes the subclass `process()`
        - passes outputs through goal/homeostasis hook
        - reinforces context if any outputs were produced
        - records activation trace
        """
        now = time.time()

        if not self.matches_topic(event.topic):
            return []

        infrastructure_input = is_infrastructure_event(event)

        # Semantic cooldown and body cadence are independent clocks. A recent
        # cognitive/perceptual firing must not suppress housekeeping, and a body
        # service opportunity must never consume the semantic cooldown itself.
        if not infrastructure_input and self._is_in_cooldown(now):
            await ctx.log_debug(
                f"[{self.name}] Skipping event due to cooldown",
                topic=event.topic,
            )
            return []

        # Body/infrastructure pulses are scheduling only. They must not mutate
        # associative/Hebbian state merely because they occur frequently.
        if not infrastructure_input:
            self._apply_hebbian_decay(now)

        start = time.perf_counter()

        # Let subclass do actual work
        raw_outputs = await self.process(event, ctx)
        raw_outputs = list(raw_outputs or [])

        # Potential goal/homeostasis gating
        gated_outputs = list(
            await self.evaluate_goals(event, raw_outputs, ctx)
        )

        # A body service pulse may legitimately cause a meaningful state event
        # (for example a cooldown expiring), but the heartbeat/service correlation
        # itself must never become the cognitive trace identity. Detach any
        # inherited correlation at this hard boundary.
        if infrastructure_input:
            for output in gated_outputs:
                if not isinstance(output, Event) or is_infrastructure_event(output):
                    continue
                if output.correlation_id == event.correlation_id:
                    output.correlation_id = uuid.uuid4().hex
                    if not isinstance(output.meta, dict):
                        output.meta = {}
                    output.meta.setdefault("infrastructure_correlation_detached", True)

        # Only semantic, explicitly eligible inputs may earn Hebbian weight.
        # Scheduler/infrastructure pulses can still drive organ maintenance, but
        # their frequency must never be mistaken for cognitive significance.
        context_key = self._hebb_context_key(event)
        new_weight = self.get_hebbian_weight(context_key) if context_key else 0.0
        if gated_outputs and context_key and self._event_reinforcement_eligible(event):
            new_weight = self.reinforce_context(context_key)
            await ctx.log_debug(
                f"[{self.name}] Reinforced context",
                context_key=context_key,
                new_weight=new_weight,
            )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0

        # Infrastructure cadence must not crowd semantic activation history.
        # It can be explicitly traced during scheduler debugging with
        # meta["trace_activation"] = True.
        trace_infrastructure = bool((event.meta or {}).get("trace_activation", False))
        if not infrastructure_input or trace_infrastructure:
            self._record_activation(
                timestamp=now,
                event=event,
                outputs=gated_outputs,
                latency_ms=latency_ms,
                hebbian_context=context_key,
                hebbian_weight_after=new_weight,
            )

        # Likewise, a service pulse must not consume a neuron's semantic cooldown
        # and accidentally suppress a real percept arriving a few milliseconds later.
        if not infrastructure_input:
            self._last_fire_time = now
        return gated_outputs

    # ------------------------------------------------------------------
    # To be implemented by subclasses
    # ------------------------------------------------------------------

    async def process(self, event: Event, ctx: NeuronContext) -> Iterable[Event]:
        """
        Override this in subclasses.

        You get the incoming event and a context object to:
        - emit new events
        - access memory
        - log
        """
        raise NotImplementedError(f"{self.__class__.__name__}.process() not implemented")
