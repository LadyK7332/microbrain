from __future__ import annotations

import asyncio

from microbrain.orchestrator.event_bus import EventBus
from microbrain.orchestrator.neuron_base import Event


def test_event_bus_all_event_observer_sees_exact_topic() -> None:
    async def run() -> None:
        bus = EventBus()
        seen: list[tuple[str, str]] = []

        async def exact(event: Event):
            seen.append(("exact", event.topic))
            return []

        async def tap(event: Event):
            seen.append(("tap", event.topic))
            return []

        bus.subscribe("exact", ["input/text"], exact, priority=10)
        bus.subscribe("tap", ["*"], tap, priority=-10)
        await bus.dispatch(Event("input/text", "hello"))
        assert seen == [("exact", "input/text"), ("tap", "input/text")]

    asyncio.run(run())


def test_wildcard_and_exact_on_same_subscription_do_not_double_dispatch() -> None:
    async def run() -> None:
        bus = EventBus()
        count = 0

        async def handler(event: Event):
            nonlocal count
            count += 1
            return []

        bus.subscribe("both", ["*", "input/text"], handler)
        await bus.dispatch(Event("input/text", "hello"))
        assert count == 1

    asyncio.run(run())
