"""Bridge between the running MicroBrain body and the native dashboard.

The dashboard observes the event bus and requests changes through this bridge.
It does not reach into neuron instances or private organ attributes.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping

from microbrain.orchestrator.neuron_base import Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.ui.frontend_common import (
    DASHBOARD_SNAPSHOT_SCHEMA,
    UIMessage,
    pressure_snapshot,
    resolve_ui_memdir,
    runtime_tuning_candidates,
    safe_json,
)
from microbrain.utils.heartbeat_stream import PRIMARY_HEARTBEAT_TOPIC, is_heartbeat_event

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

# Queue pressure controls for the diagnostic firehose.
DASHBOARD_EVENT_QUEUE_MAX = 2000
DASHBOARD_DRAIN_DROP_BATCH = 32

# UI telemetry cadence.  Units: seconds.
PRESSURE_SAMPLE_INTERVAL_S = 0.25
RUNTIME_SNAPSHOT_INTERVAL_S = 0.50

# Raw clock ticks are intentionally omitted from the visible firehose because
# they arrive twice per second forever.  Bus metrics still expose tick activity.
SHOW_CLOCK_TICK_EVENTS = False

# Runtime tuning audit is capped in KV while the full history remains in the
# memdir log.  Unit: entries.
TUNING_HISTORY_LIMIT = 128

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

DASHBOARD_TAP_NAME = "ui.dashboard.tap"
DASHBOARD_PRESSURE_TOPIC = "ui/pressure_state"
DASHBOARD_SNAPSHOT_TOPIC = "ui/dashboard_snapshot"
DASHBOARD_TUNING_CHANGED_TOPIC = "control/runtime_tuning_changed"
DASHBOARD_TUNING_AUDIT_FILENAME = "dashboard_tuning_changes.jsonl"


class DashboardBridge:
    """Transport-neutral dashboard bridge around an already running orchestrator."""

    def __init__(self, orch: Orchestrator, *, memdir: str | None = None) -> None:
        self.orch = orch
        self.memdir = str(resolve_ui_memdir(memdir))
        self.recv_q: asyncio.Queue[UIMessage] = asyncio.Queue(maxsize=DASHBOARD_EVENT_QUEUE_MAX)
        self._tap_id: int | None = None
        self._tasks: list[asyncio.Task] = []
        self._dropped_events = 0
        self._started = False
        self._last_noninfra_topic = ""
        self._last_noninfra_time = 0.0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._tap_id = self.orch.bus.subscribe(
            DASHBOARD_TAP_NAME,
            ["*"],
            self._tap_event,
            priority=-1000,
        )
        self._tasks = [
            asyncio.create_task(self._pressure_pump(), name="dashboard_pressure_sampler"),
            asyncio.create_task(self._snapshot_pump(), name="dashboard_runtime_sampler"),
        ]
        self.orch.kv_store["ui:dashboard:started"] = True
        self.orch.kv_store["outlet:textual_available"] = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._tap_id is not None:
            try:
                self.orch.bus.unsubscribe(self._tap_id)
            except Exception:
                pass
            self._tap_id = None
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks.clear()
        self.orch.kv_store["ui:dashboard:started"] = False

    async def _tap_event(self, event: Event) -> list[Event]:
        if is_heartbeat_event(event) and not SHOW_CLOCK_TICK_EVENTS:
            return []
        if not is_heartbeat_event(event):
            self._last_noninfra_topic = event.topic
            self._last_noninfra_time = float(event.timestamp or time.time())
        await self._enqueue(
            UIMessage(
                topic=event.topic,
                payload=event.payload,
                source=event.source,
                meta=dict(event.meta or {}),
                correlation_id=str(event.correlation_id or ""),
                timestamp=float(event.timestamp or time.time()),
            )
        )
        return []

    async def _enqueue(self, message: UIMessage) -> None:
        if not self.recv_q.full():
            self.recv_q.put_nowait(message)
            return
        # Keep the face responsive by dropping oldest diagnostic packets.  Never
        # block the brainstem on a human-facing monitor.
        for _ in range(DASHBOARD_DRAIN_DROP_BATCH):
            try:
                self.recv_q.get_nowait()
                self._dropped_events += 1
            except asyncio.QueueEmpty:
                break
        try:
            self.recv_q.put_nowait(message)
        except asyncio.QueueFull:
            self._dropped_events += 1

    async def _pressure_pump(self) -> None:
        while True:
            await self._enqueue(
                UIMessage(
                    topic=DASHBOARD_PRESSURE_TOPIC,
                    payload=pressure_snapshot(self.orch),
                    source="ui.dashboard.pressure_sampler",
                    meta={"ui_hidden": True, "store_in_memory": False},
                    timestamp=time.time(),
                )
            )
            await asyncio.sleep(PRESSURE_SAMPLE_INTERVAL_S)

    async def _snapshot_pump(self) -> None:
        while True:
            await self._enqueue(
                UIMessage(
                    topic=DASHBOARD_SNAPSHOT_TOPIC,
                    payload=self.runtime_snapshot(),
                    source="ui.dashboard.runtime_sampler",
                    meta={"ui_hidden": True, "store_in_memory": False},
                    timestamp=time.time(),
                )
            )
            await asyncio.sleep(RUNTIME_SNAPSHOT_INTERVAL_S)

    async def send_text(self, text: str) -> None:
        prompt = (text or "").strip()
        if not prompt:
            return
        await self.orch.push_event(
            "input/text",
            prompt,
            meta={"source": "ui", "channel": "textual", "frontend": "dashboard"},
            source="dashboard",
        )
        # Preserve Textual behavior: let the brain chew without blocking the GUI.
        await self.orch.wait_for_idle(timeout=30.0)

    def drain_nowait(self, *, limit: int = 100) -> list[UIMessage]:
        messages: list[UIMessage] = []
        for _ in range(max(1, int(limit))):
            try:
                messages.append(self.recv_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def runtime_snapshot(self) -> dict[str, Any]:
        kv = self.orch.kv_store
        metrics = self.orch.bus.metrics
        ddna_profile = kv.get("pdna:ddna_mutators") or {}
        ddna_effective = kv.get("drive:ddna_modulators") or {}
        return {
            "schema": DASHBOARD_SNAPSHOT_SCHEMA,
            "ts": time.time(),
            "queue_depth": int(self.orch.event_queue.qsize()),
            "dashboard_queue_depth": int(self.recv_q.qsize()),
            "dashboard_dropped_events": int(self._dropped_events),
            "neurons": len(self.orch.neurons),
            "bus": {
                "total_published": int(metrics.total_published),
                "total_dispatched": int(metrics.total_dispatched),
                "handler_errors": int(metrics.total_handler_errors),
                "last_error": metrics.last_error,
                "last_error_subscriber": metrics.last_error_subscriber,
                "last_event_topic": metrics.last_event_topic,
                "last_event_time": metrics.last_event_time,
                "last_noninfra_event_topic": self._last_noninfra_topic,
                "last_noninfra_event_time": self._last_noninfra_time,
                "heartbeat_primary_topic": PRIMARY_HEARTBEAT_TOPIC,
            },
            "organs": {
                "memory_composer": bool(kv.get("mem_cell:composer:started", False)),
                "read_sidecar": bool(kv.get("read:sidecar_started", False)),
                "slearn_sidecar": bool(kv.get("slearn:sidecar_started", False)),
                "power_mode": kv.get("power:mode", "awake"),
                "vision_enabled": bool(kv.get("vision:enabled", False)),
                "tts_enabled": bool(kv.get("tts:enabled", False)),
            },
            "runtime_tunables": runtime_tuning_candidates(kv),
            "ddna": safe_json(ddna_profile),
            "ddna_effective": safe_json(ddna_effective),
            "hypothesis": safe_json(kv.get("hypothesis:last", {})),
            "hypothesis_tuning": safe_json(kv.get("hypothesis:last_tuning", {})),
            "release_tuning": safe_json(kv.get("hypothesis:release_tuning", {})),
            "scene": safe_json(kv.get("scene:current", {})),
            "scene_exp": safe_json(kv.get("scene:expectation:last_exp", {})),
            "scene_delta": safe_json(kv.get("scene:expectation:last_delta", {})),
        }

    @staticmethod
    def parse_runtime_value(text: str, previous: Any) -> Any:
        raw = (text or "").strip()
        if isinstance(previous, bool):
            lower = raw.lower()
            if lower in {"1", "true", "yes", "on"}:
                return True
            if lower in {"0", "false", "no", "off"}:
                return False
            raise ValueError("boolean value must be true/false, yes/no, on/off, or 1/0")
        if isinstance(previous, int) and not isinstance(previous, bool):
            return int(raw)
        if isinstance(previous, float):
            return float(raw)
        if previous is None:
            try:
                return json.loads(raw)
            except Exception:
                return raw
        return raw

    async def set_runtime_tuning(self, key: str, text_value: str) -> dict[str, Any]:
        """Apply one runtime-KV tuning change through the dashboard bridge.

        DDNA is intentionally not editable here.  The dashboard only exposes
        runtime tuning keys discovered by :func:`runtime_tuning_candidates`.
        """

        candidates = runtime_tuning_candidates(self.orch.kv_store)
        if key not in candidates:
            raise ValueError(f"{key!r} is not an exposed runtime tuning key")
        old = candidates[key]
        new = self.parse_runtime_value(text_value, old)
        self.orch.kv_store[key] = new

        record = {
            "ts": time.time(),
            "key": key,
            "old": safe_json(old),
            "new": safe_json(new),
            "source": "dashboard",
        }
        history = self.orch.kv_store.get("dashboard:tuning_history", [])
        history = list(history) if isinstance(history, list) else []
        history.append(record)
        self.orch.kv_store["dashboard:tuning_history"] = history[-TUNING_HISTORY_LIMIT:]
        self.orch.kv_store["dashboard:last_tuning_change"] = record
        self._append_tuning_audit(record)

        await self.orch.push_event(
            DASHBOARD_TUNING_CHANGED_TOPIC,
            record,
            meta={"source": "ui", "channel": "dashboard", "store_in_memory": False},
            source="dashboard_bridge",
        )
        return record

    def _append_tuning_audit(self, record: Mapping[str, Any]) -> None:
        try:
            path = Path(self.memdir) / "logs" / DASHBOARD_TUNING_AUDIT_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            pass
