"""Bridge between the orchestrator and the Textual UI."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from microbrain.orchestrator.event_bus import Event
from microbrain.orchestrator.orchestrator import Orchestrator

from .textual_app import MicroBrainUI, UIMessage


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _age_decay(now: float, ts: Any, *, ttl_s: float) -> float:
    then = _safe_float(ts, 0.0)
    if then <= 0.0:
        return 0.0
    age = max(0.0, now - then)
    if age >= ttl_s:
        return 0.0
    return max(0.0, 1.0 - (age / max(0.001, ttl_s)))


def _pressure_snapshot(orch: Orchestrator) -> dict[str, Any]:
    """Build the two-speed pressure-band snapshot for the Textual face.

    This is UI instrumentation, not cognition. It samples already-published KV
    state so the face can show whether teaching/reward/novelty signals are
    actually moving without asking every organ to speak.
    """
    now = time.time()
    kv = getattr(orch, "kv_store", {}) or {}

    power_state = _as_dict(kv.get("power:state"))
    boredom = _as_dict(kv.get("drive:boredom"))
    social = _as_dict(kv.get("drive:social_interaction"))
    social_exp = _as_dict(kv.get("drive:social_experimentation"))
    thought_turn = _as_dict(kv.get("thought:turn:last_state"))
    thought_momentum = _as_dict(kv.get("thought:momentum"))
    capability = _as_dict(kv.get("capability:state"))
    maintenance = _as_dict(kv.get("memory:last_sleep_maintenance"))
    composer_status = _as_dict(kv.get("mem_cell:composer:last_status"))
    reinforce = _as_dict(kv.get("reinforce:last"))
    trainer = _as_dict(kv.get("trainer:last_correction"))
    reward_state = _as_dict(kv.get("affect:reward_state"))
    novelty_state = _as_dict(kv.get("affect:novelty_state"))
    salience_state = _as_dict(kv.get("affect:salience_state"))

    salience = max(
        _safe_float(kv.get("affect:global_salience"), 0.0),
        _safe_float(salience_state.get("level"), 0.0),
    )
    reinforce_raw = abs(_safe_float(reinforce.get("weight", reinforce.get("score", 0.0)), 0.0)) / 10.0
    reinforce_reward = max(0.0, min(1.0, reinforce_raw)) * _age_decay(now, reinforce.get("ts"), ttl_s=18.0)
    reward_level = max(
        reinforce_reward,
        _safe_float(reward_state.get("level", reward_state.get("dopamine", 0.0)), 0.0),
    )
    reward = max(0.0, min(1.0, reward_level))
    train = _age_decay(now, trainer.get("ts"), ttl_s=24.0)

    curiosity_boost = max(0.0, min(1.0, _safe_float(kv.get("curiosity:boost"), 0.0)))
    curiosity = max(
        curiosity_boost,
        _safe_float(novelty_state.get("level"), 0.0) * 0.35,
        _safe_float(social_exp.get("pressure"), 0.0) * 0.45,
        _safe_float(thought_momentum.get("pressure"), 0.0)
        if str(thought_momentum.get("dominant_intent", "")).lower() in {"curiosity", "seek_novelty", "social_experiment"}
        else 0.0,
    )

    body = {
        "power_mode": str(power_state.get("mode") or kv.get("power:mode") or "awake"),
        "charging": bool(power_state.get("charging", False)),
        "sleep": bool(power_state.get("sleep", kv.get("power:sleep", False))),
        "maintenance": str(maintenance.get("status") or maintenance.get("result") or "idle"),
        "memory_pending": int(_safe_float(kv.get("mem_cell:composer:pending_count"), 0.0)),
        "memory_composer": "on" if bool(kv.get("mem_cell:composer:started", False)) else "off",
        "read_sidecar": "on" if bool(kv.get("read:sidecar_started", False)) else "off",
        "cap_available": int(_safe_float(capability.get("available_count"), 0.0)),
        "cap_total": int(_safe_float(capability.get("component_count"), 0.0)),
        "drawer_waiting": int(_safe_float(thought_turn.get("waiting_count"), 0.0)),
        "drawer_ready": int(_safe_float(thought_turn.get("ready_count"), 0.0)),
    }

    pulse = {
        "salience": round(max(0.0, min(1.0, salience)), 3),
        "reward": round(max(0.0, min(1.0, reward)), 3),
        "boredom": round(max(0.0, min(1.0, _safe_float(boredom.get("level"), 0.0))), 3),
        "curiosity": round(max(0.0, min(1.0, curiosity)), 3),
        "expression": round(max(0.0, min(1.0, _safe_float(social.get("level"), 0.0))), 3),
        "trainer": round(max(0.0, min(1.0, train)), 3),
        "thought_pressure": round(max(0.0, min(1.0, _safe_float(thought_momentum.get("pressure"), 0.0))), 3),
        "thought_intent": str(
            thought_turn.get("dominant_family")
            or thought_momentum.get("dominant_intent")
            or "idle"
        ),
        "thought_status": str(thought_turn.get("dominant_status") or ("active" if thought_momentum.get("active") else "idle")),
        "novelty_delta": round(_safe_float(boredom.get("novelty_delta"), 0.0), 3),
    }

    return {"schema": "ui.pressure_band.v1", "ts": now, "body": body, "pulse": pulse}


async def run_textual_frontend(orch: Orchestrator, *, memdir: str | None = None) -> None:
    """Run Textual UI connected to an already-started orchestrator."""

    recv_q: asyncio.Queue[UIMessage] = asyncio.Queue(maxsize=500)

    async def _ui_tap(ev: Event) -> list[Event]:
        # Drop noisy internal ticks by default; UI would spam.
        if ev.topic == "clock/tick":
            return []
        meta = dict(ev.meta or {})
        # The Textual face should not show internal reasoning/request plumbing
        # unless a specific event opts into UI visibility. The log inspector can
        # still watch the firehose from microbrain.log.
        if meta.get("ui_hidden") is True or meta.get("ui_visible") is False:
            return []
        if ev.topic in {"reason/request", "reason/output"} and meta.get("ui_visible") is not True:
            return []
        try:
            recv_q.put_nowait(
                UIMessage(topic=ev.topic, payload=ev.payload, source=ev.source, meta=meta)
            )
        except asyncio.QueueFull:
            # Best-effort: if UI can't keep up, drop oldest by draining a little.
            try:
                _ = recv_q.get_nowait()
                recv_q.put_nowait(
                    UIMessage(topic=ev.topic, payload=ev.payload, source=ev.source, meta=dict(ev.meta or {}))
                )
            except Exception:
                pass
        return []

    # Subscribe to the stuff a human cares about.
    # - act/speech: assistant output
    # - vision/status: window grabber status messages
    # - control/vision: confirmations can be emitted elsewhere; still useful
    topics = [
        "act/speech",
        "ui/status",
        "ui/error",
        "control/status",
        "control/error",
        "reason/request",
        "reason/output",
        "vision/status",
        "vision/focus",
        "control/vision",
        "control/focus",
    ]
    # EventBus signature is: subscribe(name, topics, handler, priority=0)
    sub_id = orch.bus.subscribe(
        "ui.textual.tap",
        topics,
        _ui_tap,
        priority=0,
    )
    async def _send_text(text: str) -> None:
        await orch.push_event(
            "input/text",
            text,
            meta={"source": "ui", "channel": "textual"},
        )
        # Let neurons chew; if something is stuck, UI should remain responsive anyway.
        await orch.wait_for_idle(timeout=30.0)

    async def _pressure_pump() -> None:
        while True:
            try:
                recv_q.put_nowait(
                    UIMessage(
                        topic="ui/pressure_state",
                        payload=_pressure_snapshot(orch),
                        source="ui.pressure_sampler",
                        meta={"ui_hidden": True, "store_in_memory": False},
                    )
                )
            except asyncio.QueueFull:
                try:
                    _ = recv_q.get_nowait()
                except Exception:
                    pass
            except Exception:
                pass
            await asyncio.sleep(0.25)

    pressure_task = asyncio.create_task(_pressure_pump(), name="ui_pressure_band_sampler")

    app = MicroBrainUI(send_cb=_send_text, recv_q=recv_q, memdir=memdir)
    try:
        await app.run_async()
    finally:
        pressure_task.cancel()
        try:
            await pressure_task
        except asyncio.CancelledError:
            pass

    # When UI closes, best-effort unsubscribe.
    try:
        orch.bus.unsubscribe(sub_id)
    except Exception:
        pass
