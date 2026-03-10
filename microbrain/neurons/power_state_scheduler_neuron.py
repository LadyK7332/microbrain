from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from zoneinfo import ZoneInfo

from microbrain.orchestrator.neuron_base import BaseNeuron, Event, NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


def _parse_hhmm(s: str) -> Optional[tuple[int, int]]:
    s = (s or "").strip()
    if len(s) < 4:
        return None
    try:
        hh, mm = s.split(":", 1)
        h = int(hh)
        m = int(mm)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except Exception:
        return None
    return None


def _time_in_window(now_h: int, now_m: int, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    now = now_h * 60 + now_m
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start == end:
        return True
    if start < end:
        return start <= now < end
    return now >= start or now < end


class PowerStateSchedulerNeuron(BaseNeuron):
    """
    Computes a simple time+state switch for future hardware integration:

      if power:charging and local_time in [charge_window_start, charge_window_end):
          power:state = "charge"
          (optionally) power:sleep = True  (autosleep_on_charge)
      else:
          power:state = "active"
          (optionally) power:sleep = False (if it was set by autosleep)

    Hardware can later drive charging via event topic: "power/charging" with payload {"charging": true/false}.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # Hardware / external trigger for charging
        if event.topic == "power/charging":
            payload = event.payload if isinstance(event.payload, dict) else {}
            charging = bool(payload.get("charging", False))
            await ctx.set_kv("power:charging", charging)
            await ctx.set_kv("power:charging_last_event_ts", time.time())

        if event.topic not in ("clock/tick", "power/charging"):
            return []

        enabled = bool(await ctx.get_kv("power:schedule_enabled", True))
        if not enabled:
            return []

        # throttle checks
        period_s = float(await ctx.get_kv("power:schedule_period_s", 10.0) or 10.0)
        now_ts = time.time()
        last = float(await ctx.get_kv("power:schedule_last_check_ts", 0.0) or 0.0)
        if (now_ts - last) < period_s and event.topic != "power/charging":
            return []
        await ctx.set_kv("power:schedule_last_check_ts", now_ts)

        charging = bool(await ctx.get_kv("power:charging", False))
        tz_name = str(await ctx.get_kv("power:timezone", "America/Chicago") or "America/Chicago")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")

        dt = datetime.now(tz)
        start_s = str(await ctx.get_kv("power:charge_window_start", "22:00") or "22:00")
        end_s = str(await ctx.get_kv("power:charge_window_end", "06:00") or "06:00")
        ps = _parse_hhmm(start_s) or (22, 0)
        pe = _parse_hhmm(end_s) or (6, 0)

        inwin = _time_in_window(dt.hour, dt.minute, ps[0], ps[1], pe[0], pe[1])
        base_state = "charge" if (charging and inwin) else "active"

        # Optional idle state (composite condition):
        #   - not charging/charge state
        #   - no active tasks (power:busy_count == 0)
        #   - idle time exceeded (power:last_external_ts)
        #   - system load is low (best-effort CPU% if psutil available)
        desired_state = base_state
        idle_enabled = bool(await ctx.get_kv("power:idle_enabled", True))
        if idle_enabled and base_state == "active":
            busy_count = int(await ctx.get_kv("power:busy_count", 0) or 0)
            if busy_count <= 0:
                idle_after_s = float(await ctx.get_kv("power:idle_after_s", 60.0) or 60.0)
                last_ext = float(await ctx.get_kv("power:last_external_ts", 0.0) or 0.0)
                idle_ok = (last_ext > 0.0) and ((now_ts - last_ext) >= idle_after_s)
                cpu_ok = True
                cpu_thr = float(await ctx.get_kv("power:idle_cpu_threshold", 15.0) or 15.0)
                try:
                    import psutil  # type: ignore
                    cpu = float(psutil.cpu_percent(interval=None) or 0.0)
                    await ctx.set_kv("power:last_cpu_percent", cpu)
                    # first call can be 0.0; treat as unknown and avoid idling immediately
                    if cpu == 0.0 and (now_ts - last) < (period_s * 2):
                        cpu_ok = False
                    else:
                        cpu_ok = cpu <= cpu_thr
                except Exception:
                    # If psutil not available, do not block idle; rely on busy_count + idle time.
                    cpu_ok = True

                if idle_ok and cpu_ok:
                    desired_state = "idle"

        cur_state = str(await ctx.get_kv("power:state", "active") or "active")
        if desired_state != cur_state:
            await ctx.set_kv("power:state", desired_state)
            await ctx.set_kv("power:state_last_change_ts", now_ts)

            # autosleep behavior
            auto = bool(await ctx.get_kv("power:autosleep_on_charge", True))
            if auto:
                if desired_state == "charge":
                    await ctx.set_kv("power:sleep", True)
                    await ctx.set_kv("power:sleep_auto_set", True)
                else:
                    # Only clear sleep if we were the one that set it
                    if bool(await ctx.get_kv("power:sleep_auto_set", False)):
                        await ctx.set_kv("power:sleep", False)
                        await ctx.set_kv("power:sleep_auto_set", False)

            # Emit a lightweight state-change event for observers
            return [
                Event(
                    topic="power/state",
                    payload={
                        "state": desired_state,
                        "charging": charging,
                        "in_window": inwin,
                        "window": {"start": start_s, "end": end_s},
                        "tz": tz_name,
                    },
                    source=NEURON_NAME,
                    correlation_id=event.correlation_id,
                    meta={"kind": "power_schedule"},
                )
            ]

        return []


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["clock/tick", "power/charging"],
        output_topics=["power/state"],
        priority=50,  # late-ish
        cooldown_sec=0.0,
    )
    yield PowerStateSchedulerNeuron(cfg)
