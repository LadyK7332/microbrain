from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.utils.memdir import resolve_memdir_ctx

from microbrain.utils.heartbeat_stream import service_topic

NEURON_NAME = Path(__file__).stem
SERVICE_TOPIC = service_topic("memory")


def _sha_id(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


class EngramConsolidatorNeuron(BaseNeuron):
    """
    Tail memdir/synapses.jsonl and promote stable 'pattern_edge' rows
    into memdir/concepts/longterm.jsonl.

    This is "graduation-only":
      - doesn't write every time
      - writes once per promoted edge
      - keeps a file offset in KV so it doesn't rescan forever
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        if event.topic != SERVICE_TOPIC:
            return []

        now = time.time()
        sleep_active = bool(await ctx.get_kv("power:sleep", False))
        charging_active = bool(await ctx.get_kv("power:charging", False))
        raw_power_state = await ctx.get_kv("power:state", "")
        if isinstance(raw_power_state, dict):
            power_mode = str(raw_power_state.get("mode", "") or "").lower()
        else:
            power_mode = str(raw_power_state or "").lower()

        # Maintenance-only: only run when the organism is actually parked.
        if not sleep_active or not charging_active:
            return []
        if power_mode and power_mode not in ("charge", "charging", "sleep_charge"):
            return []

        sleep_set_ts = _safe_float(await ctx.get_kv("power:sleep_last_set_ts", 0.0), 0.0)
        charge_set_ts = _safe_float(
            await ctx.get_kv("power:charging_last_set_ts", await ctx.get_kv("power:charging_last_event_ts", 0.0)),
            0.0,
        )
        settle_s = _safe_float(await ctx.get_kv("engram:settle_after_sleep_charge_s", 300.0), 300.0)
        since_sleep = (now - sleep_set_ts) if sleep_set_ts > 0 else 1e9
        since_charge = (now - charge_set_ts) if charge_set_ts > 0 else 1e9
        if since_sleep < settle_s or since_charge < settle_s:
            return []

        min_interval_s = _safe_float(await ctx.get_kv("engram:min_interval_s", 900.0), 900.0)
        last_completed_ts = _safe_float(await self.load_state(ctx, "last_completed_ts", 0.0), 0.0)
        if last_completed_ts > 0 and (now - last_completed_ts) < min_interval_s:
            return []

        memdir = await resolve_memdir_ctx(ctx, fallback=r"Z:\memory")
        syn_path = memdir / "synapses.jsonl"
        concepts_dir = memdir / "concepts"
        longterm_path = concepts_dir / "longterm.jsonl"

        concepts_dir.mkdir(parents=True, exist_ok=True)
        if not longterm_path.exists():
            longterm_path.write_text("", encoding="utf-8")

        if not syn_path.exists():
            return []

        offset = _safe_int(await self.load_state(ctx, "syn_offset", 0), 0)
        min_pending_bytes = _safe_int(await ctx.get_kv("engram:min_pending_bytes", 256), 256)

        try:
            current_size = _safe_int(syn_path.stat().st_size, 0)
        except Exception:
            current_size = 0

        pending_bytes = max(0, current_size - offset)
        if pending_bytes < min_pending_bytes:
            return []

        await ctx.log_info(
            f"[{self.name}] consolidation started",
            pending_bytes=pending_bytes,
            sleep_active=sleep_active,
            charging_active=charging_active,
        )
        run_started = time.time()

        promote_w = _safe_float(await ctx.get_kv("engram:edge_promote_w", 0.30), 0.30)
        max_lines = _safe_int(await ctx.get_kv("engram:max_lines", 500), 500)

        edge_types = await ctx.get_kv("engram:edge_types", None)
        if not isinstance(edge_types, list):
            # Include BOTH directions by default
            edge_types = ["token_concept", "concept_token"]

        announce = bool(await ctx.get_kv("engram:announce", False))

        promoted: Dict[str, Any] = await self.load_state(ctx, "promoted", default={}) or {}
        if not isinstance(promoted, dict):
            promoted = {}

        new_offset = offset
        now = time.time()
        outputs: List[Event] = []

        try:
            with syn_path.open("rb") as f:
                # If file got rotated smaller, reset offset safely
                try:
                    size = syn_path.stat().st_size
                    if offset > size:
                        offset = 0
                except Exception:
                    pass

                f.seek(max(0, offset))

                for _ in range(max_lines):
                    line = f.readline()
                    if not line:
                        break

                    new_offset = f.tell()

                    s = line.decode("utf-8", errors="ignore").strip()
                    if not s:
                        continue

                    try:
                        row = json.loads(s)
                    except Exception:
                        continue

                    if not isinstance(row, dict) or row.get("kind") != "pattern_edge":
                        continue

                    et = str(row.get("edge_type") or "")
                    if et not in edge_types:
                        continue

                    src = str(row.get("src") or "")
                    dst = str(row.get("dst") or "")
                    w = _safe_float(row.get("w", 0.0), 0.0)
                    ts = _safe_float(row.get("ts", now), now)

                    if not src or not dst:
                        continue
                    if w < promote_w:
                        continue

                    engram_id = _sha_id("edge", et, src, dst)
                    if engram_id in promoted:
                        continue

                    record = {
                        "schema": "engram.edge.v1",
                        "engram_id": engram_id,
                        "edge_type": et,
                        "src": src,
                        "dst": dst,
                        "w": w,
                        "created_ts": ts,
                        "promoted_ts": now,
                        "source": NEURON_NAME,
                    }

                    try:
                        with longterm_path.open("a", encoding="utf-8") as out:
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception:
                        continue

                    promoted[engram_id] = {"ts": ts, "w": w, "edge_type": et, "src": src, "dst": dst}

                    if announce:
                        outputs.append(
                            Event(
                                topic="act/speech",
                                payload=f"[engram] promoted {et}: {src} -> {dst} (w={w:.2f})",
                                source=NEURON_NAME,
                                correlation_id=event.correlation_id,
                                meta={"kind": "engram_promoted"},
                            )
                        )

        finally:
            if new_offset != offset:
                await self.save_state(ctx, "syn_offset", new_offset)
            await self.save_state(ctx, "promoted", promoted)

            completed_ts = time.time()
            runtime_s = round(completed_ts - run_started, 4)
            await self.save_state(ctx, "last_completed_ts", completed_ts)
            await self.save_state(
                ctx,
                "last_run_completed",
                {
                    "ts": completed_ts,
                    "runtime_s": runtime_s,
                    "processed_bytes": max(0, int(new_offset) - int(offset)),
                    "promoted_count": len(outputs) if announce else 0,
                },
            )
            await ctx.log_info(
                f"[{self.name}] consolidation completed",
                runtime_s=runtime_s,
                processed_bytes=max(0, int(new_offset) - int(offset)),
                promoted_count=len(outputs) if announce else 0,
            )

        return outputs


def build_neurons(orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[SERVICE_TOPIC],
        output_topics=["act/speech"],
        priority=7,
        cooldown_sec=0.0,
    )
    yield EngramConsolidatorNeuron(cfg)
