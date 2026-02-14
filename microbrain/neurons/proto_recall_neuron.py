import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

from microbrain.patterns.pattern_edge_log import PatternEdgeLog
from microbrain.patterns.proto_concept_store import ProtoConceptStore

NEURON_NAME = Path(__file__).stem


class ProtoRecallNeuron(BaseNeuron):
    """
    Listens to percept/vision.
    If a vector is present, assigns/attaches a proto:vision:* id (online clustering).
    Then seeds recall from proto -> concepts using pattern edges (edge_type="proto_concept").

    Outputs:
      - ctx KV: recall:last_bundle
      - memdir/state/recall_last.json
      - event: memory/recall_context
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._mem_store = None
        self._memdir: Optional[Path] = None
        self._edges: Optional[PatternEdgeLog] = None
        self._proto: Optional[ProtoConceptStore] = None

    async def _ensure_ready(self, ctx) -> bool:
        if self._edges is not None and self._proto is not None and self._mem_store is not None and self._memdir is not None:
            return True

        mem_store = await ctx.get_kv("memory:store", None)
        if mem_store is None:
            return False

        memdir = Path(str(getattr(mem_store, "base_dir", "") or ""))
        if not str(memdir):
            return False

        if self._memdir != memdir:
            self._memdir = memdir
            self._mem_store = mem_store

            # Pattern edges (shared with your token/concept system)
            self._edges = await ctx.get_kv("patterns:edges", None)
            if self._edges is None:
                self._edges = PatternEdgeLog(memdir, filename="synapses.jsonl")
                await ctx.set_kv("patterns:edges", self._edges)

            # Proto concept store (vision)
            self._proto = await ctx.get_kv("patterns:proto:vision", None)
            if self._proto is None:
                self._proto = ProtoConceptStore(memdir, modality="vision")
                await ctx.set_kv("patterns:proto:vision", self._proto)

        return True

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        self.debug("received", topic=event.topic, source=event.source, payload=event.payload)

        if event.topic != "percept/vision":
            return []

        if not await self._ensure_ready(ctx):
            return []

        payload = event.payload
        if not isinstance(payload, dict):
            return []

        channel = str(payload.get("channel", "default") or "default")
        ts = float(payload.get("ts", 0.0) or event.timestamp or time.time())

        raw_meta = payload.get("raw_meta", {})
        if not isinstance(raw_meta, dict):
            raw_meta = {}
        noun_id = str(raw_meta.get("noun_id", "") or "").strip() or None
        if noun_id and not noun_id.startswith("noun:"):
            noun_id = f"noun:{noun_id}"

        assets = payload.get("assets", None)
        focus_asset_id = str(payload.get("focus_asset_id", "") or "").strip() or None
        payload_note: Optional[str] = None

        # If we didn't get the future "assets[]" shape, try the current frame-shaped payload:
        # { frame_id, data_ref, width, height, format, ... }
        if not isinstance(assets, list) or not assets:
            crop_px = int(await ctx.get_kv("patterns:proto_crop_px_vision", 256) or 256)
            grid = int(await ctx.get_kv("patterns:proto_grid_vision", 16) or 16)
            assets, focus_asset_id, payload_note = self._assets_from_frame_payload(
                payload=payload,
                raw_meta=raw_meta,
                channel=channel,
                ts=ts,
                crop_px=crop_px,
                grid=grid,
            )

        if not assets:
            bundle = {
                "seed_kind": "proto",
                "seed_id": None,
                "top_protos": [],
                "top_concepts": [],
                "channel": channel,
                "noun_id": noun_id,
                "ts": ts,
                "schema_ver": 2,
                "kind": "pattern_recall_bundle",
                "note": payload_note or "no vision assets available (expected assets[] or data_ref frame payload)",
            }
            await ctx.set_kv("recall:last_bundle", bundle)
            self._write_state(bundle)
            return [Event(topic="memory/recall_context", payload=bundle, source=self.name)]

        # ---- assign proto ids ----
        thresh = float(await ctx.get_kv("patterns:proto_thresh_vision", 0.86) or 0.86)
        alpha = float(await ctx.get_kv("patterns:proto_alpha_vision", 0.10) or 0.10)

        assigned: list[dict[str, Any]] = []
        for a in assets:
            if not isinstance(a, dict):
                continue
            asset_id = str(a.get("asset_id", "") or "").strip()
            if not asset_id:
                continue

            proto_id = str(a.get("proto_id", "") or "").strip() or None
            sim = None

            vec = a.get("vec", None)
            if proto_id is None and isinstance(vec, list) and vec:
                try:
                    proto_id, sim = self._proto.assign(
                        vec=vec,
                        asset_id=asset_id,
                        channel=channel,
                        ts=ts,
                        thresh_attach=thresh,
                        alpha_ema=alpha,
                    )
                except Exception:
                    proto_id = None

            assigned.append(
                {
                    "asset_id": asset_id,
                    "proto_id": proto_id,
                    "sim": sim,
                }
            )

        # Pick seed proto
        seed_proto_id: Optional[str] = None
        if focus_asset_id:
            for row in assigned:
                if row.get("asset_id") == focus_asset_id and row.get("proto_id"):
                    seed_proto_id = str(row["proto_id"])
                    break
        if seed_proto_id is None:
            for row in assigned:
                if row.get("proto_id"):
                    seed_proto_id = str(row["proto_id"])
                    break

        # If no proto id is available yet, we can’t do proto-driven recall.
        if seed_proto_id is None:
            bundle = {
                "seed_kind": "proto",
                "seed_id": None,
                "top_protos": [],
                "top_concepts": [],
                "channel": channel,
                "noun_id": noun_id,
                "ts": ts,
                "schema_ver": 1,
                "kind": "pattern_recall_bundle",
                "note": payload_note or "no proto_id present (provide asset.vec / asset.proto_id, or data_ref with PIL available)",
            }
            await ctx.set_kv("recall:last_bundle", bundle)
            self._write_state(bundle)
            return [Event(topic="memory/recall_context", payload=bundle, source=self.name)]

        # ---- spread activation: proto -> concept (1 hop v0) ----
        concept_scores: Dict[str, float] = {}
        W = getattr(self._edges, "_W", {}) or {}

        for k, w in W.items():
            try:
                if getattr(k, "edge_type", "") != "proto_concept":
                    continue
                if getattr(k, "src", "") != seed_proto_id:
                    continue
                cid = str(getattr(k, "dst", "") or "")
                if not cid:
                    continue
                concept_scores[cid] = concept_scores.get(cid, 0.0) + float(w)
            except Exception:
                continue

        # Rank
        ranked = sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_concepts = [{"concept_id": cid, "label": cid.split(":", 1)[1] if ":" in cid else cid, "score": round(float(s), 6)}
                        for cid, s in ranked]

        bundle = {
            "seed_kind": "proto",
            "seed_id": seed_proto_id,
            "top_protos": [{"proto_id": seed_proto_id, "score": 1.0}],
            "top_concepts": top_concepts,
            "channel": channel,
            "noun_id": noun_id,
            "ts": ts,
            "schema_ver": 2,
            "kind": "pattern_recall_bundle",
        }

        await ctx.set_kv("recall:last_bundle", bundle)
        self._write_state(bundle)

        return [Event(topic="memory/recall_context", payload=bundle, source=self.name)]

    def _assets_from_frame_payload(
        self,
        payload: dict,
        raw_meta: dict,
        channel: str,
        ts: float,
        crop_px: int,
        grid: int,
    ) -> tuple[list[dict[str, Any]], Optional[str], Optional[str]]:
        """
        Convert current frame-shaped percept/vision payload into an assets[] list.
        Expected payload keys (current):
          - frame_id (optional)
          - data_ref (path to jpg/png)
          - width/height/format (optional)
        Returns: (assets, focus_asset_id, note)
        """
        data_ref = str(payload.get("data_ref", "") or "").strip()
        frame_id = str(payload.get("frame_id", "") or "").strip()
        if not data_ref:
            return [], None, "vision payload missing data_ref"

        p = self._resolve_data_ref(data_ref)
        if p is None:
            return [], None, f"data_ref not found on disk: {data_ref}"

        asset_id = frame_id or p.stem or "vision_frame"

        # If the upstream already provided a vec, use it
        vec = payload.get("vec", None)
        if not (isinstance(vec, list) and vec):
            # Try to build a tiny “proto embedding” from the image (downsample grayscale).
            focus_xy = self._extract_focus_xy(payload, raw_meta, width=payload.get("width"), height=payload.get("height"))
            vec, note = self._image_to_vec(p, focus_xy=focus_xy, crop_px=crop_px, grid=grid)
            if vec is None:
                return [{"asset_id": asset_id}], asset_id, note

        assets = [{"asset_id": asset_id, "vec": vec}]
        return assets, asset_id, None

    def _resolve_data_ref(self, data_ref: str) -> Optional[Path]:
        """
        Resolve data_ref to a real file path. Tries:
          - direct path
          - memdir/data_ref
          - memdir/vision/data_ref
        """
        try:
            p = Path(data_ref)
            if p.exists():
                return p
        except Exception:
            pass

        if self._memdir:
            try:
                p2 = Path(self._memdir) / data_ref
                if p2.exists():
                    return p2
                p3 = Path(self._memdir) / "vision" / data_ref
                if p3.exists():
                    return p3
            except Exception:
                pass

        return None

    def _extract_focus_xy(
        self,
        payload: dict,
        raw_meta: dict,
        width: Any = None,
        height: Any = None,
    ) -> Optional[tuple[float, float]]:
        """
        Tries to find a focus point for cropping around the reticle/cursor.
        Accepts either pixel coords or normalized 0..1 coords.
        """
        for key in ("focus_xy", "cursor_xy", "reticle_xy"):
            v = payload.get(key, None)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float(v[0]), float(v[1])
            v = raw_meta.get(key, None) if isinstance(raw_meta, dict) else None
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float(v[0]), float(v[1])

        # separate x/y keys
        for xk, yk in (("focus_x", "focus_y"), ("cursor_x", "cursor_y"), ("reticle_x", "reticle_y")):
            x = payload.get(xk, None)
            y = payload.get(yk, None)
            if x is not None and y is not None:
                try:
                    return float(x), float(y)
                except Exception:
                    pass

        return None

    def _image_to_vec(
        self,
        path: Path,
        focus_xy: Optional[tuple[float, float]],
        crop_px: int,
        grid: int,
    ) -> tuple[Optional[list[float]], Optional[str]]:
        """
        Very cheap proto-embedding:
          - open image
          - crop around focus
          - resize to (grid x grid)
          - grayscale
          - flatten + mean-center + L2 normalize
        """
        try:
            from PIL import Image  # optional dependency
        except Exception:
            return None, "PIL not installed (pip install pillow) so can't build vec from data_ref yet"

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            return None, f"failed to open image for proto vec: {e}"

        w, h = img.size
        cx, cy = (w / 2.0, h / 2.0)

        if focus_xy is not None:
            fx, fy = focus_xy
            # interpret normalized focus if within 0..1
            if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
                cx = fx * w
                cy = fy * h
            else:
                cx = fx
                cy = fy

        half = max(8, int(crop_px) // 2)
        left = int(max(0, cx - half))
        top = int(max(0, cy - half))
        right = int(min(w, cx + half))
        bottom = int(min(h, cy + half))

        if right <= left or bottom <= top:
            left, top, right, bottom = 0, 0, w, h

        crop = img.crop((left, top, right, bottom)).resize((int(grid), int(grid))).convert("L")
        px = list(crop.getdata())  # 0..255

        v = [p / 255.0 for p in px]
        if not v:
            return None, "empty image vec"

        mean = sum(v) / len(v)
        v = [x - mean for x in v]

        # L2 normalize
        n = math.sqrt(sum(x * x for x in v))
        if n > 0.0:
            v = [x / n for x in v]

        return v, None

    def _write_state(self, bundle: dict) -> None:
        try:
            if not self._memdir:
                return
            state_dir = Path(self._memdir) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "recall_last.json").write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/vision"],
        output_topics=["memory/recall_context"],
        priority=6,
    )
    yield ProtoRecallNeuron(cfg)
