from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def hsv_hist_embedding(bgr: np.ndarray, bins: int = 24) -> np.ndarray:
    """Cheap, stable embedding: normalized HSV histogram."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    v = hist.astype(np.float32).reshape(-1)
    v /= (np.linalg.norm(v) + 1e-9)
    return v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class Exemplar:
    emb: np.ndarray
    path: Path


@dataclass
class Cluster:
    id: int
    rep: np.ndarray
    exemplars: List[Exemplar] = field(default_factory=list)
    count: int = 0


@dataclass
class PendingNewCluster:
    emb: np.ndarray
    hits: int = 0
    first_ts: float = 0.0


def nearest_cluster(emb: np.ndarray, clusters: List[Cluster]) -> Tuple[Optional[Cluster], float]:
    if not clusters:
        return None, 1e9
    best = None
    best_d = 1e9
    for c in clusters:
        d = cosine_distance(emb, c.rep)
        if d < best_d:
            best_d = d
            best = c
    return best, best_d


def nearest_exemplar(emb: np.ndarray, c: Cluster) -> float:
    if not c.exemplars:
        return 1e9
    best = 1e9
    for ex in c.exemplars:
        d = cosine_distance(emb, ex.emb)
        if d < best:
            best = d
    return best


def save_exemplar(out_dir: Path, cluster_id: int, img_bgr: np.ndarray, emb: np.ndarray) -> Path:
    cdir = out_dir / f"cluster-{cluster_id:03d}"
    cdir.mkdir(parents=True, exist_ok=True)

    # count existing exemplars
    existing = sorted(cdir.glob("ex-*.jpg"))
    idx = len(existing) + 1
    path = cdir / f"ex-{idx:04d}.jpg"

    cv2.imwrite(str(path), img_bgr)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=r"Z:\memory\sight\frames")
    ap.add_argument("--pattern", default="frame-*.jpg")
    ap.add_argument("--out-dir", default=r"Z:\memory\sight\exemplars")
    ap.add_argument("--poll-ms", type=int, default=200)

    # Two-threshold idea:
    # - duplicate_thresh: "same exemplar" (DON'T write)
    # - category_thresh: "same category cluster" (DON'T make a new cluster)
    ap.add_argument("--duplicate-thresh", type=float, default=0.10)
    ap.add_argument("--category-thresh", type=float, default=0.25)

    # New cluster confirmation (avoid transient noise)
    ap.add_argument("--confirm-hits", type=int, default=4)

    # Keep rep tracking lighting drift a little
    ap.add_argument("--rep-ema", type=float, default=0.05)

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_glob = str(in_dir / args.pattern)

    clusters: List[Cluster] = []
    pending: Optional[PendingNewCluster] = None
    last_seen: Optional[str] = None

    print(f"[watch] {files_glob}")
    print(f"[out]   {out_dir}")
    print(f"[thresh] duplicate={args.duplicate_thresh:.3f} category={args.category_thresh:.3f} confirm={args.confirm_hits}")

    while True:
        files = sorted(glob.glob(files_glob))
        if not files:
            time.sleep(args.poll_ms / 1000.0)
            continue

        newest = files[-1]
        if newest == last_seen:
            time.sleep(args.poll_ms / 1000.0)
            continue

        last_seen = newest
        img = cv2.imread(newest)
        if img is None:
            continue

        emb = hsv_hist_embedding(img)

        c, d_c = nearest_cluster(emb, clusters)

        # No cluster yet -> begin pending and confirm
        if c is None:
            if pending is None:
                pending = PendingNewCluster(emb=emb.copy(), hits=1, first_ts=time.time())
                print(f"[pending] new cluster? hits=1 d=inf file={Path(newest).name}")
            else:
                pending.hits += 1
                if pending.hits >= args.confirm_hits:
                    new_id = 0
                    new_cluster = Cluster(id=new_id, rep=pending.emb.copy())
                    path = save_exemplar(out_dir, new_id, img, emb)
                    new_cluster.exemplars.append(Exemplar(emb=emb.copy(), path=path))
                    new_cluster.count = 1
                    clusters.append(new_cluster)
                    pending = None
                    print(f"[SAVE] cluster={new_id} ex=1 file={path.name}")
            continue

        # If far from best cluster => pending new category
        if d_c > args.category_thresh:
            if pending is None:
                pending = PendingNewCluster(emb=emb.copy(), hits=1, first_ts=time.time())
                print(f"[pending] new cluster? hits=1 d_cluster={d_c:.3f} file={Path(newest).name}")
            else:
                pending.hits += 1
                if pending.hits >= args.confirm_hits:
                    new_id = max(x.id for x in clusters) + 1
                    new_cluster = Cluster(id=new_id, rep=pending.emb.copy())
                    path = save_exemplar(out_dir, new_id, img, emb)
                    new_cluster.exemplars.append(Exemplar(emb=emb.copy(), path=path))
                    new_cluster.count = 1
                    clusters.append(new_cluster)
                    pending = None
                    print(f"[SAVE] NEW cluster={new_id} ex=1 d_cluster={d_c:.3f} file={path.name}")
            continue

        # We are in a known category; clear pending
        pending = None

        # Check if it's the same exemplar (duplicate) or a new exemplar worth saving
        d_ex = nearest_exemplar(emb, c)
        if d_ex < args.duplicate_thresh:
            # same thing again -> do not write
            c.count += 1
            # gently track drift
            alpha = float(args.rep_ema)
            c.rep = (1 - alpha) * c.rep + alpha * emb
            c.rep /= (np.linalg.norm(c.rep) + 1e-9)
            print(f"[same] cluster={c.id} d_ex={d_ex:.3f} d_c={d_c:.3f}")
            continue

        # New exemplar inside the same category -> write once
        path = save_exemplar(out_dir, c.id, img, emb)
        c.exemplars.append(Exemplar(emb=emb.copy(), path=path))
        c.count += 1
        alpha = float(args.rep_ema)
        c.rep = (1 - alpha) * c.rep + alpha * emb
        c.rep /= (np.linalg.norm(c.rep) + 1e-9)

        print(f"[SAVE] cluster={c.id} ex={len(c.exemplars)} d_ex={d_ex:.3f} d_c={d_c:.3f} file={path.name}")

        time.sleep(args.poll_ms / 1000.0)


if __name__ == "__main__":
    main()