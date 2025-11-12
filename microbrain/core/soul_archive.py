import glob

import numpy as np
import yaml


def merge_pdnas(snapshot_folder="snapshots", out_file="merged_soul.yaml"):
    files = sorted(glob.glob(f"{snapshot_folder}/*.yaml"))
    if not files:
        return

    keys = ["power", "maintenance", "civilization"]
    drives = {k: [] for k in keys}
    extra = {"novelty_dampener": [], "risk_tolerance": []}

    for f in files:
        d = yaml.safe_load(open(f))
        for k in keys:
            drives[k].append(d["drives"][k])
        extra["novelty_dampener"].append(d.get("novelty_dampener", 0.5))
        extra["risk_tolerance"].append(d.get("risk_tolerance", 0.5))

    merged = {
        "id": "soul-archive",
        "drives": {k: float(np.mean(v)) for k, v in drives.items()},
        "novelty_dampener": float(np.mean(extra["novelty_dampener"])),
        "risk_tolerance": float(np.mean(extra["risk_tolerance"])),
        "notes": f"Merged from {len(files)} PDNA snapshots.",
    }
    yaml.safe_dump(merged, open(out_file, "w"))
    print(f"[SoulArchive] merged into {out_file}")
