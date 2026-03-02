from __future__ import annotations

import argparse
import json
from pathlib import Path

from microbrain.utils.unit_config import load_or_create_unit_config, save_unit_config, set_path


def parse_value(raw: str):
    r = raw.strip()
    if r.lower() in ("true", "false"):
        return r.lower() == "true"
    try:
        if "." in r:
            return float(r)
        return int(r)
    except Exception:
        pass
    # JSON object/array?
    if (r.startswith("{") and r.endswith("}")) or (r.startswith("[") and r.endswith("]")):
        try:
            return json.loads(r)
        except Exception:
            return r
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit a MicroBrain unit_config.json in-place.")
    ap.add_argument("--memdir", required=True, help="Path to unit memdir (e.g. Z:\\memory)")
    ap.add_argument("--set", action="append", default=[], help="Set dotted path key=value (repeatable). Example: tts_out.enabled=true")
    ap.add_argument("--print", action="store_true", help="Print config and exit.")
    args = ap.parse_args()

    memdir = Path(args.memdir)
    cfg = load_or_create_unit_config(memdir)

    if args.print and not args.set:
        print(json.dumps(cfg, indent=2))
        return 0

    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"Bad --set {kv!r}; expected key=value")
        k, v = kv.split("=", 1)
        set_path(cfg, k.strip(), parse_value(v))

    save_unit_config(memdir, cfg)
    print(f"[OK] updated: {memdir / 'unit_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
