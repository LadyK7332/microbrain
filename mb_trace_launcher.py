# mb_trace_launcher.py
from __future__ import annotations

import faulthandler
import os
import runpy
import sys
from pathlib import Path

def main() -> None:
    # Write periodic stack dumps to a file so you still get them if the console freezes.
    out_path = Path(os.environ.get("MB_TRACE_OUT", r"Z:\memory\state\faulthandler_traces.log"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f = out_path.open("a", encoding="utf-8")
    faulthandler.enable(file=f, all_threads=True)

    # Dump all thread traces every N seconds (default 10), repeatedly.
    interval = float(os.environ.get("MB_TRACE_EVERY", "10"))
    faulthandler.dump_traceback_later(interval, repeat=True, file=f)

    # Preserve CLI args for microbrain.mind
    sys.argv = ["microbrain.mind", *sys.argv[1:]]
    runpy.run_module("microbrain.mind", run_name="__main__")

if __name__ == "__main__":
    main()
