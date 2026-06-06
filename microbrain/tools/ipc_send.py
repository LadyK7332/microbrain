# microbrain/tools/ipc_send.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microbrain.ipc.file_inbox import IPCFileWriter, DrawerDoneAnnouncer


def main() -> int:
    ap = argparse.ArgumentParser(description="Send a file-based IPC message into memdir/ipc/inbox.")
    ap.add_argument("--memdir", required=True, help="Memory directory (e.g. Z:\\memory)")
    ap.add_argument("--src", default="manual-tool", help="Sender name")
    ap.add_argument("--topic", default="", help="Topic (e.g. percept/vision, drawer/done)")
    ap.add_argument("--payload", default="{}", help="JSON payload string")
    ap.add_argument("--dedupe-key", default="", help="Optional dedupe key (prevents spam)")
    ap.add_argument("--done", action="store_true", help="Shortcut: send drawer/done with --drawer and --data-ref")
    ap.add_argument("--drawer", default="", help="Drawer path for --done (e.g. sight/exemplars)")
    ap.add_argument("--data-ref", default="", help="Data ref for --done (e.g. frame-000123.jpg)")
    args = ap.parse_args()

    memdir = Path(args.memdir)
    writer = IPCFileWriter(memdir=memdir, src=args.src)

    if args.done:
        # Allow --done to work without requiring --topic.
        topic = args.topic or "drawer/done"
        if not args.drawer or not args.data_ref:
            raise SystemExit("--done requires --drawer and --data-ref")
        ann = DrawerDoneAnnouncer(writer)
        out = writer.publish(topic, {"drawer": args.drawer, "data_ref": args.data_ref})
    else:
        if not args.topic:
            raise SystemExit("--topic is required unless --done is used")
        payload = json.loads(args.payload)
        dedupe_key = args.dedupe_key if args.dedupe_key else None
        out = writer.publish(args.topic, payload, dedupe_key=dedupe_key)

    if out is None:
        print("[ipc_send] deduped/skipped")
    else:
        print(f"[ipc_send] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())