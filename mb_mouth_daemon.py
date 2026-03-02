# mb_mouth_daemon.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyttsx3

from microbrain.ipc.file_inbox import IPCFileWriter


def _pick_voice(engine: pyttsx3.Engine, preferred: str | None) -> None:
    if not preferred:
        return
    pref = preferred.lower().strip()
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        return
    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        if pref in name:
            try:
                engine.setProperty("voice", v.id)
            except Exception:
                pass
            return


def _read_token(memdir: Path) -> str:
    p = memdir / "ipc_token.txt"
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="MicroBrain Mouth sidecar (file IPC outbox -> TTS -> inbox ack).")
    ap.add_argument("--memdir", required=True, help="Memory directory (e.g. Z:\\memory)")
    ap.add_argument("--poll", type=float, default=0.15, help="Polling interval (seconds)")
    ap.add_argument("--default-voice", default=None, help="Voice substring to match (e.g. Zira, Aria)")
    ap.add_argument("--default-rate", type=int, default=155)
    ap.add_argument("--default-volume", type=float, default=0.9)
    args = ap.parse_args()

    memdir = Path(args.memdir)
    token = _read_token(memdir)
    if not token:
        raise SystemExit(f"Missing/empty ipc_token.txt at: {memdir / 'ipc_token.txt'}")

    outbox = memdir / "ipc" / "outbox"
    quarantine = memdir / "ipc" / "quarantine"
    outbox.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)

    # Writer publishes back into ipc/inbox (default behavior)
    writer = IPCFileWriter(memdir=memdir, src="mouth")

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", int(args.default_rate))
    except Exception:
        pass
    try:
        engine.setProperty("volume", max(0.0, min(1.0, float(args.default_volume))))
    except Exception:
        pass
    _pick_voice(engine, args.default_voice)

    print(f"[MOUTH] watching outbox: {outbox}")
    while True:
        files = sorted(outbox.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            time.sleep(args.poll)
            continue

        for p in files:
            try:
                raw = p.read_text(encoding="utf-8-sig", errors="strict").lstrip("\ufeff")
                msg = json.loads(raw)
            except Exception:
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            if not isinstance(msg, dict):
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            # auth gate
            if msg.get("auth") != token:
                try:
                    p.rename(quarantine / p.name)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

            payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
            text = str(payload.get("text", "") or "").strip()
            if not text:
                try:
                    p.unlink()
                except Exception:
                    pass
                continue

            corr = str(msg.get("correlation_id") or "")
            expected_sha1 = str(payload.get("expected_sha1", "") or "").strip()

            # per-utterance overrides
            voice = payload.get("voice", None) or args.default_voice
            rate = payload.get("rate", args.default_rate)
            volume = payload.get("volume", args.default_volume)

            try:
                if voice:
                    _pick_voice(engine, str(voice))
                engine.setProperty("rate", int(rate))
                engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
            except Exception:
                pass

            t0 = time.time()
            ok = True
            err = ""
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                ok = False
                err = repr(e)
            dt = time.time() - t0

            # report back to MB
            writer.publish(
                topic="act/spoken",
                payload={
                    "id": payload.get("id"),
                    "text": text,
                    "ok": ok,
                    "error": err,
                    "duration_s": dt,
                    "expected_sha1": expected_sha1,
                },
                correlation_id=corr or None,
                meta={"via": "mb_mouth_daemon"},
            )

            # delete after processing
            try:
                p.unlink()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
