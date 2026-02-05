from __future__ import annotations

import time
import os
import subprocess
from pathlib import Path



def spawn_tail_window(memdir: str, *, tail: int = 200) -> tuple[bool, str]:
    """Spawn a separate console window tailing the debug log (Windows only).

    Returns (spawned, reason). `reason` is best-effort and useful in --debug.
    """
    if os.name != "nt":
        return False, "not_windows"

    log_path = Path(memdir) / "logs" / "microbrain.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file exists so the tail window has something to follow immediately.
        if not log_path.exists():
            log_path.touch()
    except Exception as e:
        return False, f"prepare_log_failed:{e!r}"

    # Use a new console so the main UI can keep running.
    # Equivalent PowerShell: Get-Content -Path "<log>" -Wait -Tail <tail>
    # Use single quotes to reduce escaping issues.
    ps_cmd = f"Get-Content -Path '{str(log_path)}' -Wait -Tail {int(tail)}"
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        subprocess.Popen(
            ["powershell.exe", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            creationflags=creationflags,
        )
        return True, "ok"
    except FileNotFoundError as e:
        return False, f"powershell_missing:{e!r}"
    except Exception as e:
        return False, f"spawn_failed:{e!r}"


def tail_log(memdir: str) -> int:
    """Tail the debug log written under memdir/logs/microbrain.log.

    Returns process exit code.
    """
    log_path = Path(memdir) / "logs" / "microbrain.log"
    if not log_path.exists():
        print(f"[debug-tail] log file not found: {log_path}")
        print("[debug-tail] Start MicroBrain with --debug --ui textual first (or create the file).")
        return 2

    print(f"[debug-tail] Tailing: {log_path}")
    print("[debug-tail] Ctrl+C to stop.\n")

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            # Start at end
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.25)
                    continue
                print(line, end="")
    except KeyboardInterrupt:
        return 0
