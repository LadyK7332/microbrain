from __future__ import annotations

import atexit
import os
import shlex
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

_LLAMA_PROC: subprocess.Popen | None = None


def _is_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(base: str, timeout: float = 0.8) -> bool:
    for path in ("/health", "/v1/models", "/"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as r:
                # treat any HTTP < 500 as "server is up"
                if 200 <= r.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            # also treat 4xx as "up" (endpoint may not exist on this build)
            if 400 <= e.code < 500:
                return True
        except Exception:
            pass
    return False


def is_server_up(host: str = "127.0.0.1", port: int = 8080) -> bool:
    base = f"http://{host}:{port}"
    return _is_listening(host, port) and _http_probe(base)


def find_llama_server_exe(explicit: str | None = None) -> str | None:
    """
    Returns full path to llama-server.exe if found.
    Search order:
      1) explicit path (MB_LLAMA_SERVER)
      2) PATH (shutil.which)
      3) common local build locations under repo root
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)

    which = shutil.which("llama-server.exe") or shutil.which("llama-server")
    if which:
        return which

    # common local build locations
    here = Path.cwd()
    candidates = [
        here / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        here / "build" / "bin" / "Release" / "llama-server.exe",
        here / "llama-server.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def start_llama_server(
    *,
    server_path: str,
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    threads: int | None = None,
    ngl: int | None = 999,
    extra_args: str | None = None,
    log_path: str | None = "llama_server.log",
) -> subprocess.Popen:
    """Start llama-server and return the Popen handle (does not block)."""
    cmd = [server_path, "--model", model_path, "--host", host, "--port", str(port)]
    if threads:
        cmd += ["--threads", str(threads)]
    if isinstance(ngl, int) and ngl >= 0:
        # alias "-ngl" also works; both map to n-gpu-layers
        cmd += ["--n-gpu-layers", str(ngl)]
    if extra_args:
        cmd += (
            list(extra_args)
            if isinstance(extra_args, list | tuple)
            else shlex.split(str(extra_args), posix=False)
        )

    # Prefer Vulkan build (no special env needed if compiled with it)
    # Commented out due to error, kept for just in case # env = os.environ.copy()

    stdout = open(log_path, "a", encoding="utf-8")
    stderr = subprocess.STDOUT

    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    def _cleanup() -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    atexit.register(_cleanup)
    return proc


def ensure_llama_server(
    *,
    model_path: str,
    server_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    threads: int | None = None,
    ngl: int | None = 999,
    backend: str = "auto",
    extra_args: str | None = None,
    wait_sec: int = 30,
) -> None:
    """If not already up, start llama-server and wait until it’s ready (or timeout)."""
    global _LLAMA_PROC
    if is_server_up(host, port):
        return

    sp = find_llama_server_exe(server_path)
    if not sp:
        raise RuntimeError("llama-server not found. Set MB_LLAMA_SERVER or ensure it’s on PATH.")

    _LLAMA_PROC = start_llama_server(
        server_path=sp,
        model_path=model_path,
        host=host,
        port=port,
        threads=threads or os.cpu_count() or 4,
        ngl=ngl,
        extra_args=extra_args,
    )

    base = f"http://{host}:{port}"
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if is_server_up(host, port):
            return
        time.sleep(0.5)

    raise TimeoutError(
        f"Timed out waiting for llama-server at {base}. "
        f"Check {Path('llama_server.log').resolve()} for details."
    )
