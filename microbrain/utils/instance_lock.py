from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class InstanceLock:
    """Single-instance lock for one MicroBrain memdir.

    This prevents two full MB runtimes from using the same memory directory at
    the same time. Sidecars may share the current body, but there should only be
    one full MicroBrain body per memdir.

    On Windows this uses a kernel named mutex first. The JSON file is only owner
    metadata for humans/logs; it is not the authority. That avoids Windows race
    cases where a child process sees a half-written lock file and steals it.
    """

    path: Path
    fd: int | None = None
    owns_lock: bool = False
    mutex_handle: int | None = None
    mutex_name: str | None = None

    def release(self) -> None:
        if not self.owns_lock:
            return

        self.owns_lock = False

        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            finally:
                self.fd = None

        if self.mutex_handle is not None:
            _release_windows_mutex(self.mutex_handle)
            self.mutex_handle = None

        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            # Best effort. A stale metadata file is harmless because the actual
            # Windows authority is the named mutex, and non-Windows boot checks
            # whether the recorded pid is still alive.
            pass


def acquire_instance_lock(memdir: str | os.PathLike[str]) -> InstanceLock:
    """Acquire the single-MB-body lock for *memdir*.

    Raises RuntimeError when another live MicroBrain process already owns this
    memdir. Stale file metadata is cleaned automatically when safe.
    """

    runtime_dir = Path(memdir) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / "microbrain.instance.lock"

    lock = InstanceLock(path=lock_path)

    if os.name == "nt":
        mutex_name = _mutex_name_for_memdir(memdir)
        handle, already_exists = _acquire_windows_mutex(mutex_name)
        if already_exists:
            owner = _read_lock_owner(lock_path)
            owner_pid = _safe_int(owner.get("pid"))
            detail = f"owner_pid={owner_pid}" if owner_pid else "owner_pid=unknown"
            raise RuntimeError(
                "MicroBrain is already running for this memdir. "
                f"memdir={Path(memdir)} {detail} mutex={mutex_name} lock={lock_path}"
            )
        lock.mutex_handle = handle
        lock.mutex_name = mutex_name
        lock.owns_lock = True
        _write_lock_metadata(lock_path, memdir)
        atexit.register(lock.release)
        return lock

    payload = _lock_payload(memdir)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            owner = _read_lock_owner(lock_path)
            owner_pid = _safe_int(owner.get("pid"))
            owner_host = str(owner.get("host") or "")
            this_host = socket.gethostname()

            if owner_pid and (not owner_host or owner_host == this_host) and _pid_is_alive(owner_pid):
                raise RuntimeError(
                    "MicroBrain is already running for this memdir. "
                    f"memdir={Path(memdir)} owner_pid={owner_pid} lock={lock_path}"
                )

            # Stale, malformed, or foreign-host-unverifiable lock. Remove it so
            # the current process can claim the body slot.
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"MicroBrain instance lock exists and could not be cleared: {lock_path} ({exc})"
                ) from exc
            continue

        with os.fdopen(os.dup(fd), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        lock.fd = fd
        lock.owns_lock = True
        atexit.register(lock.release)
        return lock


def _lock_payload(memdir: str | os.PathLike[str]) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "created_at": time.time(),
        "memdir": str(Path(memdir)),
    }


def _write_lock_metadata(path: Path, memdir: str | os.PathLike[str]) -> None:
    payload = _lock_payload(memdir)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_lock_owner(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return {}
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _safe_int(value: Any) -> int | None:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return i if i > 0 else None


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        return _pid_is_alive_windows(pid)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_is_alive_windows(pid: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE

        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            close_handle(handle)
            return True

        err = ctypes.get_last_error()
        # Access denied means a process exists but we do not have rights to query it.
        return err == 5
    except Exception:
        # Conservative fallback: if Windows probing itself failed, do not risk
        # deleting a live owner's lock.
        return True


def _mutex_name_for_memdir(memdir: str | os.PathLike[str]) -> str:
    normalized = str(Path(memdir)).lower().replace("/", "\\")
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:32]
    # Local namespace avoids cross-session permission weirdness while still
    # protecting the normal same-user/same-session Windows launch case.
    return f"Local\\MicroBrain_{digest}"


def _acquire_windows_mutex(name: str) -> tuple[int, bool]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    create_mutex.restype = wintypes.HANDLE

    handle = create_mutex(None, True, name)
    if not handle:
        err = ctypes.get_last_error()
        raise RuntimeError(f"Could not create MicroBrain instance mutex {name!r}; winerr={err}")

    ERROR_ALREADY_EXISTS = 183
    already_exists = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
    if already_exists:
        _release_windows_mutex(handle)
    return int(handle), already_exists


def _release_windows_mutex(handle: int) -> None:
    if not handle:
        return

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        release_mutex = kernel32.ReleaseMutex
        release_mutex.argtypes = [wintypes.HANDLE]
        release_mutex.restype = wintypes.BOOL

        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        release_mutex(wintypes.HANDLE(handle))
        close_handle(wintypes.HANDLE(handle))
    except Exception:
        pass
