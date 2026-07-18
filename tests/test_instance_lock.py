from __future__ import annotations

import json
import os

import pytest

from microbrain.utils.instance_lock import acquire_instance_lock


def test_instance_lock_blocks_second_live_owner(tmp_path):
    lock1 = acquire_instance_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            acquire_instance_lock(tmp_path)
    finally:
        lock1.release()


def test_instance_lock_clears_stale_owner(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    lock_path = runtime / "microbrain.instance.lock"
    lock_path.write_text(json.dumps({"pid": 99999999, "host": ""}), encoding="utf-8")

    lock = acquire_instance_lock(tmp_path)
    try:
        assert lock.owns_lock is True
        assert lock_path.exists()
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
    finally:
        lock.release()
