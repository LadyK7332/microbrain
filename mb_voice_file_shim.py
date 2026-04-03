from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def resolve_memdir() -> Path:
    env_memdir = os.getenv('MB_MEMDIR')
    if env_memdir:
        return Path(env_memdir)
    return Path(r'Z:\memory')


def main() -> int:
    memdir = resolve_memdir()
    queue_dir = memdir / 'reading' / 'queue'
    archive_dir = memdir / 'reading' / 'archive'
    queue_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    poll_s = 0.25
    print(f'[voice_file_shim] watching: {queue_dir}')
    try:
        while True:
            files = sorted(queue_dir.glob('*.txt'))
            if files:
                path = files[0]
                try:
                    text = path.read_text(encoding='utf-8').strip()
                except Exception:
                    text = ''
                if text:
                    print(f'[voice_file_shim] read> {text}')
                try:
                    path.rename(archive_dir / path.name)
                except Exception:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
            time.sleep(poll_s)
    except KeyboardInterrupt:
        print('[voice_file_shim] stopped')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
