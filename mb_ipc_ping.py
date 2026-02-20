# mb_ipc_ping.py
from __future__ import annotations

import asyncio
import time

from microbrain.ipc.ndjson import dumps_line, loads_line
from microbrain.ipc.token import DEFAULT_TOKEN_PATH, ensure_token_file, read_token

HOST = "127.0.0.1"
PORT = 17701

async def main() -> None:
    ensure_token_file(DEFAULT_TOKEN_PATH)
    token = read_token(DEFAULT_TOKEN_PATH)

    reader, writer = await asyncio.open_connection(HOST, PORT)
    msg = {
        "v": 1,
        "ts": int(time.time() * 1000),
        "src": "ping",
        "topic": "rt/telemetry/ping",
        "auth": token,
        "payload": {"hello": "world"},
    }
    writer.write(dumps_line(msg))
    await writer.drain()

    line = await reader.readline()
    print(loads_line(line))

    writer.close()
    await writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
