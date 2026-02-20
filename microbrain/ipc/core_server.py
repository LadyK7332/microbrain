# microbrain/ipc/core_server.py
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .ndjson import dumps_line, is_too_big, loads_line
from .token import DEFAULT_TOKEN_PATH, ensure_token_file, read_token

OnMessage = Callable[[dict], Awaitable[None] | None]

ALLOWED_PREFIXES = ("rt/", "percept/", "act/", "plan/", "input/")

class CoreIpcServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 17701,
        token_path: Path = DEFAULT_TOKEN_PATH,
        on_message: Optional[OnMessage] = None,
        debug: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.token_path = token_path
        self.on_message = on_message
        self.debug = debug
        self._token = ""

    def load_token(self) -> None:
        ensure_token_file(self.token_path)
        self._token = read_token(self.token_path)

    def _auth_ok(self, msg: dict) -> bool:
        return msg.get("auth") == self._token

    def _topic_ok(self, msg: dict) -> bool:
        topic = msg.get("topic")
        if not isinstance(topic, str) or not topic:
            return False
        return topic.startswith(ALLOWED_PREFIXES)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        if self.debug:
            print(f"[ipc] client connected: {peer}")

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                if is_too_big(line):
                    writer.write(dumps_line({"ok": False, "err": "line_too_big"}))
                    await writer.drain()
                    continue

                try:
                    msg = loads_line(line)
                except Exception:
                    writer.write(dumps_line({"ok": False, "err": "bad_json"}))
                    await writer.drain()
                    continue

                if not isinstance(msg, dict):
                    writer.write(dumps_line({"ok": False, "err": "not_object"}))
                    await writer.drain()
                    continue

                if not self._auth_ok(msg):
                    writer.write(dumps_line({"ok": False, "err": "auth"}))
                    await writer.drain()
                    continue

                if not self._topic_ok(msg):
                    writer.write(dumps_line({"ok": False, "err": "topic"}))
                    await writer.drain()
                    continue

                # optional hook into MicroBrain bus later
                if self.on_message is not None:
                    res = self.on_message(msg)
                    if asyncio.iscoroutine(res):
                        await res  # type: ignore[func-returns-value]

                writer.write(dumps_line({"ok": True, "ts": int(time.time() * 1000)}))
                await writer.drain()

        finally:
            if self.debug:
                print(f"[ipc] client disconnected: {peer}")
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def run(self) -> None:
        self.load_token()
        server = await asyncio.start_server(self._handle_client, host=self.host, port=self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"[ipc] core server listening on {addrs} token={self.token_path}")
        async with server:
            await server.serve_forever()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17701)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    srv = CoreIpcServer(host=args.host, port=args.port, debug=args.debug)
    asyncio.run(srv.run())

if __name__ == "__main__":
    main()
