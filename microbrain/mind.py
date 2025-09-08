"""
microbrain.mind - entry module and runnable entrypoint.
Safe minimal scaffold so the CLI works.
"""

import asyncio
from typing import Any


async def main_async() -> Any:
    # TODO: wire your real async startup here
    print("microbrain: mind_async() starting…")
    return 0


def main() -> None:
    """Synchronous entry point that runs the async core."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("microbrain: interrupted")


if __name__ == "__main__":
    main()
