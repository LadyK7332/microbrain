from __future__ import annotations

import logging


def configure_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Central logging setup for MicroBrain.

    - Respects the requested log level (DEBUG/INFO/WARNING/ERROR).
    - Uses a single basicConfig so all loggers inherit it.
    """
    # Map string to logging level; fall back to INFO on weird input
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Main project logger
    logger = logging.getLogger("microbrain")
    logger.setLevel(level)
    return logger
