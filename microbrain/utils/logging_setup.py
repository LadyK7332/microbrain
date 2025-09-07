import logging
from logging import Logger

def configure_logging(level: str = "INFO") -> Logger:
    lv = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lv, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("microbrain")
