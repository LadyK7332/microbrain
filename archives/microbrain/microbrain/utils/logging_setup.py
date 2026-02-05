import logging
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str] = None, console: bool = True) -> None:
    """Configure root logging.

    - If log_file is provided, logs are also written to that file (parents created).
    - If console is False, no StreamHandler is added (useful for TUI apps).
    """
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    # Clear any existing handlers (re-run safe)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if console:
        sh = logging.StreamHandler()
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)
