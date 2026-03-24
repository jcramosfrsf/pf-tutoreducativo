"""Application logging setup."""

import logging

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - fallback for environments without rich
    RichHandler = None


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def setup_logger(level: str = "INFO") -> None:
    """Configure root logging with rich handler when available."""
    root_logger = logging.getLogger()
    resolved_level = _resolve_level(level)

    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return

    if RichHandler is not None:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
        logging.basicConfig(
            level=resolved_level,
            format="%(message)s",
            datefmt=DEFAULT_DATE_FORMAT,
            handlers=[handler],
        )
        return

    logging.basicConfig(
        level=resolved_level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger by module name."""
    return logging.getLogger(name)
