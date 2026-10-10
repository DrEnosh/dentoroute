"""Logging setup. API keys are never logged: a filter masks any configured secret."""
from __future__ import annotations

import logging
import sys

from backend.utils.config import get_settings

_CONFIGURED = False


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        for secret in get_settings().secrets():
            if secret and secret in msg:
                msg = msg.replace(secret, "***REDACTED***")
        record.msg = msg
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        )
        handler.addFilter(RedactSecretsFilter())
        root = logging.getLogger("dentoroute")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"dentoroute.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: object) -> None:
    """Structured key=value line: case ID, model, latency, success, validation, ..."""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, "%s %s", event, parts)
