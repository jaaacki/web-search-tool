"""Request correlation and log formatting shared by the API and provider controls."""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def current_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Copy the current request id onto every record so the handler can format it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class ContextFormatter(logging.Formatter):
    """Keep the existing text lines; append the request id only inside a request."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = getattr(record, "request_id", "-")
        return line if request_id == "-" else f"{line} [req={request_id}]"


class JsonFormatter(logging.Formatter):
    """One JSON object per line for collectors; enabled with LOG_FORMAT=json."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "level": record.levelname,
                "logger": record.name,
                "request_id": getattr(record, "request_id", "-"),
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )
