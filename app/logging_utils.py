"""Two kinds of record for every request.

1. One JSON line per request on stdout. Docker, Render and Kubernetes all collect stdout, and
   JSON lines can be searched and counted by log tools (ELK, CloudWatch, Datadog...).
2. One row per prediction in a small SQLite database (PREDICTION_LOG, default logs/predictions.db).
   Day 4 reads these rows to look at what the model saw and said.
"""
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from machine_guard.features import RAW_FEATURES

logger = logging.getLogger("machine_guard.api")


class JsonFormatter(logging.Formatter):
    """Format a log record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def setup_logging() -> None:
    """Send our logs to stdout as JSON, and quieten uvicorn's plain-text access log."""
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Our middleware already logs every request as JSON; uvicorn's own access line would be a duplicate.
    logging.getLogger("uvicorn.access").disabled = True


def log_event(event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})


# ---- SQLite prediction log ------------------------------------------------------------------

COLUMNS = ["ts", "request_id", *RAW_FEATURES, "failure_probability", "prediction",
           "model_version", "latency_ms"]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    ts TEXT, request_id TEXT,
    type TEXT, air_temp_k REAL, process_temp_k REAL, rotational_speed_rpm REAL,
    torque_nm REAL, tool_wear_min REAL,
    failure_probability REAL, prediction INTEGER, model_version TEXT, latency_ms REAL
)
"""


def prediction_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG", "logs/predictions.db"))


def save_prediction(row: dict) -> None:
    """Append one prediction to the SQLite log. Creates the folder and table if needed."""
    path = prediction_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    placeholders = ", ".join("?" for _ in COLUMNS)
    with sqlite3.connect(path) as conn:
        conn.execute(CREATE_TABLE)
        conn.execute(f"INSERT INTO predictions ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                     [row[c] for c in COLUMNS])
    conn.close()
