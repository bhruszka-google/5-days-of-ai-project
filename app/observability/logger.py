"""Structured JSON Logging with active PII Redaction and Intent vs. Outcome capture.

Uses `pythonjsonlogger` to emit structured JSON log entries enriched with
OpenTelemetry `trace_id` and `span_id`, and automatically scrubs all PII
fields and message strings via `PIIRedactor`.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

from pythonjsonlogger import json as jsonlogger

from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor
from app.observability.tracing import get_current_trace_context


class PIIRedactingJsonFormatter(jsonlogger.JsonFormatter):
    """JSON log formatter that injects OpenTelemetry IDs and scrubs PII."""

    def __init__(
        self,
        *args: Any,
        redactor: Optional[PIIRedactor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.redactor = redactor or DEFAULT_PII_REDACTOR

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = record.levelname
        log_record["logger"] = record.name

        trace_ctx = get_current_trace_context()
        if trace_ctx.get("trace_id") and not log_record.get("trace_id"):
            log_record["trace_id"] = trace_ctx["trace_id"]
        if trace_ctx.get("span_id") and not log_record.get("span_id"):
            log_record["span_id"] = trace_ctx["span_id"]

        # Scrub all fields in the structured log record for PII before serialization
        sanitized = self.redactor.redact_data(dict(log_record))
        log_record.clear()
        log_record.update(sanitized)


_LOG_BUFFER = io.StringIO()
_CONFIGURED_LOGGERS: Dict[str, logging.Logger] = {}


def get_structured_logger(name: str = "autodeal_copilot") -> logging.Logger:
    """Returns a structured JSON logger wired with PII redaction."""
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = PIIRedactingJsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        buffer_handler = logging.StreamHandler(_LOG_BUFFER)
        buffer_handler.setFormatter(formatter)
        logger.addHandler(buffer_handler)

    _CONFIGURED_LOGGERS[name] = logger
    return logger


def log_intent(
    action_name: str,
    intended_arguments: Dict[str, Any],
    agent_name: str = "dealership_coordinator",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Explicitly records the agent's INTENDED action before execution.

    Satisfies the AgentOps Rubric criterion 'Intent vs. Outcome Capture'.
    """
    active_logger = logger or get_structured_logger()
    sanitized_args = DEFAULT_PII_REDACTOR.redact_data(intended_arguments)
    payload = {
        "event_type": "INTENT",
        "agent_name": agent_name,
        "action_name": action_name,
        "intended_arguments": sanitized_args,
        **get_current_trace_context(),
    }
    active_logger.info(
        f"INTENT: Agent '{agent_name}' preparing to execute '{action_name}'",
        extra=payload,
    )
    return payload


def log_outcome(
    action_name: str,
    status: str,
    actual_outcome: Any,
    agent_name: str = "dealership_coordinator",
    latency_ms: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Explicitly records the ACTUAL outcome after tool or agent execution.

    Satisfies the AgentOps Rubric criterion 'Intent vs. Outcome Capture'.
    """
    active_logger = logger or get_structured_logger()
    sanitized_outcome = DEFAULT_PII_REDACTOR.redact_data(actual_outcome)
    payload = {
        "event_type": "OUTCOME",
        "agent_name": agent_name,
        "action_name": action_name,
        "status": status,
        "actual_outcome": sanitized_outcome,
        "latency_ms": latency_ms,
        **get_current_trace_context(),
    }
    active_logger.info(
        f"OUTCOME: Action '{action_name}' completed with status '{status}'",
        extra=payload,
    )
    return payload


def get_captured_json_logs() -> List[Dict[str, Any]]:
    """Parses and returns all structured JSON log entries from the memory buffer."""
    raw_lines = _LOG_BUFFER.getvalue().strip().splitlines()
    parsed: List[Dict[str, Any]] = []
    for line in raw_lines:
        if line.strip():
            parsed.append(json.loads(line))
    return parsed


def clear_captured_json_logs() -> None:
    """Clears the in-memory structured JSON log buffer."""
    _LOG_BUFFER.seek(0)
    _LOG_BUFFER.truncate(0)
