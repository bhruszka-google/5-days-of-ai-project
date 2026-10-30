"""Observability package for AutoDeal Copilot."""

from app.observability.logger import (
    PIIRedactingJsonFormatter,
    clear_captured_json_logs,
    get_captured_json_logs,
    get_structured_logger,
    log_intent,
    log_outcome,
)
from app.observability.tracing import (
    clear_recorded_spans,
    get_current_trace_context,
    get_recorded_spans,
    get_tracer,
    init_tracing,
    start_trace_span,
    traced_operation,
)

__all__ = [
    "PIIRedactingJsonFormatter",
    "clear_captured_json_logs",
    "clear_recorded_spans",
    "get_captured_json_logs",
    "get_current_trace_context",
    "get_recorded_spans",
    "get_structured_logger",
    "get_tracer",
    "init_tracing",
    "log_intent",
    "log_outcome",
    "start_trace_span",
    "traced_operation",
]
