"""Unit tests for app/observability/logger.py."""

from __future__ import annotations

from app.observability.logger import (
    clear_captured_json_logs,
    get_captured_json_logs,
    log_intent,
    log_outcome,
)
from app.observability.tracing import clear_recorded_spans, start_trace_span


class TestStructuredLogger:
    """Tests for JSON logging, PII redaction filter, and Intent vs. Outcome capture."""

    def test_intent_and_outcome_json_logging_with_pii_redaction(self) -> None:
        clear_captured_json_logs()
        clear_recorded_spans()

        with start_trace_span("test_dealership_turn", {"customer.tier": "tier_1"}):
            log_intent(
                action_name="search_dealership_inventory",
                intended_arguments={
                    "body_style": "suv",
                    "customer_email": "buyer@private.com",
                    "ssn": "123-45-6789",
                },
                agent_name="inventory_specialist",
            )
            log_outcome(
                action_name="search_dealership_inventory",
                status="success",
                actual_outcome={"matched_count": 3, "note": "Call 800-555-1212"},
                agent_name="inventory_specialist",
                latency_ms=42.5,
            )

        logs = get_captured_json_logs()
        assert len(logs) == 2

        intent_log = logs[0]
        outcome_log = logs[1]

        assert intent_log["event_type"] == "INTENT"
        assert intent_log["action_name"] == "search_dealership_inventory"
        assert intent_log["agent_name"] == "inventory_specialist"
        assert intent_log["intended_arguments"]["customer_email"] == "[REDACTED_EMAIL]"
        assert intent_log["intended_arguments"]["ssn"] == "[REDACTED_SSN]"
        assert intent_log["trace_id"] is not None
        assert intent_log["span_id"] is not None

        assert outcome_log["event_type"] == "OUTCOME"
        assert outcome_log["status"] == "success"
        assert outcome_log["actual_outcome"]["note"] == "Call [REDACTED_PHONE]"
        assert outcome_log["trace_id"] == intent_log["trace_id"]
