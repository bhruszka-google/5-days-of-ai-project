"""Unit tests for app/observability/tracing.py."""

from __future__ import annotations

from app.observability.tracing import (
    clear_recorded_spans,
    get_current_trace_context,
    get_recorded_spans,
    start_trace_span,
    traced_operation,
)


class TestDistributedTracing:
    """Tests for OpenTelemetry span hierarchy and trace context propagation."""

    def test_opentelemetry_decorator_links_parent_and_child_spans(self) -> None:
        clear_recorded_spans()

        @traced_operation("autodeal.child_tool")
        def sample_tool(x: int) -> int:
            ctx = get_current_trace_context()
            assert ctx["trace_id"] is not None
            assert ctx["span_id"] is not None
            return x * 2

        with start_trace_span("autodeal.parent_agent", {"agent": "coordinator"}):
            result = sample_tool(21)

        assert result == 42
        spans = get_recorded_spans()
        span_names = [s.name for s in spans]
        assert "autodeal.child_tool" in span_names
        assert "autodeal.parent_agent" in span_names

        child_span = next(s for s in spans if s.name == "autodeal.child_tool")
        parent_span = next(s for s in spans if s.name == "autodeal.parent_agent")
        assert child_span.context.trace_id == parent_span.context.trace_id
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id
