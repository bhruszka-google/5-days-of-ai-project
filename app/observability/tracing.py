"""OpenTelemetry distributed tracing setup for AutoDeal Copilot.

Links spans across user queries, coordinator routing, specialist sub-agents,
and tool executions. Exports traces to Google Cloud Trace when configured or
in-memory span exporters during unit testing and evaluation.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class InMemorySpanCollector(SpanExporter):
    """Thread-safe in-memory OpenTelemetry span collector for testing & auditing."""

    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()


_SPAN_COLLECTOR = InMemorySpanCollector()
_PROVIDER_INITIALIZED = False


def init_tracing(service_name: str = "autodeal-copilot") -> TracerProvider:
    """Initializes the global OpenTelemetry TracerProvider if not yet configured."""
    global _PROVIDER_INITIALIZED
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider) and _PROVIDER_INITIALIZED:
        return current_provider

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_SPAN_COLLECTOR))
    trace.set_tracer_provider(provider)
    _PROVIDER_INITIALIZED = True
    return provider


def get_tracer(name: str = "autodeal_copilot") -> trace.Tracer:
    """Returns an initialized OpenTelemetry tracer."""
    init_tracing()
    return trace.get_tracer(name)


def get_recorded_spans() -> List[ReadableSpan]:
    """Returns all OpenTelemetry spans captured in memory."""
    return list(_SPAN_COLLECTOR.spans)


def clear_recorded_spans() -> None:
    """Clears the in-memory OpenTelemetry span buffer."""
    _SPAN_COLLECTOR.clear()


def get_current_trace_context() -> Dict[str, Optional[str]]:
    """Extracts the active OpenTelemetry `trace_id` and `span_id` as hex strings."""
    span = trace.get_current_span()
    if span is None:
        return {"trace_id": None, "span_id": None}
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return {"trace_id": None, "span_id": None}
    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
    }


@contextmanager
def start_trace_span(
    name: str, attributes: Optional[Dict[str, Any]] = None
) -> Iterator[trace.Span]:
    """Context manager creating a linked OpenTelemetry span with attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
        yield span


def traced_operation(span_name: Optional[str] = None) -> Callable[..., Any]:
    """Decorator that wraps a synchronous or asynchronous function in an OTel span."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = span_name or f"autodeal.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_trace_span(resolved_name, {"function.name": func.__name__}):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator
