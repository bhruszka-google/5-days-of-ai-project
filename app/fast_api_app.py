"""FastAPI Cloud Run & Agent Runtime Entrypoint for AutoDeal Copilot.

Exposes the Google ADK FastAPI application (`google.adk.cli.fast_api.get_fast_api_app`)
alongside dedicated dealership REST endpoints:
- `GET /healthz`: Readiness and liveness probe reporting model routing tiers and
  active guardrails.
- `POST /v1/offers/generate`: Deterministic Top-3 Vehicle Offer workflow endpoint
  backed by `generate_top_three_offers_workflow`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from pydantic import BaseModel, ConfigDict, Field

from app.agent import generate_top_three_offers_workflow
from app.config import DEFAULT_CONFIG
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import init_tracing, traced_operation
from app.routing.model_router import DEFAULT_MODEL_ROUTER

AGENTS_DIR = str(Path(__file__).resolve().parent.parent)


class OfferGenerationRequest(BaseModel):
    """Strict Pydantic request schema for `POST /v1/offers/generate`."""

    model_config = ConfigDict(extra="forbid")

    user_query: str = Field(
        ...,
        min_length=3,
        description="Customer or salesperson desk request text.",
    )
    session_id: str = Field(
        default="cloud-run-desk-session",
        min_length=3,
        description="Persistent session identifier.",
    )
    customer_name: str = Field(
        default="Valued Dealership Guest",
        min_length=1,
        description="Customer name or buyer label (automatically scrubbed for PII).",
    )
    body_style: Optional[str] = Field(
        default="suv",
        description="Optional vehicle body style filter ('sedan', 'suv', 'crossover', 'truck', 'ev').",
    )
    max_price: Optional[float] = Field(
        default=52000.0,
        gt=0.0,
        description="Maximum selling price filter in USD.",
    )
    min_seating: Optional[int] = Field(
        default=None,
        ge=2,
        le=8,
        description="Optional minimum seating capacity.",
    )
    fuel_type: Optional[str] = Field(
        default=None,
        description="Optional fuel/propulsion filter ('gas', 'hybrid', 'phev', 'ev').",
    )
    down_payment: float = Field(
        default=4000.0,
        ge=0.0,
        description="Cash down payment in USD.",
    )
    credit_score: int = Field(
        default=740,
        ge=300,
        le=850,
        description="Customer FICO credit score (300 to 850).",
    )
    loan_term_months: int = Field(
        default=60,
        description="Loan term in months (24, 36, 48, 60, or 72).",
    )
    trade_in_year: Optional[int] = Field(default=None, ge=2000, le=2027)
    trade_in_make: Optional[str] = Field(default=None)
    trade_in_model: Optional[str] = Field(default=None)
    trade_in_mileage: Optional[int] = Field(default=None, ge=0, le=350000)
    trade_in_condition: str = Field(default="good")
    trade_in_payoff: float = Field(default=0.0, ge=0.0)
    requested_discount_amount: Optional[float] = Field(default=None, gt=0.0)
    discount_reason: str = Field(default="Customer desk negotiation request")
    manager_approval_token: Optional[str] = Field(default=None)


@traced_operation("autodeal.fastapi.create_fast_api_app")
def create_fast_api_app(
    agents_dir: str = AGENTS_DIR,
    otel_to_cloud: Optional[bool] = None,
) -> FastAPI:
    """Constructs the production ADK FastAPI application for Cloud Run.

    Uses `google.adk.cli.fast_api.get_fast_api_app(agents_dir=..., web=False, otel_to_cloud=True)`
    when running in Google Cloud and falls back gracefully when local/offline
    test environments do not have GCP telemetry credentials installed.

    Args:
        agents_dir: Root directory containing the `app/` ADK agent package.
        otel_to_cloud: Whether to enable Cloud Trace OTLP exporter in ADK.

    Returns:
        FastAPI: Configured FastAPI application with `/healthz` and
        `/v1/offers/generate` routes registered.
    """
    init_tracing()
    enable_cloud_otel = (
        otel_to_cloud
        if otel_to_cloud is not None
        else os.environ.get("OTEL_TRACES_EXPORTER", "").lower() == "google_cloud_trace"
    )

    log_intent(
        action_name="create_fast_api_app",
        intended_arguments={
            "agents_dir": agents_dir,
            "otel_to_cloud": enable_cloud_otel,
        },
        agent_name="fast_api_entrypoint",
    )

    try:
        fast_api_app: FastAPI = get_fast_api_app(
            agents_dir=agents_dir,
            web=False,
            otel_to_cloud=True if enable_cloud_otel else False,
        )
    except Exception:
        fast_api_app = get_fast_api_app(
            agents_dir=agents_dir,
            web=False,
            otel_to_cloud=False,
        )

    @fast_api_app.get("/healthz", tags=["ops"])
    def healthcheck() -> Dict[str, Any]:
        """Cloud Run readiness and liveness probe endpoint."""
        log_intent(
            action_name="healthcheck",
            intended_arguments={"endpoint": "/healthz"},
            agent_name="fast_api_entrypoint",
        )
        payload = {
            "status": "ok",
            "service": "autodeal-copilot",
            "app_name": DEFAULT_CONFIG.app_name,
            "fast_model": DEFAULT_CONFIG.fast_model,
            "pro_model": DEFAULT_CONFIG.pro_model,
            "routing_table": DEFAULT_MODEL_ROUTER.get_routing_summary(),
        }
        log_outcome(
            action_name="healthcheck",
            status="ok",
            actual_outcome=payload,
            agent_name="fast_api_entrypoint",
        )
        return payload

    @fast_api_app.post("/v1/offers/generate", tags=["dealership"])
    def generate_offers_endpoint(request: OfferGenerationRequest) -> Dict[str, Any]:
        """Generates Top 3 Tailored Vehicle Offers for a dealership desk session."""
        log_intent(
            action_name="generate_offers_endpoint",
            intended_arguments={
                "session_id": request.session_id,
                "body_style": request.body_style,
                "max_price": request.max_price,
            },
            agent_name="fast_api_entrypoint",
        )
        result = generate_top_three_offers_workflow(
            request.user_query,
            session_id=request.session_id,
            customer_name=request.customer_name,
            body_style=request.body_style,
            max_price=request.max_price,
            min_seating=request.min_seating,
            fuel_type=request.fuel_type,
            down_payment=request.down_payment,
            credit_score=request.credit_score,
            loan_term_months=request.loan_term_months,
            trade_in_year=request.trade_in_year,
            trade_in_make=request.trade_in_make,
            trade_in_model=request.trade_in_model,
            trade_in_mileage=request.trade_in_mileage,
            trade_in_condition=request.trade_in_condition,
            trade_in_payoff=request.trade_in_payoff,
            requested_discount_amount=request.requested_discount_amount,
            discount_reason=request.discount_reason,
            manager_approval_token=request.manager_approval_token,
        )
        log_outcome(
            action_name="generate_offers_endpoint",
            status=str(result.get("status", "success")),
            actual_outcome={
                "session_id": request.session_id,
                "status": result.get("status"),
            },
            agent_name="fast_api_entrypoint",
        )
        return result

    log_outcome(
        action_name="create_fast_api_app",
        status="success",
        actual_outcome={"routes_registered": ["/healthz", "/v1/offers/generate"]},
        agent_name="fast_api_entrypoint",
    )
    return fast_api_app


app: FastAPI = create_fast_api_app()
