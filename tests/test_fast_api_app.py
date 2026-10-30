"""Unit tests for the Cloud Run FastAPI Entrypoint (`app/fast_api_app.py`)."""

from __future__ import annotations

from app.fast_api_app import OfferGenerationRequest, app, create_fast_api_app
from fastapi.testclient import TestClient


def test_healthz_endpoint_returns_model_routing_and_ok_status() -> None:
    """Verifies GET /healthz returns status='ok' and active Gemini 3 model tiers."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "autodeal-copilot"
    assert data["fast_model"] == "gemini-3-flash-preview"
    assert data["pro_model"] == "gemini-3.1-pro-preview"
    assert data["routing_table"]["top_offers_strategist"] == "gemini-3.1-pro-preview"
    assert data["routing_table"]["dealership_coordinator"] == "gemini-3-flash-preview"


def test_generate_offers_endpoint_returns_top_three_offers_and_scrubs_pii() -> None:
    """Verifies POST /v1/offers/generate produces 3 distinct offers and scrubs PII."""
    custom_app = create_fast_api_app(otel_to_cloud=False)
    client = TestClient(custom_app)

    payload = {
        "user_query": (
            "Family of 6 needs a 3-row hybrid SUV under $50,000, 750 credit score, "
            "$4,500 down, trading in a 2021 Honda CR-V with 40,000 miles. "
            "SSN: 123-45-6789, email: buyer@example.com."
        ),
        "session_id": "test-fastapi-session-01",
        "customer_name": "FastAPI Test Buyer (buyer@example.com)",
        "body_style": "suv",
        "max_price": 50000.0,
        "min_seating": 6,
        "fuel_type": "hybrid",
        "down_payment": 4500.0,
        "credit_score": 750,
        "loan_term_months": 60,
        "trade_in_year": 2021,
        "trade_in_make": "Honda",
        "trade_in_model": "CR-V",
        "trade_in_mileage": 40000,
        "trade_in_condition": "good",
        "trade_in_payoff": 3000.0,
    }
    response = client.post("/v1/offers/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert len(body["offer_sheet"]["offers"]) == 3
    raw_text = response.text
    assert "123-45-6789" not in raw_text
    assert "buyer@example.com" not in raw_text


def test_offer_generation_request_forbids_extra_fields() -> None:
    """Verifies OfferGenerationRequest rejects unknown extra fields (422)."""
    client = TestClient(app)
    response = client.post(
        "/v1/offers/generate",
        json={
            "user_query": "Find me a hybrid SUV",
            "unexpected_field": "not_allowed",
        },
    )
    assert response.status_code == 422
    assert OfferGenerationRequest.model_config.get("extra") == "forbid"
