"""Unit tests for `app/tools/schemas.py` strict Pydantic input and output models."""

from __future__ import annotations

import pytest
from app.tools.schemas import (
    FinancingCalculationInput,
    InventorySearchInput,
    ManagerDiscountInput,
    MarginCheckInput,
    OfferSheetInput,
    RebateLookupInput,
    ToolErrorResponse,
    TopThreeOffersResponse,
    TradeInAppraisalInput,
    VehicleOfferCard,
    build_guided_error_response,
)
from pydantic import ValidationError


def test_inventory_search_input_valid_and_normalizes() -> None:
    model = InventorySearchInput(
        body_style="SUV",
        max_price=50000.0,
        min_seating=7,
        fuel_type="Hybrid",
        condition="NEW",
        feature_keyword="captain chairs",
        limit=5,
    )
    assert model.body_style == "suv"
    assert model.fuel_type == "hybrid"
    assert model.condition == "new"
    assert model.limit == 5


def test_inventory_search_input_rejects_extra_fields_and_invalid_values() -> None:
    with pytest.raises(ValidationError):
        InventorySearchInput(body_style="suv", unknown_field="forbidden")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        InventorySearchInput(body_style="spaceship")

    with pytest.raises(ValidationError):
        InventorySearchInput(max_price=-100.0)

    with pytest.raises(ValidationError):
        InventorySearchInput(min_seating=12)


def test_trade_in_appraisal_input_validation() -> None:
    valid = TradeInAppraisalInput(
        year=2021,
        make="Honda",
        model="CR-V",
        mileage=45000,
        condition="Rough",
        existing_loan_payoff=4200.0,
    )
    assert valid.condition == "rough"

    with pytest.raises(ValidationError):
        TradeInAppraisalInput(
            year=1992,
            make="Honda",
            model="Civic",
            mileage=120000,
            condition="good",
        )

    with pytest.raises(ValidationError):
        TradeInAppraisalInput(
            year=2021,
            make="Honda",
            model="CR-V",
            mileage=45000,
            condition="salvage",  # type: ignore[arg-type]
        )


def test_financing_calculation_input_enforces_allowed_loan_terms() -> None:
    valid = FinancingCalculationInput(
        vehicle_identifier="STK-2026-001",
        down_payment=3000.0,
        trade_in_equity=5000.0,
        credit_score=740,
        loan_term_months=60,
    )
    assert valid.loan_term_months == 60

    with pytest.raises(ValidationError, match="Allowed loan terms"):
        FinancingCalculationInput(
            vehicle_identifier="STK-2026-001",
            loan_term_months=45,
        )

    with pytest.raises(ValidationError):
        FinancingCalculationInput(
            vehicle_identifier="STK-2026-001",
            credit_score=250,
        )


def test_rebate_margin_and_manager_discount_inputs_forbid_extra() -> None:
    rebate_in = RebateLookupInput(
        vehicle_identifier="STK-2026-001",
        customer_qualifiers=["loyalty", "first_time_buyer"],
    )
    assert rebate_in.vehicle_identifier == "STK-2026-001"

    margin_in = MarginCheckInput(
        vehicle_identifier="STK-2026-001",
        proposed_selling_price=46000.0,
        rebates_total=1250.0,
    )
    assert margin_in.proposed_selling_price == 46000.0

    discount_in = ManagerDiscountInput(
        vehicle_identifier="STK-2026-001",
        requested_discount_amount=2000.0,
        reason="Competitive match with neighboring dealer quote",
    )
    assert discount_in.requested_discount_amount == 2000.0

    with pytest.raises(ValidationError):
        ManagerDiscountInput(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=2000.0,
            reason="ok",
        )


def test_vehicle_offer_card_and_top_three_offers_response() -> None:
    card = VehicleOfferCard(
        rank=1,
        tier_label="Best Overall Match",
        vin="5TDKDRBH1TS100001",
        stock_number="STK-2026-001",
        vehicle_title="2026 Toyota Highlander Hybrid XLE",
        msrp=48250.0,
        selling_price=46800.0,
        rebates_applied=1250.0,
        net_cap_cost=41550.0,
        estimated_monthly_finance=774.61,
        apr_percent=4.49,
        loan_term_months=60,
        estimated_monthly_lease_36m=452.10,
        why_it_fits="Ideal 3-row hybrid SUV with captain chairs and AWD.",
        floor_margin_compliant=True,
    )
    sheet = TopThreeOffersResponse(
        customer_name="Jane Buyer",
        customer_summary="Prepared Top 3 Offers for Jane Buyer.",
        offers=[card],
        compliance_disclaimer="Truth-in-Lending Regulation Z APR disclosure statement.",
    )
    assert sheet.status == "success"
    assert len(sheet.offers) == 1

    with pytest.raises(ValidationError):
        VehicleOfferCard(
            rank=4,
            tier_label="Best Overall Match",
            vin="5TDKDRBH1TS100001",
            stock_number="STK-2026-001",
            vehicle_title="2026 Toyota Highlander Hybrid XLE",
            msrp=48250.0,
            selling_price=46800.0,
            rebates_applied=1250.0,
            net_cap_cost=41550.0,
            estimated_monthly_finance=774.61,
            apr_percent=4.49,
            loan_term_months=60,
            estimated_monthly_lease_36m=452.10,
            why_it_fits="Invalid rank test.",
            floor_margin_compliant=True,
        )

    with pytest.raises(ValidationError):
        OfferSheetInput(
            customer_name="Jane Buyer",
            best_match_vin="STK-2026-001",
            value_pick_vin="STK-2026-001",
            upgrade_pick_vin="STK-2026-002",
        )


def test_tool_error_response_builder() -> None:
    err = build_guided_error_response(
        error_code="SAMPLE_ERR",
        message="Something failed validation.",
        recovery_suggestion="Adjust loan_term_months to 60 and retry.",
    )
    parsed = ToolErrorResponse(**err)
    assert parsed.status == "error"
    assert parsed.error_code == "SAMPLE_ERR"
