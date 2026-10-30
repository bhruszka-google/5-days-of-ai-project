"""Unit tests for `app/tools/financing_tools.py` loan/lease math, floor margin, and offer sheet."""

from __future__ import annotations

from app.tools.financing_tools import (
    calculate_financing_and_lease_options,
    generate_customer_offer_sheet,
    verify_dealership_floor_margin,
)


def test_calculate_financing_and_lease_options_accuracy() -> None:
    # STK-2026-001: 2026 Toyota Highlander Hybrid XLE
    # selling_price = 46800, hybrid rebate = 1250, trade_in > 0 adds 750 loyalty rebate -> rebates_applied = 2000
    # down_payment = 4000, trade_in_equity = 5000 -> net_cap_cost = 46800 - 2000 - 4000 - 5000 = 35800
    # credit_score = 740 (tier_1_excellent) -> 60m APR = 4.49%
    quote = calculate_financing_and_lease_options(
        vehicle_identifier="STK-2026-001",
        down_payment=4000.0,
        trade_in_equity=5000.0,
        credit_score=740,
        loan_term_months=60,
        include_rebates=True,
    )
    assert quote["status"] == "success"
    assert quote["vin"] == "5TDKDRBH1TS100001"
    assert quote["rebates_applied"] == 2000.0
    assert quote["net_cap_cost"] == 35800.0
    assert quote["apr_percent"] == 4.49
    assert quote["credit_tier"] == "tier_1_excellent"

    # Manual check: 35800 at 4.49% for 60m -> r = 0.0449/12, P*r*(1+r)^60 / ((1+r)^60 - 1) = 667.26
    assert abs(quote["estimated_monthly_finance"] - 667.26) < 0.05
    assert quote["total_interest_paid"] > 4000.0
    assert quote["estimated_monthly_lease_36m"] > 250.0
    assert "Truth-in-Lending" in quote["truth_in_lending_disclosure"]


def test_calculate_financing_invalid_loan_term_and_unknown_vehicle() -> None:
    bad_term = calculate_financing_and_lease_options(
        vehicle_identifier="STK-2026-001",
        loan_term_months=45,
    )
    assert bad_term["status"] == "error"
    assert bad_term["error_code"] == "INVALID_FINANCING_INPUT"
    assert "24, 36, 48, 60, 72" in bad_term["recovery_suggestion"]

    missing_car = calculate_financing_and_lease_options(
        vehicle_identifier="UNKNOWN-STOCK-999",
        loan_term_months=60,
    )
    assert missing_car["status"] == "error"
    assert missing_car["error_code"] == "VEHICLE_NOT_FOUND"


def test_verify_dealership_floor_margin_above_and_below_floor() -> None:
    # STK-2026-001: invoice_price = 44390.0, min_margin = 500.0 -> minimum_allowed_selling_price = 44890.0
    compliant = verify_dealership_floor_margin(
        vehicle_identifier="STK-2026-001",
        proposed_selling_price=46000.0,
        rebates_total=1250.0,
    )
    assert compliant["status"] == "success"
    assert compliant["floor_margin_compliant"] is True
    assert compliant["requires_hitl_manager_approval"] is False
    assert compliant["dealer_gross_margin_dollars"] == 1610.0
    assert compliant["minimum_allowed_selling_price"] == 44890.0
    assert compliant["effective_customer_price_after_rebates"] == 44750.0

    below_floor = verify_dealership_floor_margin(
        vehicle_identifier="STK-2026-001",
        proposed_selling_price=44500.0,
        rebates_total=1250.0,
    )
    assert below_floor["status"] == "success"
    assert below_floor["floor_margin_compliant"] is False
    assert below_floor["requires_hitl_manager_approval"] is True
    assert below_floor["dealer_gross_margin_dollars"] == 110.0
    assert "MANAGER APPROVAL REQUIRED" in below_floor["policy_guidance"]


def test_generate_customer_offer_sheet_and_pii_redaction() -> None:
    sheet = generate_customer_offer_sheet(
        customer_name="John Doe (SSN: 123-45-6789, john.doe@example.com)",
        best_match_vin="STK-2026-001",
        value_pick_vin="STK-2024-008",
        upgrade_pick_vin="STK-2026-002",
        down_payment=4000.0,
        trade_in_equity=5500.0,
        credit_score=750,
        loan_term_months=60,
        custom_notes="Call customer at (415) 555-0199 or email john.doe@example.com",
    )
    assert sheet["status"] == "success"
    assert len(sheet["offers"]) == 3
    assert sheet["offers"][0]["tier_label"] == "Best Overall Match"
    assert sheet["offers"][1]["tier_label"] == "Budget / Value Pick"
    assert sheet["offers"][2]["tier_label"] == "Premium / Upgrade Pick"

    # Verify PII was scrubbed from customer_name, customer_summary, and custom_notes
    assert "123-45-6789" not in sheet["customer_name"]
    assert "[REDACTED_SSN]" in sheet["customer_name"]
    assert "john.doe@example.com" not in (sheet["custom_notes"] or "")
    assert "555-0199" not in (sheet["custom_notes"] or "")
    assert "[REDACTED_PHONE]" in (sheet["custom_notes"] or "")


def test_generate_customer_offer_sheet_duplicate_and_unknown_vin_errors() -> None:
    dup = generate_customer_offer_sheet(
        customer_name="Buyer",
        best_match_vin="5TDKDRBH1TS100001",
        value_pick_vin="STK-2026-001",  # Resolves to the exact same VIN!
        upgrade_pick_vin="STK-2026-002",
    )
    assert dup["status"] == "error"
    assert dup["error_code"] == "DUPLICATE_OFFER_VEHICLES"

    unknown = generate_customer_offer_sheet(
        customer_name="Buyer",
        best_match_vin="STK-2026-001",
        value_pick_vin="STK-NON-EXISTENT",
        upgrade_pick_vin="STK-2026-002",
    )
    assert unknown["status"] == "error"
    assert unknown["error_code"] == "VEHICLE_NOT_FOUND"
