"""Unit tests for `app/tools/inventory_tools.py` with guided error recovery."""

from __future__ import annotations

from app.observability.logger import clear_captured_json_logs, get_captured_json_logs
from app.observability.tracing import clear_recorded_spans, get_recorded_spans
from app.tools.inventory_tools import (
    appraise_trade_in_vehicle,
    fetch_active_manufacturer_rebates,
    search_dealership_inventory,
)


def test_search_dealership_inventory_happy_path_and_telemetry() -> None:
    clear_captured_json_logs()
    clear_recorded_spans()

    result = search_dealership_inventory(
        body_style="suv",
        max_price=48000.0,
        min_seating=7,
        fuel_type="hybrid",
        limit=5,
    )
    assert result["status"] == "success"
    assert result["matched_count"] >= 2
    for vehicle in result["vehicles"]:
        assert vehicle["body_style"] == "suv"
        assert vehicle["selling_price"] <= 48000.0
        assert vehicle["seating_capacity"] >= 7
        assert vehicle["fuel_type"] == "hybrid"

    logs = get_captured_json_logs()
    events = [entry.get("event_type") for entry in logs]
    assert "INTENT" in events
    assert "OUTCOME" in events

    spans = get_recorded_spans()
    assert any("search_dealership_inventory" in s.name for s in spans)


def test_search_dealership_inventory_zero_matches_returns_guided_recovery() -> None:
    result = search_dealership_inventory(
        body_style="suv",
        max_price=18000.0,
        min_seating=7,
    )
    assert result["status"] == "error"
    assert result["error_code"] == "ZERO_INVENTORY_MATCHES"
    assert "Highlander" in result["recovery_suggestion"]
    assert "max_price" in result["recovery_suggestion"]


def test_search_dealership_inventory_invalid_input_returns_guided_error() -> None:
    result = search_dealership_inventory(
        body_style="convertible",
        max_price=-5000.0,
    )
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_INVENTORY_SEARCH_INPUT"
    assert "body_style" in result["recovery_suggestion"]


def test_appraise_trade_in_vehicle_math_and_net_equity() -> None:
    result = appraise_trade_in_vehicle(
        year=2021,
        make="Honda",
        model="CR-V",
        mileage=45000,
        condition="good",
        existing_loan_payoff=6000.0,
    )
    assert result["status"] == "success"
    assert result["estimated_allowance"] > 15000.0
    assert result["existing_loan_payoff"] == 6000.0
    assert (
        round(result["estimated_allowance"] - 6000.0, 2)
        == result["net_trade_in_equity"]
    )
    assert result["positive_equity"] is True
    assert result["eligible_for_trade_in_rebate"] is True

    rough_result = appraise_trade_in_vehicle(
        year=2021,
        make="Honda",
        model="CR-V",
        mileage=45000,
        condition="rough",
        existing_loan_payoff=0.0,
    )
    assert rough_result["status"] == "success"
    assert rough_result["condition_multiplier"] == 0.74
    assert rough_result["estimated_allowance"] < result["estimated_allowance"]


def test_appraise_trade_in_vehicle_invalid_input_guided_recovery() -> None:
    result = appraise_trade_in_vehicle(
        year=1985,
        make="Honda",
        model="Civic",
        mileage=500000,
        condition="good",
    )
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_TRADE_IN_INPUT"
    assert "2000 and 2027" in result["recovery_suggestion"]


def test_fetch_active_manufacturer_rebates_and_unknown_vin_recovery() -> None:
    # Mazda CX-90 PHEV (STK-2026-003) is PHEV + 68 days on lot -> qualifies for Clean EV ($3500) + Aged Lot ($1000) + Loyalty ($750) + First-Time Buyer ($500)
    rebates = fetch_active_manufacturer_rebates(
        vehicle_identifier="STK-2026-003",
        customer_qualifiers=["loyalty", "first_time_buyer"],
    )
    assert rebates["status"] == "success"
    assert rebates["rebate_count"] == 4
    assert rebates["total_rebate_amount"] == 5750.0

    unknown = fetch_active_manufacturer_rebates(
        vehicle_identifier="VIN-DOES-NOT-EXIST-999",
    )
    assert unknown["status"] == "error"
    assert unknown["error_code"] == "VEHICLE_NOT_FOUND"
    assert "search_dealership_inventory" in unknown["recovery_suggestion"]
