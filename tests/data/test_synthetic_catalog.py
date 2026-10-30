"""Unit tests for app/data/synthetic_catalog.py."""

from __future__ import annotations

import pytest
from app.data.synthetic_catalog import (
    CREDIT_TIER_RATES,
    SYNTHETIC_REBATES,
    SYNTHETIC_VEHICLES,
    VehicleRecord,
    estimate_trade_in_value,
    get_all_vehicles,
    get_applicable_rebates,
    get_credit_tier_rate,
    get_vehicle_by_id_or_vin,
)
from pydantic import ValidationError


class TestSyntheticCatalog:
    """Tests for synthetic dealership vehicles, rebates, credit tiers, and trade-ins."""

    def test_catalogue_contains_at_least_20_diverse_vehicles(self) -> None:
        vehicles = get_all_vehicles()
        assert len(vehicles) >= 20

        body_styles = {v.body_style for v in vehicles}
        fuel_types = {v.fuel_type for v in vehicles}
        conditions = {v.condition for v in vehicles}

        assert {"sedan", "suv", "crossover", "truck", "ev"}.issubset(body_styles)
        assert {"gas", "hybrid", "phev", "ev"}.issubset(fuel_types)
        assert {"new", "cpo"}.issubset(conditions)

    def test_all_vehicles_respect_price_hierarchy_and_vin_format(self) -> None:
        vins = set()
        stock_numbers = set()
        for v in SYNTHETIC_VEHICLES:
            assert len(v.vin) == 17
            assert v.msrp >= v.selling_price > v.invoice_price > 0
            assert 4.0 <= v.safety_rating <= 5.0
            assert len(v.key_features) >= 3
            vins.add(v.vin)
            stock_numbers.add(v.stock_number)

        assert len(vins) == len(SYNTHETIC_VEHICLES)
        assert len(stock_numbers) == len(SYNTHETIC_VEHICLES)

    def test_vehicle_record_rejects_invalid_price_hierarchy_or_extra_fields(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            VehicleRecord(
                vin="5TDKDRBH1TS999999",
                stock_number="STK-ERR-001",
                year=2026,
                make="Toyota",
                model="Highlander",
                trim="XLE",
                body_style="suv",
                fuel_type="hybrid",
                drivetrain="AWD",
                condition="new",
                mileage=10,
                msrp=45000.0,
                invoice_price=44500.0,
                selling_price=44000.0,  # invalid: selling_price < invoice_price
                seating_capacity=7,
                mpg_combined=35,
                safety_rating=5.0,
                key_features=["3-row seating"],
                days_on_lot=10,
            )

    def test_lookup_vehicle_by_vin_stock_or_display_name(self) -> None:
        by_vin = get_vehicle_by_id_or_vin("5TDKDRBH1TS100001")
        assert by_vin is not None
        assert by_vin.make == "Toyota"
        assert by_vin.model == "Highlander Hybrid"

        by_stock = get_vehicle_by_id_or_vin("STK-2026-001")
        assert by_stock is not None
        assert by_stock.vin == "5TDKDRBH1TS100001"

        by_name = get_vehicle_by_id_or_vin("2026 Toyota Highlander Hybrid XLE")
        assert by_name is not None
        assert by_name.vin == "5TDKDRBH1TS100001"

        assert get_vehicle_by_id_or_vin("NON-EXISTENT-VEHICLE") is None

    def test_applicable_rebates_matching(self) -> None:
        assert len(SYNTHETIC_REBATES) >= 4
        cx90_phev = get_vehicle_by_id_or_vin("STK-2026-003")
        assert cx90_phev is not None

        rebates = get_applicable_rebates(
            cx90_phev, customer_qualifiers=["trade_in", "military_or_grad"]
        )
        rebate_ids = {r.rebate_id for r in rebates}
        # CX-90 PHEV has days_on_lot=68 and fuel_type='phev'
        assert "REB-CLEAN-EV-2026" in rebate_ids
        assert "REB-AGED-LOT-60D" in rebate_ids
        assert "REB-LOYALTY-TRADE" in rebate_ids
        assert "REB-MILITARY-GRAD" in rebate_ids

    def test_credit_tier_lookup_by_score_and_tier_id(self) -> None:
        assert set(CREDIT_TIER_RATES.keys()) == {
            "tier_1_excellent",
            "tier_2_good",
            "tier_3_fair",
            "tier_4_subprime",
        }
        assert get_credit_tier_rate(credit_score=760).tier_id == "tier_1_excellent"
        assert get_credit_tier_rate(credit_score=700).tier_id == "tier_2_good"
        assert get_credit_tier_rate(credit_score=650).tier_id == "tier_3_fair"
        assert get_credit_tier_rate(credit_score=580).tier_id == "tier_4_subprime"
        assert get_credit_tier_rate(credit_tier="tier_2_good").tier_id == "tier_2_good"

    def test_estimate_trade_in_value_calculation(self) -> None:
        good_crv = estimate_trade_in_value(
            year=2021, make="Honda", model="CR-V", mileage=45000, condition="good"
        )
        poor_crv = estimate_trade_in_value(
            year=2021, make="Honda", model="CR-V", mileage=95000, condition="poor"
        )
        assert good_crv["estimated_allowance"] > poor_crv["estimated_allowance"]
        assert good_crv["estimated_allowance"] > 12000.0
