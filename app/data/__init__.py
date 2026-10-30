"""Synthetic dealership vehicle catalogue, rebates, credit tiers, and trade-in data."""

from app.data.synthetic_catalog import (
    CREDIT_TIER_RATES,
    SYNTHETIC_REBATES,
    SYNTHETIC_VEHICLES,
    TRADE_IN_BASE_VALUES,
    CreditTierRate,
    ManufacturerRebate,
    TradeInReference,
    VehicleRecord,
    estimate_trade_in_value,
    get_all_vehicles,
    get_applicable_rebates,
    get_credit_tier_rate,
    get_vehicle_by_id_or_vin,
)

__all__ = [
    "CREDIT_TIER_RATES",
    "SYNTHETIC_REBATES",
    "SYNTHETIC_VEHICLES",
    "TRADE_IN_BASE_VALUES",
    "CreditTierRate",
    "ManufacturerRebate",
    "TradeInReference",
    "VehicleRecord",
    "estimate_trade_in_value",
    "get_all_vehicles",
    "get_applicable_rebates",
    "get_credit_tier_rate",
    "get_vehicle_by_id_or_vin",
]
