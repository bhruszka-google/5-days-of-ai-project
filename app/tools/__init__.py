"""Dealership Inventory, Trade-In, Rebate, Financing, and Top-3 Offer Tools."""

from __future__ import annotations

from app.tools.financing_tools import (
    calculate_financing_and_lease_options,
    generate_customer_offer_sheet,
    verify_dealership_floor_margin,
)
from app.tools.inventory_tools import (
    appraise_trade_in_vehicle,
    fetch_active_manufacturer_rebates,
    search_dealership_inventory,
)
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

__all__ = [
    "FinancingCalculationInput",
    "InventorySearchInput",
    "ManagerDiscountInput",
    "MarginCheckInput",
    "OfferSheetInput",
    "RebateLookupInput",
    "ToolErrorResponse",
    "TopThreeOffersResponse",
    "TradeInAppraisalInput",
    "VehicleOfferCard",
    "appraise_trade_in_vehicle",
    "build_guided_error_response",
    "calculate_financing_and_lease_options",
    "fetch_active_manufacturer_rebates",
    "generate_customer_offer_sheet",
    "search_dealership_inventory",
    "verify_dealership_floor_margin",
]
