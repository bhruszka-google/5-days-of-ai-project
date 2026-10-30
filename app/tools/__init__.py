"""Dealership Inventory, Trade-In, Rebate, Financing, HITL Discount, and Top-3 Offer Tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from app.guardrails.hitl import (
        apply_manager_special_discount,
        build_manager_discount_tool,
        requires_sales_manager_approval,
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
    "apply_manager_special_discount",
    "appraise_trade_in_vehicle",
    "build_guided_error_response",
    "build_manager_discount_tool",
    "calculate_financing_and_lease_options",
    "fetch_active_manufacturer_rebates",
    "generate_customer_offer_sheet",
    "requires_sales_manager_approval",
    "search_dealership_inventory",
    "verify_dealership_floor_margin",
]

_HITL_TOOL_EXPORTS = {
    "apply_manager_special_discount",
    "build_manager_discount_tool",
    "requires_sales_manager_approval",
}


def __getattr__(name: str) -> Any:
    """Lazily resolves HITL discount tool symbols to prevent circular imports."""
    if name in _HITL_TOOL_EXPORTS:
        from app.guardrails import hitl

        return getattr(hitl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
