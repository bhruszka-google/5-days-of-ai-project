"""Strict Pydantic Input and Output JSON Schemas for AutoDeal Copilot Tools.

Enforces `extra="forbid"`, numeric bounds, and domain invariants across all
dealership inventory, trade-in, rebate, financing, floor-margin, manager
discount, and Top-3 Offer presentation tools.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_BODY_STYLES = ("sedan", "suv", "crossover", "truck", "ev")
ALLOWED_FUEL_TYPES = ("gas", "hybrid", "phev", "ev")
ALLOWED_VEHICLE_CONDITIONS = ("new", "cpo")
ALLOWED_TRADE_IN_CONDITIONS = ("excellent", "good", "fair", "rough", "poor")
ALLOWED_LOAN_TERMS = (24, 36, 48, 60, 72)

OfferTierLabel = Literal[
    "Best Overall Match",
    "Budget / Value Pick",
    "Premium / Upgrade Pick",
]


class ToolErrorResponse(BaseModel):
    """Structured guided error payload returned to the LLM when a tool call fails."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["error"] = "error"
    error_code: str = Field(
        ...,
        min_length=3,
        description="Machine-readable error classification code.",
    )
    message: str = Field(
        ...,
        min_length=5,
        description="Human- and LLM-readable explanation of what failed.",
    )
    recovery_suggestion: str = Field(
        ...,
        min_length=10,
        description="Actionable guidance telling the LLM how to adjust parameters and retry.",
    )


def build_guided_error_response(
    error_code: str,
    message: str,
    recovery_suggestion: str,
) -> Dict[str, str]:
    """Constructs and validates a `ToolErrorResponse` dictionary."""
    return ToolErrorResponse(
        error_code=error_code,
        message=message,
        recovery_suggestion=recovery_suggestion,
    ).model_dump()


class InventorySearchInput(BaseModel):
    """Strict input schema for `search_dealership_inventory`."""

    model_config = ConfigDict(extra="forbid")

    body_style: Optional[str] = Field(
        default=None,
        description="Optional vehicle body style ('sedan', 'suv', 'crossover', 'truck', 'ev').",
    )
    max_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional maximum advertised selling price in USD (must be > 0).",
    )
    min_seating: Optional[int] = Field(
        default=None,
        ge=2,
        le=8,
        description="Optional minimum seating capacity required (2 to 8 passengers).",
    )
    fuel_type: Optional[str] = Field(
        default=None,
        description="Optional fuel/propulsion filter ('gas', 'hybrid', 'phev', 'ev').",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Optional inventory condition ('new' or 'cpo').",
    )
    feature_keyword: Optional[str] = Field(
        default=None,
        description="Optional equipment or model keyword (e.g., '3-row', 'awd', 'carplay').",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching vehicles to return (1 to 20, default 5).",
    )

    @field_validator("body_style")
    @classmethod
    def validate_body_style(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized or normalized == "any":
            return None
        if normalized not in ALLOWED_BODY_STYLES:
            raise ValueError(
                f"Unsupported body_style '{value}'. Allowed values: {list(ALLOWED_BODY_STYLES)}."
            )
        return normalized

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized or normalized == "any":
            return None
        if normalized not in ALLOWED_FUEL_TYPES:
            raise ValueError(
                f"Unsupported fuel_type '{value}'. Allowed values: {list(ALLOWED_FUEL_TYPES)}."
            )
        return normalized

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized or normalized == "any":
            return None
        if normalized not in ALLOWED_VEHICLE_CONDITIONS:
            raise ValueError(
                f"Unsupported condition '{value}'. Allowed values: {list(ALLOWED_VEHICLE_CONDITIONS)}."
            )
        return normalized


class TradeInAppraisalInput(BaseModel):
    """Strict input schema for `appraise_trade_in_vehicle`."""

    model_config = ConfigDict(extra="forbid")

    year: int = Field(
        ...,
        ge=2000,
        le=2027,
        description="Model year of the customer's trade-in vehicle (2000 to 2027).",
    )
    make: str = Field(
        ...,
        min_length=2,
        description="Vehicle manufacturer brand (e.g., 'Honda', 'Toyota').",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Vehicle model name (e.g., 'CR-V', 'RAV4').",
    )
    mileage: int = Field(
        ...,
        ge=0,
        le=350000,
        description="Current odometer reading in miles (0 to 350,000).",
    )
    condition: Literal["excellent", "good", "fair", "rough", "poor"] = Field(
        default="good",
        description="Physical and mechanical condition ('excellent', 'good', 'fair', 'rough').",
    )
    existing_loan_payoff: float = Field(
        default=0.0,
        ge=0.0,
        description="Remaining loan payoff balance on the trade-in vehicle in USD (>= 0).",
    )

    @field_validator("condition", mode="before")
    @classmethod
    def normalize_condition(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class RebateLookupInput(BaseModel):
    """Strict input schema for `fetch_active_manufacturer_rebates`."""

    model_config = ConfigDict(extra="forbid")

    vehicle_identifier: str = Field(
        ...,
        min_length=2,
        description="17-char VIN, stock number (e.g., 'STK-2026-001'), or vehicle display title.",
    )
    customer_qualifiers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional customer eligibility tags such as "
            "['loyalty', 'trade_in', 'military', 'first_time_buyer', 'clean_vehicle']."
        ),
    )


class FinancingCalculationInput(BaseModel):
    """Strict input schema for `calculate_financing_and_lease_options`."""

    model_config = ConfigDict(extra="forbid")

    vehicle_identifier: str = Field(
        ...,
        min_length=2,
        description="17-char VIN, stock number, or display title of the target vehicle.",
    )
    down_payment: float = Field(
        default=0.0,
        ge=0.0,
        description="Customer cash down payment in USD (>= 0).",
    )
    trade_in_equity: float = Field(
        default=0.0,
        description="Net trade-in credit (or negative equity rollover) in USD.",
    )
    credit_score: int = Field(
        default=720,
        ge=300,
        le=850,
        description="Customer FICO credit score (300 to 850).",
    )
    loan_term_months: int = Field(
        default=60,
        description="Amortized auto loan term in months (must be one of 24, 36, 48, 60, 72).",
    )
    include_rebates: bool = Field(
        default=True,
        description="Whether to automatically deduct stackable manufacturer rebates.",
    )

    @field_validator("loan_term_months")
    @classmethod
    def validate_loan_term(cls, value: int) -> int:
        if value not in ALLOWED_LOAN_TERMS:
            raise ValueError(
                f"Invalid loan_term_months={value}. Allowed loan terms are {list(ALLOWED_LOAN_TERMS)} months."
            )
        return value


class MarginCheckInput(BaseModel):
    """Strict input schema for `verify_dealership_floor_margin`."""

    model_config = ConfigDict(extra="forbid")

    vehicle_identifier: str = Field(
        ...,
        min_length=2,
        description="17-char VIN, stock number, or display title of the target vehicle.",
    )
    proposed_selling_price: float = Field(
        ...,
        gt=0.0,
        description="Proposed dealership selling price before manufacturer rebates in USD (> 0).",
    )
    rebates_total: float = Field(
        default=0.0,
        ge=0.0,
        description="Total manufacturer rebates applied in USD (does not reduce dealer gross margin).",
    )


class ManagerDiscountInput(BaseModel):
    """Strict input schema for `apply_manager_special_discount`."""

    model_config = ConfigDict(extra="forbid")

    vehicle_identifier: str = Field(
        ...,
        min_length=2,
        description="17-char VIN, stock number, or display title of the target vehicle.",
    )
    requested_discount_amount: float = Field(
        ...,
        gt=0.0,
        description="Dollar discount requested below advertised selling price (> 0).",
    )
    reason: str = Field(
        ...,
        min_length=3,
        description="Business justification for requesting the manager special discount.",
    )
    manager_approval_token: Optional[str] = Field(
        default=None,
        description="Explicit Sales Manager HITL approval token when discount exceeds threshold.",
    )


class VehicleOfferCard(BaseModel):
    """Strict schema for a single ranked vehicle offer card in the Top-3 presentation."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(
        ...,
        ge=1,
        le=3,
        description="Offer card rank (1 = Best Overall Match, 2 = Budget / Value Pick, 3 = Premium / Upgrade Pick).",
    )
    tier_label: OfferTierLabel = Field(
        ...,
        description="Canonical tier label for the offer card.",
    )
    vin: str = Field(..., min_length=17, max_length=17, description="17-character VIN.")
    stock_number: str = Field(..., min_length=5, description="Dealership stock number.")
    vehicle_title: str = Field(
        ..., min_length=4, description="Formatted Year Make Model Trim."
    )
    msrp: float = Field(
        ..., gt=0.0, description="Manufacturer Suggested Retail Price in USD."
    )
    selling_price: float = Field(
        ..., gt=0.0, description="Dealership selling price in USD."
    )
    rebates_applied: float = Field(
        ..., ge=0.0, description="Total manufacturer rebates deducted in USD."
    )
    net_cap_cost: float = Field(
        ..., ge=0.0, description="Net capitalized / financed amount in USD."
    )
    estimated_monthly_finance: float = Field(
        ...,
        ge=0.0,
        description="Estimated monthly amortized loan payment in USD.",
    )
    apr_percent: float = Field(
        ..., ge=0.0, description="Annual Percentage Rate (APR %) applied."
    )
    loan_term_months: int = Field(
        ...,
        description="Loan term in months (24, 36, 48, 60, or 72).",
    )
    estimated_monthly_lease_36m: float = Field(
        ...,
        ge=0.0,
        description="Estimated 36-month monthly lease payment in USD.",
    )
    why_it_fits: str = Field(
        ...,
        min_length=5,
        description="Tailored explanation of why this vehicle fits the customer's needs.",
    )
    floor_margin_compliant: bool = Field(
        ...,
        description="True if selling_price meets or exceeds invoice_price + minimum required dealer margin.",
    )

    @field_validator("loan_term_months")
    @classmethod
    def validate_card_loan_term(cls, value: int) -> int:
        if value not in ALLOWED_LOAN_TERMS:
            raise ValueError(
                f"Invalid loan_term_months={value}. Must be one of {list(ALLOWED_LOAN_TERMS)}."
            )
        return value


class OfferSheetInput(BaseModel):
    """Strict input schema for `generate_customer_offer_sheet`."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str = Field(
        ...,
        min_length=1,
        description="Customer name or buyer label (automatically scrubbed for PII).",
    )
    best_match_vin: str = Field(
        ...,
        min_length=2,
        description="VIN or stock number for Rank 1 (Best Overall Match).",
    )
    value_pick_vin: str = Field(
        ...,
        min_length=2,
        description="VIN or stock number for Rank 2 (Budget / Value Pick).",
    )
    upgrade_pick_vin: str = Field(
        ...,
        min_length=2,
        description="VIN or stock number for Rank 3 (Premium / Upgrade Pick).",
    )
    down_payment: float = Field(
        default=0.0,
        ge=0.0,
        description="Customer cash down payment in USD (>= 0).",
    )
    trade_in_equity: float = Field(
        default=0.0,
        description="Net trade-in equity applied to the transaction in USD.",
    )
    credit_score: int = Field(
        default=720,
        ge=300,
        le=850,
        description="Customer FICO score (300 to 850).",
    )
    loan_term_months: int = Field(
        default=60,
        description="Loan term in months (24, 36, 48, 60, or 72).",
    )
    custom_notes: Optional[str] = Field(
        default=None,
        description="Optional desk salesperson notes (automatically scrubbed for PII).",
    )

    @field_validator("loan_term_months")
    @classmethod
    def validate_offer_loan_term(cls, value: int) -> int:
        if value not in ALLOWED_LOAN_TERMS:
            raise ValueError(
                f"Invalid loan_term_months={value}. Allowed loan terms are {list(ALLOWED_LOAN_TERMS)} months."
            )
        return value

    @model_validator(mode="after")
    def validate_distinct_vehicle_picks(self) -> "OfferSheetInput":
        picks = [
            self.best_match_vin.strip().upper(),
            self.value_pick_vin.strip().upper(),
            self.upgrade_pick_vin.strip().upper(),
        ]
        if len(set(picks)) != 3:
            raise ValueError(
                "All three offer picks (best_match_vin, value_pick_vin, upgrade_pick_vin) "
                "must reference distinct vehicles."
            )
        return self


class TopThreeOffersResponse(BaseModel):
    """Strict output schema for the compiled Top-3 Customer Offer Sheet."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    customer_name: str = Field(..., min_length=1)
    customer_summary: str = Field(..., min_length=10)
    offers: List[VehicleOfferCard] = Field(..., min_length=1, max_length=3)
    compliance_disclaimer: str = Field(..., min_length=20)
    custom_notes: Optional[str] = None
