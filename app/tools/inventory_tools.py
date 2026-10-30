"""Dealership Inventory Search, Trade-In Appraisal, and Manufacturer Rebate Tools.

Provides ADK-compatible tool functions for the `inventory_specialist` agent with:
- Strict Pydantic schema validation (`app/tools/schemas.py`)
- Guided error recovery (`ToolErrorResponse`) so invalid LLM parameters or
  zero-match queries never raise unhandled exceptions
- OpenTelemetry distributed trace spans (`@traced_operation`)
- Structured JSON `INTENT` and `OUTCOME` logging (`log_intent`, `log_outcome`)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.data.synthetic_catalog import (
    estimate_trade_in_value,
    get_all_vehicles,
    get_applicable_rebates,
    get_vehicle_by_id_or_vin,
)
from app.memory.session_store import DEFAULT_SESSION_STORE
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.tools.schemas import (
    ALLOWED_BODY_STYLES,
    ALLOWED_FUEL_TYPES,
    ALLOWED_TRADE_IN_CONDITIONS,
    ALLOWED_VEHICLE_CONDITIONS,
    InventorySearchInput,
    RebateLookupInput,
    TradeInAppraisalInput,
    build_guided_error_response,
)


def _normalize_customer_qualifiers(
    raw_qualifiers: Optional[List[str]],
) -> List[str]:
    """Normalizes and expands common customer rebate qualifier aliases."""
    if not raw_qualifiers:
        return []
    expanded: set[str] = set()
    for item in raw_qualifiers:
        if not isinstance(item, str):
            continue
        token = item.strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            continue
        expanded.add(token)
        if token in {"loyalty", "owner_loyalty", "trade", "trade_in", "conquest"}:
            expanded.add("trade_in")
        if token in {
            "military",
            "college_grad",
            "grad",
            "first_time_buyer",
            "military_or_grad",
        }:
            expanded.add("military_or_grad")
    return sorted(expanded)


@traced_operation("autodeal.tools.search_dealership_inventory")
def search_dealership_inventory(
    body_style: Optional[str] = None,
    max_price: Optional[float] = None,
    min_seating: Optional[int] = None,
    fuel_type: Optional[str] = None,
    condition: Optional[str] = None,
    feature_keyword: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Searches live dealership lot inventory using multi-attribute filters.

    Queries the persistent vehicle catalog by body style, maximum selling price,
    minimum seating capacity, fuel/propulsion type, condition (`new` or `cpo`),
    and equipment keywords. If no vehicles match or if filter arguments are
    invalid, returns a structured guided recovery payload advising the agent how
    to broaden the search criteria.

    Args:
        body_style: Optional vehicle body classification (`'sedan'`, `'suv'`,
            `'crossover'`, `'truck'`, `'ev'`).
        max_price: Optional maximum advertised selling price in USD (must be > 0).
        min_seating: Optional minimum passenger seating capacity (2 to 8).
        fuel_type: Optional propulsion type (`'gas'`, `'hybrid'`, `'phev'`, `'ev'`).
        condition: Optional inventory condition (`'new'` or `'cpo'`).
        feature_keyword: Optional keyword matched against vehicle title or key
            features (e.g., `'3-row'`, `'captain chairs'`, `'carplay'`).
        limit: Maximum number of matching vehicles to return (1 to 20, default 5).

    Returns:
        Dict[str, Any]: On success, returns `status="success"`, `matched_count`,
        `applied_filters`, and a `vehicles` list containing full vehicle records
        and active stackable rebates. On validation or zero-match failure,
        returns `status="error"`, `error_code`, `message`, and
        `recovery_suggestion`.

    Raises:
        None: All validation and domain errors are caught and converted into
        structured `ToolErrorResponse` dictionaries for LLM self-correction.

    Example:
        >>> result = search_dealership_inventory(
        ...     body_style="suv",
        ...     max_price=48000.0,
        ...     min_seating=7,
        ...     fuel_type="hybrid",
        ... )
        >>> result["status"]
        'success'
    """
    raw_args = {
        "body_style": body_style,
        "max_price": max_price,
        "min_seating": min_seating,
        "fuel_type": fuel_type,
        "condition": condition,
        "feature_keyword": feature_keyword,
        "limit": limit,
    }
    log_intent(
        action_name="search_dealership_inventory",
        intended_arguments=raw_args,
        agent_name="inventory_specialist",
    )

    try:
        validated = InventorySearchInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_INVENTORY_SEARCH_INPUT",
            message=f"Inventory search filter validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                f"Ensure body_style is one of {list(ALLOWED_BODY_STYLES)}, "
                f"fuel_type is one of {list(ALLOWED_FUEL_TYPES)}, "
                f"condition is one of {list(ALLOWED_VEHICLE_CONDITIONS)}, "
                "max_price > 0, min_seating is between 2 and 8, and limit is between 1 and 20."
            ),
        )
        log_outcome(
            action_name="search_dealership_inventory",
            status="error",
            actual_outcome=error_payload,
            agent_name="inventory_specialist",
        )
        return error_payload

    matches = DEFAULT_SESSION_STORE.query_vehicles(
        body_style=validated.body_style,
        max_price=validated.max_price,
        min_seating=validated.min_seating,
        fuel_type=validated.fuel_type,
        condition=validated.condition,
        feature_keyword=validated.feature_keyword,
        limit=validated.limit,
    )

    if not matches:
        all_vehicles = get_all_vehicles()
        same_style = [
            v
            for v in all_vehicles
            if validated.body_style is None or v.body_style == validated.body_style
        ]
        pool = same_style if same_style else all_vehicles
        cheapest = min(pool, key=lambda v: v.selling_price)
        max_seats_available = max(v.seating_capacity for v in pool)

        error_payload = build_guided_error_response(
            error_code="ZERO_INVENTORY_MATCHES",
            message=(
                "No vehicles in dealership inventory matched all requested filters "
                f"(body_style={validated.body_style}, max_price={validated.max_price}, "
                f"min_seating={validated.min_seating}, fuel_type={validated.fuel_type}, "
                f"condition={validated.condition}, feature_keyword={validated.feature_keyword})."
            ),
            recovery_suggestion=(
                f"Retry `search_dealership_inventory` with broader criteria: "
                f"the lowest-priced vehicle in this category is the {cheapest.display_name} "
                f"({cheapest.vin}) at ${cheapest.selling_price:,.0f} ({cheapest.condition.upper()}, "
                f"{cheapest.fuel_type}, {cheapest.seating_capacity} seats), and maximum seating in "
                f"this segment is {max_seats_available}. Increase `max_price` to at least "
                f"${cheapest.selling_price:,.0f}, omit `feature_keyword`, or include `'cpo'` / `'crossover'` options."
            ),
        )
        log_outcome(
            action_name="search_dealership_inventory",
            status="error",
            actual_outcome=error_payload,
            agent_name="inventory_specialist",
        )
        return error_payload

    serialized_vehicles: List[Dict[str, Any]] = []
    for vehicle in matches:
        rebates = get_applicable_rebates(vehicle)
        rebate_sum = sum(r.amount for r in rebates)
        serialized_vehicles.append(
            {
                "vin": vehicle.vin,
                "stock_number": vehicle.stock_number,
                "vehicle_title": vehicle.display_name,
                "year": vehicle.year,
                "make": vehicle.make,
                "model": vehicle.model,
                "trim": vehicle.trim,
                "body_style": vehicle.body_style,
                "fuel_type": vehicle.fuel_type,
                "drivetrain": vehicle.drivetrain,
                "condition": vehicle.condition,
                "mileage": vehicle.mileage,
                "msrp": vehicle.msrp,
                "invoice_price": vehicle.invoice_price,
                "selling_price": vehicle.selling_price,
                "unconditional_rebates_total": rebate_sum,
                "net_price_after_rebates": round(vehicle.selling_price - rebate_sum, 2),
                "seating_capacity": vehicle.seating_capacity,
                "mpg_combined": vehicle.mpg_combined,
                "safety_rating": vehicle.safety_rating,
                "key_features": list(vehicle.key_features),
                "days_on_lot": vehicle.days_on_lot,
            }
        )

    response = {
        "status": "success",
        "matched_count": len(serialized_vehicles),
        "applied_filters": validated.model_dump(exclude_none=True),
        "vehicles": serialized_vehicles,
    }
    log_outcome(
        action_name="search_dealership_inventory",
        status="success",
        actual_outcome={
            "matched_count": len(serialized_vehicles),
            "vins": [v["vin"] for v in serialized_vehicles],
        },
        agent_name="inventory_specialist",
    )
    return response


@traced_operation("autodeal.tools.appraise_trade_in_vehicle")
def appraise_trade_in_vehicle(
    year: int,
    make: str,
    model: str,
    mileage: int,
    condition: str = "good",
    existing_loan_payoff: float = 0.0,
) -> Dict[str, Any]:
    """Calculates a wholesale trade-in appraisal and net equity credit.

    Evaluates a customer's existing vehicle using model year depreciation curves,
    odometer mileage adjustments (12,000 miles/year benchmark), segment demand
    bonuses, and cosmetic/mechanical condition multipliers, then subtracts any
    existing loan payoff balance to determine net trade-in equity.

    Args:
        year: Model year of the trade-in vehicle (2000 to 2027).
        make: Vehicle manufacturer name (e.g., `'Honda'`, `'Toyota'`).
        model: Vehicle model name (e.g., `'CR-V'`, `'RAV4'`).
        mileage: Current odometer reading in miles (0 to 350,000).
        condition: Physical condition tier (`'excellent'`, `'good'`, `'fair'`,
            or `'rough'`). Defaults to `'good'`.
        existing_loan_payoff: Remaining lien payoff amount in USD (>= 0.0).

    Returns:
        Dict[str, Any]: On success, returns `status="success"`,
        `estimated_allowance`, `existing_loan_payoff`, `net_trade_in_equity`,
        `positive_equity`, `eligible_for_trade_in_rebate`, and valuation breakdown
        fields. On validation failure, returns a guided `ToolErrorResponse` dict.

    Raises:
        None: Input validation errors are caught and returned as guided error dicts.

    Example:
        >>> appraisal = appraise_trade_in_vehicle(
        ...     year=2021,
        ...     make="Honda",
        ...     model="CR-V",
        ...     mileage=45000,
        ...     condition="good",
        ...     existing_loan_payoff=5000.0,
        ... )
        >>> appraisal["status"]
        'success'
    """
    raw_args = {
        "year": year,
        "make": make,
        "model": model,
        "mileage": mileage,
        "condition": condition,
        "existing_loan_payoff": existing_loan_payoff,
    }
    log_intent(
        action_name="appraise_trade_in_vehicle",
        intended_arguments=raw_args,
        agent_name="inventory_specialist",
    )

    try:
        validated = TradeInAppraisalInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_TRADE_IN_INPUT",
            message=f"Trade-in appraisal validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                "Verify that `year` is between 2000 and 2027, `mileage` is between 0 and 350,000, "
                f"`condition` is one of {list(ALLOWED_TRADE_IN_CONDITIONS)}, "
                "`make` has at least 2 characters, and `existing_loan_payoff` >= 0."
            ),
        )
        log_outcome(
            action_name="appraise_trade_in_vehicle",
            status="error",
            actual_outcome=error_payload,
            agent_name="inventory_specialist",
        )
        return error_payload

    catalog_condition = (
        "poor" if validated.condition == "rough" else validated.condition
    )
    valuation = estimate_trade_in_value(
        year=validated.year,
        make=validated.make,
        model=validated.model,
        mileage=validated.mileage,
        condition=catalog_condition,
    )

    allowance = float(valuation["estimated_allowance"])
    net_equity = round(allowance - validated.existing_loan_payoff, 2)

    response = {
        "status": "success",
        "vehicle_summary": f"{validated.year} {valuation['make']} {valuation['model']}",
        "year": validated.year,
        "make": valuation["make"],
        "model": valuation["model"],
        "mileage": validated.mileage,
        "condition": validated.condition,
        "baseline_value": valuation["baseline_value"],
        "mileage_adjustment": valuation["mileage_adjustment"],
        "popular_segment_bonus": valuation["popular_segment_bonus"],
        "condition_multiplier": valuation["condition_multiplier"],
        "estimated_allowance": allowance,
        "existing_loan_payoff": round(validated.existing_loan_payoff, 2),
        "net_trade_in_equity": net_equity,
        "positive_equity": net_equity >= 0.0,
        "eligible_for_trade_in_rebate": True,
        "recommended_rebate_qualifier": "trade_in",
    }
    log_outcome(
        action_name="appraise_trade_in_vehicle",
        status="success",
        actual_outcome={
            "vehicle_summary": response["vehicle_summary"],
            "estimated_allowance": allowance,
            "net_trade_in_equity": net_equity,
        },
        agent_name="inventory_specialist",
    )
    return response


@traced_operation("autodeal.tools.fetch_active_manufacturer_rebates")
def fetch_active_manufacturer_rebates(
    vehicle_identifier: str,
    customer_qualifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Finds all stackable manufacturer and dealer rebates for a specific vehicle.

    Matches a vehicle by VIN, stock number, or display title and evaluates all
    active incentive programs (Clean EV/PHEV Conquest Cash, Eco-Hybrid Retail
    Bonus, 60+ Day Aged Inventory Clearance, Owner Loyalty/Trade-In Bonus, and
    Military/College Grad Appreciation).

    Args:
        vehicle_identifier: 17-character VIN, dealership stock number (e.g.,
            `'STK-2026-001'`), or vehicle display name.
        customer_qualifiers: Optional list of eligibility tags such as
            `['loyalty', 'trade_in', 'military', 'first_time_buyer']`.

    Returns:
        Dict[str, Any]: On success, returns `status="success"`, `vin`,
        `stock_number`, `vehicle_title`, `rebates` list, and
        `total_rebate_amount`. On invalid input or unknown vehicle identifier,
        returns a structured `ToolErrorResponse` dictionary with sample valid
        identifiers.

    Raises:
        None: Unknown identifiers and validation errors return guided error dicts.

    Example:
        >>> res = fetch_active_manufacturer_rebates(
        ...     vehicle_identifier="STK-2026-001",
        ...     customer_qualifiers=["loyalty"],
        ... )
        >>> res["status"]
        'success'
    """
    raw_args = {
        "vehicle_identifier": vehicle_identifier,
        "customer_qualifiers": customer_qualifiers,
    }
    log_intent(
        action_name="fetch_active_manufacturer_rebates",
        intended_arguments=raw_args,
        agent_name="inventory_specialist",
    )

    try:
        validated = RebateLookupInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_REBATE_LOOKUP_INPUT",
            message=f"Rebate lookup validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                "Pass a non-empty `vehicle_identifier` (17-character VIN, stock number like "
                "'STK-2026-001', or vehicle name) and an optional list of strings for `customer_qualifiers`."
            ),
        )
        log_outcome(
            action_name="fetch_active_manufacturer_rebates",
            status="error",
            actual_outcome=error_payload,
            agent_name="inventory_specialist",
        )
        return error_payload

    vehicle = get_vehicle_by_id_or_vin(validated.vehicle_identifier)
    if vehicle is None:
        sample_vehicles = get_all_vehicles()[:4]
        sample_hints = ", ".join(
            f"{v.stock_number} ({v.display_name}, VIN {v.vin})" for v in sample_vehicles
        )
        error_payload = build_guided_error_response(
            error_code="VEHICLE_NOT_FOUND",
            message=(
                f"Could not find any vehicle matching identifier '{validated.vehicle_identifier}' "
                "in dealership inventory."
            ),
            recovery_suggestion=(
                "Call `search_dealership_inventory` first to retrieve valid VINs or stock numbers, "
                f"or retry with a known identifier such as: {sample_hints}."
            ),
        )
        log_outcome(
            action_name="fetch_active_manufacturer_rebates",
            status="error",
            actual_outcome=error_payload,
            agent_name="inventory_specialist",
        )
        return error_payload

    normalized_qualifiers = _normalize_customer_qualifiers(
        validated.customer_qualifiers
    )
    matched_rebates = get_applicable_rebates(
        vehicle=vehicle,
        customer_qualifiers=normalized_qualifiers,
    )
    total_rebate = round(sum(r.amount for r in matched_rebates), 2)

    response = {
        "status": "success",
        "vin": vehicle.vin,
        "stock_number": vehicle.stock_number,
        "vehicle_title": vehicle.display_name,
        "normalized_qualifiers": normalized_qualifiers,
        "rebate_count": len(matched_rebates),
        "total_rebate_amount": total_rebate,
        "rebates": [r.model_dump() for r in matched_rebates],
    }
    log_outcome(
        action_name="fetch_active_manufacturer_rebates",
        status="success",
        actual_outcome={
            "vin": vehicle.vin,
            "rebate_count": len(matched_rebates),
            "total_rebate_amount": total_rebate,
        },
        agent_name="inventory_specialist",
    )
    return response
