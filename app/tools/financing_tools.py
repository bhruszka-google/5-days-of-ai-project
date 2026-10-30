"""Financing/Lease Calculators, Floor-Margin Verification, and Top-3 Offer Sheet Generator.

Provides ADK-compatible tool functions for the `top_offers_strategist` and
`deal_desk_closer` agents with:
- Strict Pydantic input and output schema validation (`app/tools/schemas.py`)
- Exact amortized loan math and 36-month residual/money-factor lease calculations
- Dealership floor-margin enforcement (`invoice_price + min_dealer_margin_dollars`)
- Active PII redaction (`DEFAULT_PII_REDACTOR`) and Truth-in-Lending Regulation Z disclosures
- Guided error recovery (`ToolErrorResponse`) so invalid parameters never crash the agent
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.config import DEFAULT_CONFIG
from app.data.synthetic_catalog import (
    VehicleRecord,
    get_all_vehicles,
    get_applicable_rebates,
    get_credit_tier_rate,
    get_vehicle_by_id_or_vin,
)
from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.tools.schemas import (
    ALLOWED_LOAN_TERMS,
    FinancingCalculationInput,
    MarginCheckInput,
    OfferSheetInput,
    OfferTierLabel,
    TopThreeOffersResponse,
    VehicleOfferCard,
    build_guided_error_response,
)


def _compute_amortized_monthly_payment(
    principal: float,
    annual_apr_percent: float,
    term_months: int,
) -> Tuple[float, float, float]:
    """Computes exact amortized monthly payment, total loan cost, and total interest."""
    if principal <= 0.0:
        return 0.0, 0.0, 0.0

    if annual_apr_percent <= 0.0:
        monthly = round(principal / term_months, 2)
        total_paid = round(monthly * term_months, 2)
        return monthly, total_paid, 0.0

    monthly_rate = (annual_apr_percent / 100.0) / 12.0
    compound = (1.0 + monthly_rate) ** term_months
    raw_monthly = principal * (monthly_rate * compound) / (compound - 1.0)
    monthly_payment = round(raw_monthly, 2)
    total_payments = round(monthly_payment * term_months, 2)
    total_interest = round(max(0.0, total_payments - principal), 2)
    return monthly_payment, total_payments, total_interest


def _compute_36m_lease_payment(
    msrp: float,
    net_cap_cost: float,
    residual_pct_36m: float,
    money_factor_36m: float,
) -> Tuple[float, float, float, float]:
    """Computes 36-month monthly lease payment (depreciation fee + finance fee)."""
    residual_value = round(msrp * residual_pct_36m, 2)
    effective_cap = max(0.0, net_cap_cost)
    depreciation_fee = max(0.0, (effective_cap - residual_value) / 36.0)
    finance_fee = (effective_cap + residual_value) * money_factor_36m
    monthly_lease = round(depreciation_fee + finance_fee, 2)
    return (
        monthly_lease,
        residual_value,
        round(depreciation_fee, 2),
        round(finance_fee, 2),
    )


@traced_operation("autodeal.tools.calculate_financing_and_lease_options")
def calculate_financing_and_lease_options(
    vehicle_identifier: str,
    down_payment: float = 0.0,
    trade_in_equity: float = 0.0,
    credit_score: int = 720,
    loan_term_months: int = 60,
    include_rebates: bool = True,
) -> Dict[str, Any]:
    """Calculates exact amortized auto loan and 36-month lease payments for a vehicle.

    Resolves the vehicle from dealership inventory, applies stackable manufacturer
    rebates (including Loyalty/Trade-In bonus when `trade_in_equity > 0`), looks up
    the customer's credit tier APR and lease money factor from `get_credit_tier_rate`,
    and computes exact monthly loan and lease figures with Truth-in-Lending Act
    (Regulation Z) disclosures.

    Args:
        vehicle_identifier: 17-character VIN, stock number (e.g., `'STK-2026-001'`),
            or vehicle display name.
        down_payment: Cash down payment in USD (>= 0.0). Defaults to 0.0.
        trade_in_equity: Net trade-in credit (or negative rollover) in USD. Defaults to 0.0.
        credit_score: Customer FICO credit score (300 to 850). Defaults to 720.
        loan_term_months: Loan term in months (`24`, `36`, `48`, `60`, or `72`).
            Defaults to 60.
        include_rebates: Whether to automatically deduct eligible manufacturer
            rebates from the selling price. Defaults to True.

    Returns:
        Dict[str, Any]: On success, returns `status="success"`, `vin`,
        `stock_number`, `vehicle_title`, `msrp`, `selling_price`,
        `rebates_applied`, `net_cap_cost`, `credit_tier`, `apr_percent`,
        `loan_term_months`, `estimated_monthly_finance`, `total_interest_paid`,
        `estimated_monthly_lease_36m`, and `truth_in_lending_disclosure`. On
        invalid loan term, out-of-range credit score, or unknown vehicle,
        returns a structured `ToolErrorResponse` dictionary.

    Raises:
        None: All validation and lookup errors are returned as guided error dicts.

    Example:
        >>> quote = calculate_financing_and_lease_options(
        ...     vehicle_identifier="STK-2026-001",
        ...     down_payment=4000.0,
        ...     trade_in_equity=6500.0,
        ...     credit_score=740,
        ...     loan_term_months=60,
        ... )
        >>> quote["status"]
        'success'
    """
    raw_args = {
        "vehicle_identifier": vehicle_identifier,
        "down_payment": down_payment,
        "trade_in_equity": trade_in_equity,
        "credit_score": credit_score,
        "loan_term_months": loan_term_months,
        "include_rebates": include_rebates,
    }
    log_intent(
        action_name="calculate_financing_and_lease_options",
        intended_arguments=raw_args,
        agent_name="top_offers_strategist",
    )

    try:
        validated = FinancingCalculationInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_FINANCING_INPUT",
            message=f"Financing parameter validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                f"Adjust parameters and retry: `loan_term_months` must be one of {list(ALLOWED_LOAN_TERMS)}, "
                "`credit_score` must be an integer between 300 and 850, `down_payment` must be >= 0, "
                "and `vehicle_identifier` must be a valid VIN, stock number, or vehicle title."
            ),
        )
        log_outcome(
            action_name="calculate_financing_and_lease_options",
            status="error",
            actual_outcome=error_payload,
            agent_name="top_offers_strategist",
        )
        return error_payload

    vehicle = get_vehicle_by_id_or_vin(validated.vehicle_identifier)
    if vehicle is None:
        sample = get_all_vehicles()[:3]
        sample_str = ", ".join(
            f"{v.stock_number} ({v.display_name}, VIN {v.vin})" for v in sample
        )
        error_payload = build_guided_error_response(
            error_code="VEHICLE_NOT_FOUND",
            message=f"Vehicle '{validated.vehicle_identifier}' was not found in dealership inventory.",
            recovery_suggestion=(
                "Call `search_dealership_inventory` to obtain valid VINs or stock numbers, "
                f"or use one of: {sample_str}."
            ),
        )
        log_outcome(
            action_name="calculate_financing_and_lease_options",
            status="error",
            actual_outcome=error_payload,
            agent_name="top_offers_strategist",
        )
        return error_payload

    qualifiers = ["trade_in"] if validated.trade_in_equity > 0 else []
    rebates = (
        get_applicable_rebates(vehicle, customer_qualifiers=qualifiers)
        if validated.include_rebates
        else []
    )
    rebates_total = round(sum(r.amount for r in rebates), 2)

    net_cap_cost = round(
        max(
            0.0,
            vehicle.selling_price
            - rebates_total
            - validated.down_payment
            - validated.trade_in_equity,
        ),
        2,
    )

    tier = get_credit_tier_rate(credit_score=validated.credit_score)
    apr_percent = float(tier.apr_by_term_months[validated.loan_term_months])

    monthly_loan, total_loan_payments, total_interest = (
        _compute_amortized_monthly_payment(
            principal=net_cap_cost,
            annual_apr_percent=apr_percent,
            term_months=validated.loan_term_months,
        )
    )

    monthly_lease, residual_36m, depreciation_fee, finance_fee = (
        _compute_36m_lease_payment(
            msrp=vehicle.msrp,
            net_cap_cost=net_cap_cost,
            residual_pct_36m=tier.lease_residual_pct_36m,
            money_factor_36m=tier.lease_money_factor,
        )
    )

    disclosure = (
        f"Truth-in-Lending (Regulation Z) Disclosure: {apr_percent:.2f}% APR for "
        f"{validated.loan_term_months} months ({tier.label}) on ${net_cap_cost:,.2f} "
        f"amount financed equals ${monthly_loan:,.2f}/mo (total of payments "
        f"${total_loan_payments:,.2f}; finance charge ${total_interest:,.2f}). "
        "Excludes state/local taxes, title, and registration fees."
    )

    response = {
        "status": "success",
        "vin": vehicle.vin,
        "stock_number": vehicle.stock_number,
        "vehicle_title": vehicle.display_name,
        "msrp": vehicle.msrp,
        "selling_price": vehicle.selling_price,
        "rebates_applied": rebates_total,
        "rebate_programs": [r.program_name for r in rebates],
        "down_payment": round(validated.down_payment, 2),
        "trade_in_equity": round(validated.trade_in_equity, 2),
        "net_cap_cost": net_cap_cost,
        "credit_score": validated.credit_score,
        "credit_tier": tier.tier_id,
        "credit_tier_label": tier.label,
        "loan_term_months": validated.loan_term_months,
        "apr_percent": apr_percent,
        "estimated_monthly_finance": monthly_loan,
        "monthly_loan_payment": monthly_loan,
        "total_loan_payments": total_loan_payments,
        "total_interest_paid": total_interest,
        "estimated_monthly_lease_36m": monthly_lease,
        "monthly_lease_payment_36m": monthly_lease,
        "lease_residual_percent_36m": tier.lease_residual_pct_36m,
        "lease_residual_value_36m": residual_36m,
        "lease_money_factor_36m": tier.lease_money_factor,
        "lease_depreciation_fee": depreciation_fee,
        "lease_finance_fee": finance_fee,
        "truth_in_lending_disclosure": disclosure,
    }
    log_outcome(
        action_name="calculate_financing_and_lease_options",
        status="success",
        actual_outcome={
            "vin": vehicle.vin,
            "net_cap_cost": net_cap_cost,
            "apr_percent": apr_percent,
            "estimated_monthly_finance": monthly_loan,
            "estimated_monthly_lease_36m": monthly_lease,
        },
        agent_name="top_offers_strategist",
    )
    return response


@traced_operation("autodeal.tools.verify_dealership_floor_margin")
def verify_dealership_floor_margin(
    vehicle_identifier: str,
    proposed_selling_price: float,
    rebates_total: float = 0.0,
) -> Dict[str, Any]:
    """Audits a proposed vehicle selling price against dealership floor-margin policy.

    Compares `proposed_selling_price` against `vehicle.invoice_price +
    DEFAULT_CONFIG.min_dealer_margin_dollars` ($500.00). Note that factory-backed
    manufacturer rebates (`rebates_total`) are reimbursed by the OEM and therefore
    reduce the customer's net price without reducing dealer gross margin. Also
    determines whether Human-in-the-Loop (HITL) Sales Manager approval is required.

    Args:
        vehicle_identifier: 17-character VIN, stock number, or vehicle display name.
        proposed_selling_price: Proposed dealership selling price before
            manufacturer rebates in USD (must be > 0).
        rebates_total: Total OEM manufacturer rebates applied in USD (>= 0.0).

    Returns:
        Dict[str, Any]: On success, returns `status="success"`, `vin`,
        `stock_number`, `vehicle_title`, `invoice_price`, `advertised_selling_price`,
        `proposed_selling_price`, `minimum_allowed_selling_price`,
        `dealer_gross_margin_dollars`, `min_required_margin_dollars`,
        `floor_margin_compliant`, `requires_hitl_manager_approval`, and
        `policy_guidance`. On validation error or unknown vehicle, returns a
        guided `ToolErrorResponse` dictionary.

    Raises:
        None: All validation and lookup errors return guided error dicts.

    Example:
        >>> audit = verify_dealership_floor_margin(
        ...     vehicle_identifier="STK-2026-001",
        ...     proposed_selling_price=46800.0,
        ...     rebates_total=1250.0,
        ... )
        >>> audit["floor_margin_compliant"]
        True
    """
    raw_args = {
        "vehicle_identifier": vehicle_identifier,
        "proposed_selling_price": proposed_selling_price,
        "rebates_total": rebates_total,
    }
    log_intent(
        action_name="verify_dealership_floor_margin",
        intended_arguments=raw_args,
        agent_name="top_offers_strategist",
    )

    try:
        validated = MarginCheckInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_MARGIN_CHECK_INPUT",
            message=f"Floor margin check validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                "Provide a valid `vehicle_identifier`, a positive `proposed_selling_price` (> 0), "
                "and a non-negative `rebates_total` (>= 0)."
            ),
        )
        log_outcome(
            action_name="verify_dealership_floor_margin",
            status="error",
            actual_outcome=error_payload,
            agent_name="top_offers_strategist",
        )
        return error_payload

    vehicle = get_vehicle_by_id_or_vin(validated.vehicle_identifier)
    if vehicle is None:
        sample = get_all_vehicles()[:3]
        sample_str = ", ".join(
            f"{v.stock_number} ({v.display_name}, VIN {v.vin})" for v in sample
        )
        error_payload = build_guided_error_response(
            error_code="VEHICLE_NOT_FOUND",
            message=f"Vehicle '{validated.vehicle_identifier}' was not found in dealership inventory.",
            recovery_suggestion=(
                "Call `search_dealership_inventory` to locate valid vehicle identifiers, "
                f"or use a known stock number/VIN such as: {sample_str}."
            ),
        )
        log_outcome(
            action_name="verify_dealership_floor_margin",
            status="error",
            actual_outcome=error_payload,
            agent_name="top_offers_strategist",
        )
        return error_payload

    min_margin_req = DEFAULT_CONFIG.min_dealer_margin_dollars
    minimum_allowed_price = round(vehicle.invoice_price + min_margin_req, 2)
    dealer_gross_margin = round(
        validated.proposed_selling_price - vehicle.invoice_price, 2
    )
    discount_from_asking = round(
        max(0.0, vehicle.selling_price - validated.proposed_selling_price), 2
    )
    floor_compliant = validated.proposed_selling_price >= minimum_allowed_price
    requires_hitl = (not floor_compliant) or (
        discount_from_asking > DEFAULT_CONFIG.manager_approval_discount_threshold
    )

    if floor_compliant and not requires_hitl:
        policy_guidance = (
            f"COMPLIANT: Proposed selling price (${validated.proposed_selling_price:,.2f}) "
            f"maintains ${dealer_gross_margin:,.2f} gross margin above invoice "
            f"(${vehicle.invoice_price:,.2f}), meeting the ${min_margin_req:,.2f} floor policy."
        )
    else:
        shortfall = round(
            max(0.0, minimum_allowed_price - validated.proposed_selling_price), 2
        )
        policy_guidance = (
            f"MANAGER APPROVAL REQUIRED: Proposed price (${validated.proposed_selling_price:,.2f}) "
            f"is ${shortfall:,.2f} below the autonomous floor (${minimum_allowed_price:,.2f}) "
            f"or exceeds the ${DEFAULT_CONFIG.manager_approval_discount_threshold:,.2f} discount cap. "
            "Invoke `apply_manager_special_discount` to request Sales Manager HITL authorization."
        )

    response = {
        "status": "success",
        "vin": vehicle.vin,
        "stock_number": vehicle.stock_number,
        "vehicle_title": vehicle.display_name,
        "msrp": vehicle.msrp,
        "invoice_price": vehicle.invoice_price,
        "advertised_selling_price": vehicle.selling_price,
        "proposed_selling_price": round(validated.proposed_selling_price, 2),
        "rebates_total": round(validated.rebates_total, 2),
        "effective_customer_price_after_rebates": round(
            max(0.0, validated.proposed_selling_price - validated.rebates_total), 2
        ),
        "minimum_allowed_selling_price": minimum_allowed_price,
        "min_required_margin_dollars": min_margin_req,
        "dealer_gross_margin_dollars": dealer_gross_margin,
        "discount_from_advertised_price": discount_from_asking,
        "floor_margin_compliant": floor_compliant,
        "requires_hitl_manager_approval": requires_hitl,
        "policy_guidance": policy_guidance,
    }
    log_outcome(
        action_name="verify_dealership_floor_margin",
        status="success",
        actual_outcome={
            "vin": vehicle.vin,
            "dealer_gross_margin_dollars": dealer_gross_margin,
            "floor_margin_compliant": floor_compliant,
            "requires_hitl_manager_approval": requires_hitl,
        },
        agent_name="top_offers_strategist",
    )
    return response


def _build_why_it_fits(
    tier_label: OfferTierLabel,
    vehicle: VehicleRecord,
    rebates_applied: float,
    monthly_finance: float,
) -> str:
    """Generates a concise, tailored value proposition for each offer tier."""
    features_preview = ", ".join(vehicle.key_features[:3])
    if tier_label == "Best Overall Match":
        return (
            f"Balances family comfort ({vehicle.seating_capacity} seats, {vehicle.mpg_combined} MPG "
            f"{vehicle.fuel_type.upper()}, {vehicle.drivetrain}) with key equipment ({features_preview}) "
            f"at ${monthly_finance:,.2f}/mo."
        )
    if tier_label == "Budget / Value Pick":
        return (
            f"Maximizes affordability ({vehicle.condition.upper()}, ${rebates_applied:,.0f} in stackable "
            f"rebates, {vehicle.mpg_combined} MPG) for the lowest monthly payment (${monthly_finance:,.2f}/mo)."
        )
    return (
        f"Upgrades to {vehicle.trim} trim with premium amenities ({features_preview}) "
        f"and {vehicle.drivetrain} capability at ${monthly_finance:,.2f}/mo."
    )


@traced_operation("autodeal.tools.generate_customer_offer_sheet")
def generate_customer_offer_sheet(
    customer_name: str,
    best_match_vin: str,
    value_pick_vin: str,
    upgrade_pick_vin: str,
    down_payment: float = 0.0,
    trade_in_equity: float = 0.0,
    credit_score: int = 720,
    loan_term_months: int = 60,
    custom_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Compiles a validated Top-3 Customer Offer Sheet with Regulation Z disclosures.

    Constructs the three canonical offer cards (`Best Overall Match`,
    `Budget / Value Pick`, and `Premium / Upgrade Pick`), computes exact
    financing and 36-month lease figures for each vehicle, verifies floor-margin
    compliance, and scrubs any accidental customer PII from `customer_name` and
    `custom_notes` before returning a strict `TopThreeOffersResponse` payload.

    Args:
        customer_name: Customer name or buyer profile label (PII is automatically
            redacted).
        best_match_vin: VIN, stock number, or display title for Rank 1
            (`Best Overall Match`).
        value_pick_vin: VIN, stock number, or display title for Rank 2
            (`Budget / Value Pick`).
        upgrade_pick_vin: VIN, stock number, or display title for Rank 3
            (`Premium / Upgrade Pick`).
        down_payment: Cash down payment in USD (>= 0.0). Defaults to 0.0.
        trade_in_equity: Net trade-in equity credit in USD. Defaults to 0.0.
        credit_score: Customer FICO credit score (300 to 850). Defaults to 720.
        loan_term_months: Loan term in months (`24`, `36`, `48`, `60`, or `72`).
            Defaults to 60.
        custom_notes: Optional salesperson desk notes (automatically scrubbed for
            SSN, email, phone, credit card, and driver's license PII).

    Returns:
        Dict[str, Any]: On success, returns a dictionary validated by
        `TopThreeOffersResponse` containing `status="success"`, `customer_name`,
        `customer_summary`, `offers` (list of 3 `VehicleOfferCard` dicts),
        `compliance_disclaimer`, and `custom_notes`. On duplicate vehicles,
        unknown VINs, or invalid financing terms, returns a guided
        `ToolErrorResponse` dictionary.

    Raises:
        None: All validation and lookup errors are returned as guided error dicts.

    Example:
        >>> sheet = generate_customer_offer_sheet(
        ...     customer_name="Family SUV Buyer",
        ...     best_match_vin="STK-2026-001",
        ...     value_pick_vin="STK-2024-008",
        ...     upgrade_pick_vin="STK-2026-002",
        ...     down_payment=4000.0,
        ...     trade_in_equity=6500.0,
        ...     credit_score=740,
        ...     loan_term_months=60,
        ... )
        >>> len(sheet["offers"])
        3
    """
    raw_args = {
        "customer_name": customer_name,
        "best_match_vin": best_match_vin,
        "value_pick_vin": value_pick_vin,
        "upgrade_pick_vin": upgrade_pick_vin,
        "down_payment": down_payment,
        "trade_in_equity": trade_in_equity,
        "credit_score": credit_score,
        "loan_term_months": loan_term_months,
        "custom_notes": custom_notes,
    }
    log_intent(
        action_name="generate_customer_offer_sheet",
        intended_arguments=raw_args,
        agent_name="deal_desk_closer",
    )

    try:
        validated = OfferSheetInput(**raw_args)
    except ValidationError as exc:
        first_msg = exc.errors()[0]["msg"]
        is_duplicate = "distinct" in first_msg.lower()
        error_payload = build_guided_error_response(
            error_code=(
                "DUPLICATE_OFFER_VEHICLES"
                if is_duplicate
                else "INVALID_OFFER_SHEET_INPUT"
            ),
            message=f"Offer sheet validation failed: {first_msg}",
            recovery_suggestion=(
                "Ensure `best_match_vin`, `value_pick_vin`, and `upgrade_pick_vin` reference "
                f"three distinct vehicles, `loan_term_months` is one of {list(ALLOWED_LOAN_TERMS)}, "
                "and `credit_score` is between 300 and 850."
            ),
        )
        log_outcome(
            action_name="generate_customer_offer_sheet",
            status="error",
            actual_outcome=error_payload,
            agent_name="deal_desk_closer",
        )
        return error_payload

    tier_specs: List[Tuple[int, OfferTierLabel, str]] = [
        (1, "Best Overall Match", validated.best_match_vin),
        (2, "Budget / Value Pick", validated.value_pick_vin),
        (3, "Premium / Upgrade Pick", validated.upgrade_pick_vin),
    ]

    resolved_vehicles: List[Tuple[int, OfferTierLabel, VehicleRecord]] = []
    for rank, tier_label, identifier in tier_specs:
        vehicle = get_vehicle_by_id_or_vin(identifier)
        if vehicle is None:
            sample = get_all_vehicles()[:4]
            sample_str = ", ".join(
                f"{v.stock_number} ({v.display_name}, VIN {v.vin})" for v in sample
            )
            error_payload = build_guided_error_response(
                error_code="VEHICLE_NOT_FOUND",
                message=(
                    f"Offer rank {rank} ({tier_label}) vehicle identifier '{identifier}' "
                    "was not found in dealership inventory."
                ),
                recovery_suggestion=(
                    "Call `search_dealership_inventory` to get valid VINs or stock numbers, "
                    f"then pass three distinct vehicles such as: {sample_str}."
                ),
            )
            log_outcome(
                action_name="generate_customer_offer_sheet",
                status="error",
                actual_outcome=error_payload,
                agent_name="deal_desk_closer",
            )
            return error_payload
        resolved_vehicles.append((rank, tier_label, vehicle))

    resolved_vins = [v.vin for _, _, v in resolved_vehicles]
    if len(set(resolved_vins)) != 3:
        error_payload = build_guided_error_response(
            error_code="DUPLICATE_OFFER_VEHICLES",
            message=(
                f"Resolved VINs {resolved_vins} contain duplicate vehicles. "
                "Each of the Top 3 Offer tiers must present a unique vehicle."
            ),
            recovery_suggestion=(
                "Select three different vehicles from `search_dealership_inventory` for "
                "`best_match_vin`, `value_pick_vin`, and `upgrade_pick_vin`."
            ),
        )
        log_outcome(
            action_name="generate_customer_offer_sheet",
            status="error",
            actual_outcome=error_payload,
            agent_name="deal_desk_closer",
        )
        return error_payload

    offer_cards: List[VehicleOfferCard] = []
    for rank, tier_label, vehicle in resolved_vehicles:
        fin = calculate_financing_and_lease_options(
            vehicle_identifier=vehicle.vin,
            down_payment=validated.down_payment,
            trade_in_equity=validated.trade_in_equity,
            credit_score=validated.credit_score,
            loan_term_months=validated.loan_term_months,
            include_rebates=True,
        )
        margin = verify_dealership_floor_margin(
            vehicle_identifier=vehicle.vin,
            proposed_selling_price=vehicle.selling_price,
            rebates_total=float(fin["rebates_applied"]),
        )
        card = VehicleOfferCard(
            rank=rank,
            tier_label=tier_label,
            vin=vehicle.vin,
            stock_number=vehicle.stock_number,
            vehicle_title=vehicle.display_name,
            msrp=vehicle.msrp,
            selling_price=vehicle.selling_price,
            rebates_applied=float(fin["rebates_applied"]),
            net_cap_cost=float(fin["net_cap_cost"]),
            estimated_monthly_finance=float(fin["estimated_monthly_finance"]),
            apr_percent=float(fin["apr_percent"]),
            loan_term_months=validated.loan_term_months,
            estimated_monthly_lease_36m=float(fin["estimated_monthly_lease_36m"]),
            why_it_fits=_build_why_it_fits(
                tier_label=tier_label,
                vehicle=vehicle,
                rebates_applied=float(fin["rebates_applied"]),
                monthly_finance=float(fin["estimated_monthly_finance"]),
            ),
            floor_margin_compliant=bool(margin["floor_margin_compliant"]),
        )
        offer_cards.append(card)

    sanitized_name = DEFAULT_PII_REDACTOR.redact_text(validated.customer_name)
    sanitized_notes = (
        DEFAULT_PII_REDACTOR.redact_text(validated.custom_notes)
        if validated.custom_notes is not None
        else None
    )

    tier_info = get_credit_tier_rate(credit_score=validated.credit_score)
    apr_pct = tier_info.apr_by_term_months[validated.loan_term_months]
    customer_summary = (
        f"Prepared Top 3 Tailored Vehicle Offers for {sanitized_name} based on "
        f"${validated.down_payment:,.2f} down payment, ${validated.trade_in_equity:,.2f} "
        f"net trade-in equity, and {tier_info.label} ({apr_pct:.2f}% APR for "
        f"{validated.loan_term_months} months)."
    )
    compliance_disclaimer = (
        f"Truth-in-Lending Act (Regulation Z) Disclosure: Advertised financing assumes "
        f"{tier_info.label} credit approval ({apr_pct:.2f}% APR for {validated.loan_term_months} "
        "months). 36-month closed-end lease quotes assume 12,000 miles/year with $0 security "
        "deposit. All prices exclude state/local sales tax, title, license, and dealer doc fees."
    )

    sheet = TopThreeOffersResponse(
        customer_name=sanitized_name,
        customer_summary=customer_summary,
        offers=offer_cards,
        compliance_disclaimer=compliance_disclaimer,
        custom_notes=sanitized_notes,
    )
    output_dict = sheet.model_dump()
    log_outcome(
        action_name="generate_customer_offer_sheet",
        status="success",
        actual_outcome={
            "customer_name": sanitized_name,
            "offer_vins": [c.vin for c in offer_cards],
            "all_margin_compliant": all(c.floor_margin_compliant for c in offer_cards),
        },
        agent_name="deal_desk_closer",
    )
    return output_dict
