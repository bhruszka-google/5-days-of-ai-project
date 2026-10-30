"""Sales Manager Human-in-the-Loop (HITL) Confirmation Gate & Discount Tool.

Implements the explicit code stop and ADK `FunctionTool(require_confirmation=...)`
hook for high-stakes promotional vehicle discounts (`apply_manager_special_discount`).
Whenever a salesperson or agent requests a discount exceeding
`DEFAULT_CONFIG.manager_approval_discount_threshold` ($1,500.00) or reducing the
vehicle selling price below `invoice_price + DEFAULT_CONFIG.min_dealer_margin_dollars`
($500.00), execution halts and returns a structured `PENDING_MANAGER_APPROVAL`
interrupt until a human Sales Manager provides a confirmation token.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool
from pydantic import ValidationError

from app.config import DEFAULT_CONFIG
from app.data.synthetic_catalog import get_all_vehicles, get_vehicle_by_id_or_vin
from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.tools.schemas import ManagerDiscountInput, build_guided_error_response

VALID_APPROVAL_PREFIXES = ("MGR-APPROVED-", "APPROVED-", "SM-APPROVED-")


def is_valid_manager_approval_token(token: Optional[str]) -> bool:
    """Verifies whether a Sales Manager HITL approval token is well-formed.

    Args:
        token: Optional token string provided by the human Sales Manager.

    Returns:
        bool: True if the token begins with a recognized approval prefix
        (e.g., `'MGR-APPROVED-2026'`) or equals `'APPROVED'`.
    """
    if not isinstance(token, str) or not token.strip():
        return False
    normalized = token.strip().upper()
    if normalized == "APPROVED":
        return True
    for prefix in VALID_APPROVAL_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return True
    return False


@traced_operation("autodeal.guardrails.requires_sales_manager_approval")
def requires_sales_manager_approval(
    vehicle_identifier: str,
    requested_discount_amount: float = 0.0,
    **kwargs: Any,
) -> bool:
    """Predicate determining if a discount requires human Sales Manager confirmation.

    Compatible directly with `google.adk.tools.FunctionTool(require_confirmation=...)`.
    Returns `True` when:
    1. `requested_discount_amount > DEFAULT_CONFIG.manager_approval_discount_threshold`
       ($1,500.00), OR
    2. Subtracting `requested_discount_amount` from the vehicle's advertised
       `selling_price` drops the net price below
       `vehicle.invoice_price + DEFAULT_CONFIG.min_dealer_margin_dollars` ($500.00).

    Args:
        vehicle_identifier: 17-character VIN, stock number, or display name.
        requested_discount_amount: Proposed dollar discount off asking price.
        **kwargs: Additional tool keyword arguments passed by the ADK runtime.

    Returns:
        bool: True if human Sales Manager approval is required; False if within
        standard desk salesperson authority.
    """
    _ = kwargs
    try:
        discount = float(requested_discount_amount)
    except (TypeError, ValueError):
        return True

    if discount > DEFAULT_CONFIG.manager_approval_discount_threshold:
        return True

    vehicle = get_vehicle_by_id_or_vin(str(vehicle_identifier))
    if vehicle is None:
        return False

    floor_price_threshold = round(
        vehicle.invoice_price + DEFAULT_CONFIG.min_dealer_margin_dollars, 2
    )
    proposed_selling_price = round(vehicle.selling_price - discount, 2)
    return proposed_selling_price < floor_price_threshold


@traced_operation("autodeal.tools.apply_manager_special_discount")
def apply_manager_special_discount(
    vehicle_identifier: str,
    requested_discount_amount: float,
    reason: str,
    manager_approval_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Applies a promotional discount to a vehicle with a Sales Manager HITL gate.

    Enforces the dealership's two-tier pricing governance policy:
    - **Standard Desk Authority**: Discounts `<= $1,500.00` that maintain at least
      `$500.00` gross margin above factory `invoice_price` are auto-approved
      (`approval_mode="AUTO_SALESPERSON_AUTHORITY"`).
    - **High-Stakes HITL Code Stop**: Discounts `> $1,500.00` or any discount that
      reduces the selling price below `invoice_price + $500.00` trigger an immediate
      execution stop (`status="PENDING_MANAGER_APPROVAL"`,
      `requires_human_confirmation=True`) unless a valid `manager_approval_token`
      (e.g., `"MGR-APPROVED-2026"`) is supplied by the Sales Manager.

    Args:
        vehicle_identifier: 17-character VIN, stock number (e.g., `'STK-2026-001'`),
            or vehicle display title.
        requested_discount_amount: Dollar discount requested below the advertised
            selling price (must be > 0).
        reason: Business justification for requesting the promotional discount
            (automatically scrubbed for customer PII).
        manager_approval_token: Optional confirmation token from the human Sales
            Manager (e.g., `'MGR-APPROVED-2026'`) authorizing a high-stakes discount.

    Returns:
        Dict[str, Any]:
        - When paused awaiting human confirmation: returns
          `status="PENDING_MANAGER_APPROVAL"`, `requires_human_confirmation=True`,
          `hitl_interrupt_id`, `vehicle_identifier`, `vehicle_title`, `msrp`,
          `current_selling_price`, `invoice_price`, `requested_discount_amount`,
          `proposed_discounted_price`, `floor_price_threshold`, and
          `approval_instructions`.
        - When approved: returns `status="APPROVED"`,
          `requires_human_confirmation=False`, `approval_mode`,
          `approved_selling_price`, `dealer_gross_margin_dollars`,
          `discount_applied`, and `reason`.
        - On invalid input or unknown vehicle: returns a structured
          `ToolErrorResponse` dictionary with `recovery_suggestion`.

    Raises:
        None: All validation and lookup errors are returned as guided error dicts.

    Example:
        >>> pending = apply_manager_special_discount(
        ...     vehicle_identifier="STK-2026-001",
        ...     requested_discount_amount=2500.0,
        ...     reason="Customer competing quote from rival dealership",
        ... )
        >>> pending["status"]
        'PENDING_MANAGER_APPROVAL'
        >>> approved = apply_manager_special_discount(
        ...     vehicle_identifier="STK-2026-001",
        ...     requested_discount_amount=2500.0,
        ...     reason="Customer competing quote from rival dealership",
        ...     manager_approval_token="MGR-APPROVED-2026",
        ... )
        >>> approved["status"]
        'APPROVED'
    """
    raw_args = {
        "vehicle_identifier": vehicle_identifier,
        "requested_discount_amount": requested_discount_amount,
        "reason": reason,
        "manager_approval_token": manager_approval_token,
    }
    log_intent(
        action_name="apply_manager_special_discount",
        intended_arguments=raw_args,
        agent_name="deal_desk_closer",
    )

    try:
        validated = ManagerDiscountInput(**raw_args)
    except ValidationError as exc:
        error_payload = build_guided_error_response(
            error_code="INVALID_DISCOUNT_INPUT",
            message=f"Manager discount input validation failed: {exc.errors()[0]['msg']}",
            recovery_suggestion=(
                "Provide a valid `vehicle_identifier`, a positive `requested_discount_amount` (> 0), "
                "and a clear business `reason` (at least 3 characters)."
            ),
        )
        log_outcome(
            action_name="apply_manager_special_discount",
            status="error",
            actual_outcome=error_payload,
            agent_name="deal_desk_closer",
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
                "Call `search_dealership_inventory` to retrieve valid stock numbers or VINs, "
                f"such as: {sample_str}."
            ),
        )
        log_outcome(
            action_name="apply_manager_special_discount",
            status="error",
            actual_outcome=error_payload,
            agent_name="deal_desk_closer",
        )
        return error_payload

    if validated.requested_discount_amount >= vehicle.selling_price:
        error_payload = build_guided_error_response(
            error_code="EXCESSIVE_DISCOUNT_AMOUNT",
            message=(
                f"Requested discount (${validated.requested_discount_amount:,.2f}) cannot equal "
                f"or exceed the vehicle's selling price (${vehicle.selling_price:,.2f})."
            ),
            recovery_suggestion=(
                "Specify a realistic promotional discount amount smaller than the vehicle's "
                f"advertised selling price (${vehicle.selling_price:,.2f})."
            ),
        )
        log_outcome(
            action_name="apply_manager_special_discount",
            status="error",
            actual_outcome=error_payload,
            agent_name="deal_desk_closer",
        )
        return error_payload

    sanitized_reason = DEFAULT_PII_REDACTOR.redact_text(validated.reason)
    discount_amount = round(validated.requested_discount_amount, 2)
    proposed_price = round(vehicle.selling_price - discount_amount, 2)
    floor_price_threshold = round(
        vehicle.invoice_price + DEFAULT_CONFIG.min_dealer_margin_dollars, 2
    )
    projected_margin = round(proposed_price - vehicle.invoice_price, 2)

    needs_approval = requires_sales_manager_approval(
        vehicle_identifier=vehicle.vin,
        requested_discount_amount=discount_amount,
    )
    token_is_valid = is_valid_manager_approval_token(validated.manager_approval_token)

    if needs_approval and not token_is_valid:
        triggers: List[str] = []
        if discount_amount > DEFAULT_CONFIG.manager_approval_discount_threshold:
            triggers.append(
                f"Requested discount (${discount_amount:,.2f}) exceeds the "
                f"${DEFAULT_CONFIG.manager_approval_discount_threshold:,.2f} desk authority cap."
            )
        if proposed_price < floor_price_threshold:
            triggers.append(
                f"Proposed selling price (${proposed_price:,.2f}) drops below the dealer floor "
                f"threshold of ${floor_price_threshold:,.2f} (invoice ${vehicle.invoice_price:,.2f} "
                f"+ ${DEFAULT_CONFIG.min_dealer_margin_dollars:,.2f} minimum gross margin)."
            )

        hitl_interrupt_id = f"HITL-{vehicle.stock_number}-{int(discount_amount)}"
        approval_instructions = (
            "HUMAN-IN-THE-LOOP STOP: Sales Manager authorization is required before locking "
            f"this ${discount_amount:,.2f} discount on {vehicle.display_name} "
            f"({vehicle.stock_number}). Reason(s): {' '.join(triggers)} "
            "To approve and resume execution, have the Sales Manager review the deal and "
            "re-invoke `apply_manager_special_discount` with "
            "`manager_approval_token='MGR-APPROVED-<CODE>'`."
        )

        pending_payload: Dict[str, Any] = {
            "status": "PENDING_MANAGER_APPROVAL",
            "requires_human_confirmation": True,
            "hitl_interrupt_id": hitl_interrupt_id,
            "vin": vehicle.vin,
            "stock_number": vehicle.stock_number,
            "vehicle_identifier": validated.vehicle_identifier,
            "vehicle_title": vehicle.display_name,
            "msrp": vehicle.msrp,
            "current_selling_price": vehicle.selling_price,
            "invoice_price": vehicle.invoice_price,
            "requested_discount_amount": discount_amount,
            "proposed_discounted_price": proposed_price,
            "floor_price_threshold": floor_price_threshold,
            "projected_dealer_gross_margin_dollars": projected_margin,
            "reason": sanitized_reason,
            "approval_triggers": triggers,
            "approval_instructions": approval_instructions,
        }
        log_outcome(
            action_name="apply_manager_special_discount",
            status="pending_manager_approval",
            actual_outcome={
                "hitl_interrupt_id": hitl_interrupt_id,
                "vin": vehicle.vin,
                "requested_discount_amount": discount_amount,
                "proposed_discounted_price": proposed_price,
                "requires_human_confirmation": True,
            },
            agent_name="deal_desk_closer",
        )
        return pending_payload

    approval_mode = (
        "HUMAN_MANAGER_CONFIRMED" if token_is_valid else "AUTO_SALESPERSON_AUTHORITY"
    )
    approved_payload: Dict[str, Any] = {
        "status": "APPROVED",
        "requires_human_confirmation": False,
        "approval_mode": approval_mode,
        "vin": vehicle.vin,
        "stock_number": vehicle.stock_number,
        "vehicle_identifier": validated.vehicle_identifier,
        "vehicle_title": vehicle.display_name,
        "msrp": vehicle.msrp,
        "original_selling_price": vehicle.selling_price,
        "approved_selling_price": proposed_price,
        "invoice_price": vehicle.invoice_price,
        "floor_price_threshold": floor_price_threshold,
        "dealer_gross_margin_dollars": projected_margin,
        "discount_applied": discount_amount,
        "reason": sanitized_reason,
        "manager_approval_token": (
            validated.manager_approval_token.strip()
            if validated.manager_approval_token
            else None
        ),
    }
    log_outcome(
        action_name="apply_manager_special_discount",
        status="approved",
        actual_outcome={
            "vin": vehicle.vin,
            "approval_mode": approval_mode,
            "approved_selling_price": proposed_price,
            "dealer_gross_margin_dollars": projected_margin,
            "discount_applied": discount_amount,
        },
        agent_name="deal_desk_closer",
    )
    return approved_payload


def build_manager_discount_tool() -> FunctionTool:
    """Constructs the ADK `FunctionTool` with `require_confirmation` HITL hook enabled.

    Returns:
        FunctionTool: ADK FunctionTool wrapping `apply_manager_special_discount`
        with `require_confirmation=requires_sales_manager_approval`.
    """
    return FunctionTool(
        func=apply_manager_special_discount,
        require_confirmation=requires_sales_manager_approval,
    )
