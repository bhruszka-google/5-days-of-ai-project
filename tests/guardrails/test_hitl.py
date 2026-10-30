"""Unit tests for app/guardrails/hitl.py."""

from __future__ import annotations

from app.guardrails.hitl import (
    apply_manager_special_discount,
    build_manager_discount_tool,
    is_valid_manager_approval_token,
    requires_sales_manager_approval,
)
from app.observability.logger import clear_captured_json_logs, get_captured_json_logs
from google.adk.tools import FunctionTool


class TestSalesManagerHitlGate:
    """Tests for Sales Manager HITL approval predicate and discount tool."""

    def setup_method(self) -> None:
        clear_captured_json_logs()

    def test_is_valid_manager_approval_token(self) -> None:
        assert is_valid_manager_approval_token("MGR-APPROVED-2026") is True
        assert is_valid_manager_approval_token("approved-991") is True
        assert is_valid_manager_approval_token("APPROVED") is True
        assert is_valid_manager_approval_token(None) is False
        assert is_valid_manager_approval_token("") is False
        assert is_valid_manager_approval_token("PENDING-123") is False
        assert is_valid_manager_approval_token("MGR-APPROVED-") is False

    def test_requires_sales_manager_approval_thresholds(self) -> None:
        # STK-2026-001 (2026 Highlander Hybrid XLE): selling=$46,800, invoice=$44,390, floor=$44,890
        assert (
            requires_sales_manager_approval(
                vehicle_identifier="STK-2026-001",
                requested_discount_amount=750.0,
            )
            is False
        )
        # Exceeds $1,500 threshold -> requires manager approval
        assert (
            requires_sales_manager_approval(
                vehicle_identifier="STK-2026-001",
                requested_discount_amount=2500.0,
            )
            is True
        )
        # STK-2025-016 (2025 Corolla LE): selling=$22,990, invoice=$22,150, floor=$22,650
        # A $500 discount drops price to $22,490 (< $22,650 floor) -> requires manager approval
        assert (
            requires_sales_manager_approval(
                vehicle_identifier="STK-2025-016",
                requested_discount_amount=500.0,
            )
            is True
        )

    def test_auto_approves_small_discount_within_salesperson_authority(self) -> None:
        result = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=800.0,
            reason="Loyal repeat customer (email: buyer@example.com) closing today",
        )

        assert result["status"] == "APPROVED"
        assert result["requires_human_confirmation"] is False
        assert result["approval_mode"] == "AUTO_SALESPERSON_AUTHORITY"
        assert result["approved_selling_price"] == 46000.0
        assert result["discount_applied"] == 800.0
        assert result["dealer_gross_margin_dollars"] == 1610.0
        assert "buyer@example.com" not in result["reason"]
        assert "[REDACTED_EMAIL]" in result["reason"]

    def test_pauses_for_hitl_confirmation_when_discount_exceeds_threshold(self) -> None:
        result = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=2500.0,
            reason="Matching competing dealer written quote",
        )

        assert result["status"] == "PENDING_MANAGER_APPROVAL"
        assert result["requires_human_confirmation"] is True
        assert result["hitl_interrupt_id"] == "HITL-STK-2026-001-2500"
        assert result["proposed_discounted_price"] == 44300.0
        assert result["floor_price_threshold"] == 44890.0
        assert "MGR-APPROVED-" in result["approval_instructions"]

        logs = get_captured_json_logs()
        events = [entry.get("event_type") for entry in logs]
        assert "INTENT" in events
        assert "OUTCOME" in events

    def test_approves_high_stakes_discount_with_valid_manager_token(self) -> None:
        result = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=2500.0,
            reason="End-of-month volume unit approved by General Sales Manager",
            manager_approval_token="MGR-APPROVED-2026",
        )

        assert result["status"] == "APPROVED"
        assert result["requires_human_confirmation"] is False
        assert result["approval_mode"] == "HUMAN_MANAGER_CONFIRMED"
        assert result["approved_selling_price"] == 44300.0
        assert result["discount_applied"] == 2500.0
        assert result["manager_approval_token"] == "MGR-APPROVED-2026"

    def test_guided_error_recovery_on_invalid_input_or_unknown_vehicle(self) -> None:
        bad_amount = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=-100.0,
            reason="Invalid negative discount",
        )
        assert bad_amount["status"] == "error"
        assert bad_amount["error_code"] == "INVALID_DISCOUNT_INPUT"
        assert "recovery_suggestion" in bad_amount

        unknown_car = apply_manager_special_discount(
            vehicle_identifier="STK-9999-999",
            requested_discount_amount=500.0,
            reason="Unknown stock number",
        )
        assert unknown_car["status"] == "error"
        assert unknown_car["error_code"] == "VEHICLE_NOT_FOUND"

        excessive = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=60000.0,
            reason="Discount exceeds vehicle price",
        )
        assert excessive["status"] == "error"
        assert excessive["error_code"] == "EXCESSIVE_DISCOUNT_AMOUNT"

    def test_build_manager_discount_tool_returns_adk_function_tool(self) -> None:
        tool = build_manager_discount_tool()
        assert isinstance(tool, FunctionTool)
        assert tool.name == "apply_manager_special_discount"
