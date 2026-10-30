"""Unit tests for `app.routing.model_router`."""

from __future__ import annotations

import pytest
from app.config import DEFAULT_CONFIG
from app.routing.model_router import (
    DEFAULT_MODEL_ROUTER,
    RoutingDecision,
    StrategicModelRouter,
    TaskComplexity,
)
from pydantic import ValidationError


def test_select_model_for_agent_assigns_flash_and_pro_appropriately() -> None:
    """Verifies Flash is assigned to fast/HITL agents and Pro to `top_offers_strategist`."""
    router = StrategicModelRouter()

    coord_decision = router.select_model_for_agent("dealership_coordinator")
    assert coord_decision.selected_model == DEFAULT_CONFIG.fast_model
    assert coord_decision.selected_model == "gemini-3-flash-preview"
    assert coord_decision.complexity == TaskComplexity.ORCHESTRATION

    inv_decision = router.select_model_for_agent("inventory_specialist")
    assert inv_decision.selected_model == "gemini-3-flash-preview"
    assert inv_decision.complexity == TaskComplexity.FAST_LOOKUP

    closer_decision = router.select_model_for_agent("deal_desk_closer")
    assert closer_decision.selected_model == "gemini-3-flash-preview"
    assert closer_decision.complexity == TaskComplexity.HITL_EXECUTION

    strategist_decision = router.select_model_for_agent("top_offers_strategist")
    assert strategist_decision.selected_model == DEFAULT_CONFIG.pro_model
    assert strategist_decision.selected_model == "gemini-3.1-pro-preview"
    assert strategist_decision.complexity == TaskComplexity.DEEP_REASONING


def test_select_model_for_agent_unknown_raises_key_error() -> None:
    """Verifies unknown agent names raise a descriptive `KeyError`."""
    with pytest.raises(KeyError, match="Unknown agent_name"):
        DEFAULT_MODEL_ROUTER.select_model_for_agent("unknown_specialist")


def test_route_query_classifies_fast_lookup_deep_reasoning_and_hitl() -> None:
    """Tests `route_query` across inventory lookup, Top-3 ranking, and HITL discount prompts."""
    fast_query = "Check how many 2026 Toyota Camry sedans are currently on the lot."
    fast_decision = DEFAULT_MODEL_ROUTER.route_query(fast_query)
    assert fast_decision.target_agent == "inventory_specialist"
    assert fast_decision.selected_model == "gemini-3-flash-preview"
    assert fast_decision.complexity == TaskComplexity.FAST_LOOKUP

    pro_query = (
        "Customer needs a 3-row hybrid SUV under $650/mo with a 740 credit score, "
        "$4,000 down, and a 2021 Honda CR-V trade-in. Rank the Top 3 offers."
    )
    pro_decision = DEFAULT_MODEL_ROUTER.route_query(pro_query)
    assert pro_decision.target_agent == "top_offers_strategist"
    assert pro_decision.selected_model == "gemini-3.1-pro-preview"
    assert pro_decision.complexity == TaskComplexity.DEEP_REASONING

    hitl_query = (
        "Apply a $2,500 manager special discount on STK-2026-001 to close the deal."
    )
    hitl_decision = DEFAULT_MODEL_ROUTER.route_query(hitl_query)
    assert hitl_decision.target_agent == "deal_desk_closer"
    assert hitl_decision.selected_model == "gemini-3-flash-preview"
    assert hitl_decision.complexity == TaskComplexity.HITL_EXECUTION


def test_routing_decision_forbids_extra_fields_and_summary_map() -> None:
    """Verifies `RoutingDecision` enforces `extra='forbid'` and `get_routing_summary` works."""
    summary = DEFAULT_MODEL_ROUTER.get_routing_summary()
    assert summary == {
        "dealership_coordinator": "gemini-3-flash-preview",
        "inventory_specialist": "gemini-3-flash-preview",
        "top_offers_strategist": "gemini-3.1-pro-preview",
        "deal_desk_closer": "gemini-3-flash-preview",
    }

    with pytest.raises(ValidationError):
        RoutingDecision(
            target_agent="inventory_specialist",
            selected_model="gemini-3-flash-preview",
            complexity=TaskComplexity.FAST_LOOKUP,
            rationale="Valid rationale string.",
            unexpected_field="not_allowed",  # type: ignore[call-arg]
        )
