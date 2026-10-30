"""Unit tests for `app.prompts.constitution`."""

from __future__ import annotations

import pytest
from app.prompts.constitution import (
    AGENT_CONSTITUTION_MAP,
    COORDINATOR_CONSTITUTION,
    DEAL_DESK_CLOSER_CONSTITUTION,
    GLOBAL_SAFETY_CONSTITUTION,
    INVENTORY_SPECIALIST_CONSTITUTION,
    TOP_OFFERS_STRATEGIST_CONSTITUTION,
    get_agent_constitution,
)


def test_all_constitutions_contain_persona_and_domain_rules() -> None:
    """Verifies every agent constitution defines persona, domain rules, and Regulation Z."""
    constitutions = [
        GLOBAL_SAFETY_CONSTITUTION,
        COORDINATOR_CONSTITUTION,
        INVENTORY_SPECIALIST_CONSTITUTION,
        TOP_OFFERS_STRATEGIST_CONSTITUTION,
        DEAL_DESK_CLOSER_CONSTITUTION,
    ]
    for prompt in constitutions:
        assert "AutoDeal Copilot" in prompt
        assert "PERSONA & ROLE" in prompt
        assert "AUTOMOTIVE RETAIL DOMAIN KNOWLEDGE" in prompt
        assert "Truth-in-Lending Act (Regulation Z)" in prompt
        assert "Tier 1 (Super Prime, FICO 740+)" in prompt
        assert "Tier 2 (Prime, FICO 680-739)" in prompt
        assert "Tier 3 (Near Prime, FICO 620-679)" in prompt
        assert "Tier 4 (Subprime, FICO <620)" in prompt
        assert "invoice_price + $500" in prompt


def test_all_constitutions_enforce_top_three_offer_tiers_and_safety_constraints() -> (
    None
):
    """Verifies Top-3 Offer tiers, PII redaction, and HITL rules are present in every prompt."""
    for role_name, prompt in AGENT_CONSTITUTION_MAP.items():
        assert "Offer #1 — Best Overall Match" in prompt, (
            f"Missing Offer #1 in {role_name}"
        )
        assert "Offer #2 — Budget / Value Pick" in prompt, (
            f"Missing Offer #2 in {role_name}"
        )
        assert "Offer #3 — Premium / Upgrade Pick" in prompt, (
            f"Missing Offer #3 in {role_name}"
        )
        assert "[REDACTED_SSN]" in prompt
        assert "apply_manager_special_discount" in prompt
        assert "MGR-APPROVED-" in prompt


def test_get_agent_constitution_resolves_known_roles_and_rejects_unknown() -> None:
    """Tests `get_agent_constitution` lookup and `KeyError` on invalid role."""
    assert get_agent_constitution("dealership_coordinator") == COORDINATOR_CONSTITUTION
    assert (
        get_agent_constitution(" INVENTORY_SPECIALIST ")
        == INVENTORY_SPECIALIST_CONSTITUTION
    )
    assert (
        get_agent_constitution("top_offers_strategist")
        == TOP_OFFERS_STRATEGIST_CONSTITUTION
    )
    assert get_agent_constitution("deal_desk_closer") == DEAL_DESK_CLOSER_CONSTITUTION

    with pytest.raises(KeyError, match="Unknown agent_role"):
        get_agent_constitution("non_existent_agent")
