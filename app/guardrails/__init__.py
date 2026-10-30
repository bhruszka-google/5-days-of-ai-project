"""Guardrails, PII Redaction, Policy Plugin, and HITL Hooks for AutoDeal Copilot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor

if TYPE_CHECKING:
    from app.guardrails.hitl import (
        apply_manager_special_discount,
        build_manager_discount_tool,
        is_valid_manager_approval_token,
        requires_sales_manager_approval,
    )
    from app.guardrails.policy_plugin import (
        DEFAULT_POLICY_PLUGIN,
        POLICY_BLOCK_MESSAGE,
        PolicyAndPiiPlugin,
        after_model_self_eval_callback,
        after_tool_audit_callback,
        before_model_guardrail_callback,
        before_tool_policy_callback,
        evaluate_agent_response,
        inspect_user_prompt,
    )

__all__ = [
    "DEFAULT_PII_REDACTOR",
    "DEFAULT_POLICY_PLUGIN",
    "PIIRedactor",
    "POLICY_BLOCK_MESSAGE",
    "PolicyAndPiiPlugin",
    "after_model_self_eval_callback",
    "after_tool_audit_callback",
    "apply_manager_special_discount",
    "before_model_guardrail_callback",
    "before_tool_policy_callback",
    "build_manager_discount_tool",
    "evaluate_agent_response",
    "inspect_user_prompt",
    "is_valid_manager_approval_token",
    "requires_sales_manager_approval",
]

_HITL_EXPORTS = {
    "apply_manager_special_discount",
    "build_manager_discount_tool",
    "is_valid_manager_approval_token",
    "requires_sales_manager_approval",
}

_POLICY_EXPORTS = {
    "DEFAULT_POLICY_PLUGIN",
    "POLICY_BLOCK_MESSAGE",
    "PolicyAndPiiPlugin",
    "after_model_self_eval_callback",
    "after_tool_audit_callback",
    "before_model_guardrail_callback",
    "before_tool_policy_callback",
    "evaluate_agent_response",
    "inspect_user_prompt",
}


def __getattr__(name: str) -> Any:
    """Lazily resolves HITL and Policy Plugin symbols to avoid circular imports."""
    if name in _HITL_EXPORTS:
        from app.guardrails import hitl

        return getattr(hitl, name)
    if name in _POLICY_EXPORTS:
        from app.guardrails import policy_plugin

        return getattr(policy_plugin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
