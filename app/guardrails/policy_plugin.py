"""Security, Policy, PII Redaction, and Self-Evaluation Guardrail Plugin for ADK.

Implements both:
1. `PolicyAndPiiPlugin(BasePlugin)` for runner-wide interception across all agents
   and tools in Google ADK.
2. Standalone callback functions (`before_model_guardrail_callback`,
   `after_model_self_eval_callback`, `before_tool_policy_callback`,
   `after_tool_audit_callback`) that can be attached directly to `LlmAgent`
   instances.

Enforces:
- Prompt-injection and policy-bypass detection before model calls
- Automatic PII redaction on incoming `LlmRequest` contents, tool arguments, and
  outgoing `LlmResponse` / tool results
- Dealership floor-margin and Truth-in-Lending (Regulation Z) compliance
- Automated output self-evaluation (`evaluate_agent_response`)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.tools.schemas import build_guided_error_response

_PROMPT_INJECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "PROMPT_INJECTION_IGNORE_INSTRUCTIONS",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|system|above)\s+"
            r"(?:instructions|prompts|rules|constraints|guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "POLICY_BYPASS_FLOOR_MARGIN",
        re.compile(
            r"\b(?:ignore|bypass|disable|skip|override)\s+(?:the\s+)?(?:dealer(?:ship)?\s+)?"
            r"(?:margin|floor\s*margin|invoice|manager\s+approval|hitl)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "UNAUTHORIZED_BELOW_COST_SALE",
        re.compile(
            r"\b(?:sell|give|offer)\s+(?:me\s+)?(?:the\s+|this\s+|any\s+)?(?:car|vehicle|suv|truck|sedan)\s+"
            r"for\s+\$?\s*(?:0|1|1\.00|0\.00|free)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SYSTEM_PROMPT_EXFILTRATION",
        re.compile(
            r"\b(?:reveal|print|output|show|dump)\s+(?:your\s+|the\s+)?(?:system\s+prompt|hidden\s+instructions|secret\s+keys)\b",
            re.IGNORECASE,
        ),
    ),
]

_DECEPTIVE_LENDING_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "DECEPTIVE_ZERO_APR_GUARANTEE",
        re.compile(
            r"\b0(?:\.0+)?%\s*apr\s+guaranteed\s+for\s+all(?:\s+credit\s+scores)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DECEPTIVE_NO_CREDIT_CHECK",
        re.compile(
            r"\bno\s+credit\s+check\s+required\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DECEPTIVE_GUARANTEED_APPROVAL",
        re.compile(
            r"\bguaranteed\s+(?:loan\s+|financing\s+)?approval\s+without\s+(?:income|credit)\b",
            re.IGNORECASE,
        ),
    ),
]

_NON_REDACTED_ID_FIELDS = frozenset(
    {
        "vehicle_identifier",
        "best_match_vin",
        "value_pick_vin",
        "upgrade_pick_vin",
        "vin",
        "stock_number",
        "manager_approval_token",
    }
)

POLICY_BLOCK_MESSAGE = (
    "I cannot fulfill requests to bypass dealership pricing guardrails, override "
    "Sales Manager approval policies, sell vehicles below authorized floor margins, "
    "or disclose internal system instructions. I would be happy to help you compare "
    "vehicles from our inventory and build compliant Top-3 financing or lease offers."
)


@traced_operation("autodeal.guardrails.inspect_user_prompt")
def inspect_user_prompt(
    prompt_text: str,
    redactor: Optional[PIIRedactor] = None,
) -> Dict[str, Any]:
    """Inspects user input for adversarial prompt injection and scrubs customer PII.

    Args:
        prompt_text: Raw user or salesperson text message.
        redactor: Optional `PIIRedactor` instance (defaults to `DEFAULT_PII_REDACTOR`).

    Returns:
        Dict[str, Any]: Inspection result containing `allowed`, `is_blocked`,
        `violation_code`, `matched_text`, `contained_pii`, `sanitized_text`, and
        `policy_message`.
    """
    active_redactor = redactor or DEFAULT_PII_REDACTOR
    text = prompt_text if isinstance(prompt_text, str) else str(prompt_text or "")
    contained_pii = active_redactor.contains_pii(text)
    sanitized_text = active_redactor.redact_text(text)

    for violation_code, pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return {
                "allowed": False,
                "is_blocked": True,
                "violation_code": violation_code,
                "matched_text": match.group(0),
                "contained_pii": contained_pii,
                "sanitized_text": sanitized_text,
                "policy_message": POLICY_BLOCK_MESSAGE,
            }

    return {
        "allowed": True,
        "is_blocked": False,
        "violation_code": None,
        "matched_text": None,
        "contained_pii": contained_pii,
        "sanitized_text": sanitized_text,
        "policy_message": None,
    }


@traced_operation("autodeal.guardrails.evaluate_agent_response")
def evaluate_agent_response(
    response_text: str,
    redactor: Optional[PIIRedactor] = None,
) -> Dict[str, Any]:
    """Performs automated self-evaluation on an agent response for safety and quality.

    Verifies that the response:
    1. Contains zero unredacted customer PII (`no_unredacted_pii`).
    2. Contains zero deceptive or non-compliant Truth-in-Lending claims
       (`truth_in_lending_compliant`).
    3. Does not promise unauthorized $0/$1 vehicle giveaways (`floor_policy_compliant`).

    Args:
        response_text: Model-generated text to evaluate.
        redactor: Optional `PIIRedactor` instance.

    Returns:
        Dict[str, Any]: Self-evaluation report with `passed` (bool), `score`
        (0.0 to 1.0), `checks` (dict of check names to booleans), `issues`
        (list of failure descriptions), and `sanitized_text`.
    """
    active_redactor = redactor or DEFAULT_PII_REDACTOR
    text = response_text if isinstance(response_text, str) else str(response_text or "")

    has_pii = active_redactor.contains_pii(text)
    sanitized_text = active_redactor.redact_text(text)

    issues: List[str] = []
    if has_pii:
        issues.append("Detected unredacted customer PII in model output.")

    deceptive_matches: List[str] = []
    for code, pattern in _DECEPTIVE_LENDING_PATTERNS:
        if pattern.search(text) is not None:
            deceptive_matches.append(code)
            issues.append(
                f"Truth-in-Lending violation ({code}): response contains prohibited lending guarantee."
            )
            sanitized_text = pattern.sub(
                "[ financing subject to credit approval and Regulation Z terms ]",
                sanitized_text,
            )

    unauthorized_sale = (
        re.search(
            r"\b(?:selling|sold|approved)\s+(?:the\s+)?(?:car|vehicle)\s+for\s+\$?\s*(?:0|1)\b",
            text,
            re.IGNORECASE,
        )
        is not None
    )
    if unauthorized_sale:
        issues.append(
            "Floor margin violation: response promises an unauthorized $0/$1 vehicle sale."
        )

    checks = {
        "no_unredacted_pii": not has_pii,
        "truth_in_lending_compliant": len(deceptive_matches) == 0,
        "floor_policy_compliant": not unauthorized_sale,
    }
    passed_count = sum(1 for ok in checks.values() if ok)
    score = round(passed_count / len(checks), 2)
    passed = all(checks.values())

    return {
        "passed": passed,
        "score": score,
        "checks": checks,
        "issues": issues,
        "sanitized_text": sanitized_text,
    }


def _build_guardrail_llm_response(message: str) -> LlmResponse:
    """Constructs an ADK `LlmResponse` containing a guardrail block message."""
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        ),
        turn_complete=True,
    )


@traced_operation("autodeal.guardrails.before_model_guardrail_callback")
def before_model_guardrail_callback(
    callback_context: Optional[CallbackContext] = None,
    llm_request: Optional[LlmRequest] = None,
    **kwargs: Any,
) -> Optional[LlmResponse]:
    """ADK `before_model_callback` that blocks prompt injection and scrubs PII.

    Inspects all user text parts in `llm_request.contents`. Any detected customer
    PII is scrubbed in-place via `DEFAULT_PII_REDACTOR.redact_text` before the
    request is sent to the Gemini model. If a prompt-injection or policy-bypass
    attempt is detected, short-circuits the model call and returns a polite
    policy-compliant `LlmResponse`.

    Args:
        callback_context: ADK `CallbackContext` for the active agent turn.
        llm_request: Outgoing `LlmRequest` about to be sent to Gemini.
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[LlmResponse]: `None` when the request is safe to proceed, or an
        `LlmResponse` blocking the request when a policy violation is detected.
    """
    _ = kwargs
    if isinstance(callback_context, LlmRequest) and llm_request is None:
        llm_request = callback_context
        callback_context = None

    agent_name = (
        getattr(callback_context, "agent_name", None) or "dealership_coordinator"
    )
    log_intent(
        action_name="before_model_guardrail_callback",
        intended_arguments={"agent_name": agent_name},
        agent_name=agent_name,
    )

    if llm_request is None or not getattr(llm_request, "contents", None):
        log_outcome(
            action_name="before_model_guardrail_callback",
            status="allowed",
            actual_outcome={"pii_redacted": False, "blocked": False},
            agent_name=agent_name,
        )
        return None

    pii_redacted_any = False
    for content in llm_request.contents or []:
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if not isinstance(part_text, str) or not part_text:
                continue

            inspection = inspect_user_prompt(part_text)
            if inspection["contained_pii"]:
                part.text = inspection["sanitized_text"]
                pii_redacted_any = True

            if inspection["is_blocked"]:
                block_msg = str(inspection["policy_message"] or POLICY_BLOCK_MESSAGE)
                log_outcome(
                    action_name="before_model_guardrail_callback",
                    status="blocked_by_guardrail",
                    actual_outcome={
                        "violation_code": inspection["violation_code"],
                        "matched_text": inspection["matched_text"],
                        "pii_redacted": pii_redacted_any,
                    },
                    agent_name=agent_name,
                )
                return _build_guardrail_llm_response(block_msg)

    log_outcome(
        action_name="before_model_guardrail_callback",
        status="allowed",
        actual_outcome={"pii_redacted": pii_redacted_any, "blocked": False},
        agent_name=agent_name,
    )
    return None


@traced_operation("autodeal.guardrails.after_model_self_eval_callback")
def after_model_self_eval_callback(
    callback_context: Optional[CallbackContext] = None,
    llm_response: Optional[LlmResponse] = None,
    **kwargs: Any,
) -> Optional[LlmResponse]:
    """ADK `after_model_callback` performing PII scrubbing and self-evaluation.

    Audits the model's `LlmResponse` for accidental PII leakage or deceptive
    Truth-in-Lending claims. Redacts any detected issues in-place and records the
    self-evaluation score in structured logs.

    Args:
        callback_context: ADK `CallbackContext` for the active agent turn.
        llm_response: `LlmResponse` returned by the Gemini model.
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[LlmResponse]: `None` if the response required no replacement, or
        a sanitized `LlmResponse` if a policy correction was applied.
    """
    _ = kwargs
    if isinstance(callback_context, LlmResponse) and llm_response is None:
        llm_response = callback_context
        callback_context = None

    agent_name = (
        getattr(callback_context, "agent_name", None) or "dealership_coordinator"
    )
    log_intent(
        action_name="after_model_self_eval_callback",
        intended_arguments={"agent_name": agent_name},
        agent_name=agent_name,
    )

    if llm_response is None or getattr(llm_response, "content", None) is None:
        log_outcome(
            action_name="after_model_self_eval_callback",
            status="passed_self_eval",
            actual_outcome={"passed": True, "score": 1.0, "issues": []},
            agent_name=agent_name,
        )
        return None

    parts = getattr(llm_response.content, "parts", None) or []
    all_issues: List[str] = []
    min_score = 1.0
    modified = False

    for part in parts:
        part_text = getattr(part, "text", None)
        if not isinstance(part_text, str) or not part_text:
            continue

        eval_report = evaluate_agent_response(part_text)
        min_score = min(min_score, float(eval_report["score"]))
        if not eval_report["passed"]:
            all_issues.extend(eval_report["issues"])
            part.text = eval_report["sanitized_text"]
            modified = True

    status = "modified_by_self_eval" if modified else "passed_self_eval"
    log_outcome(
        action_name="after_model_self_eval_callback",
        status=status,
        actual_outcome={
            "passed": not modified,
            "score": min_score,
            "issues": all_issues,
        },
        agent_name=agent_name,
    )
    if modified:
        return llm_response
    return None


@traced_operation("autodeal.guardrails.before_tool_policy_callback")
def before_tool_policy_callback(
    tool: Optional[BaseTool] = None,
    args: Optional[Dict[str, Any]] = None,
    tool_context: Optional[ToolContext] = None,
    *,
    tool_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """ADK `before_tool_callback` enforcing parameter safety and PII scrubbing.

    Audits tool arguments prior to execution, scrubs customer PII from free-text
    fields (e.g., `customer_name`, `custom_notes`, `reason`), and blocks negative
    pricing parameters or prompt-injection payloads.

    Args:
        tool: ADK tool instance or function being invoked.
        args: Positional dictionary of tool arguments (used by `LlmAgent` callbacks).
        tool_context: ADK `ToolContext` for the invocation.
        tool_args: Keyword dictionary of tool arguments (used by `BasePlugin`).
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[Dict[str, Any]]: `None` if tool execution may proceed, or a
        structured `ToolErrorResponse` dictionary to short-circuit invalid calls.
    """
    _ = kwargs
    active_args = (
        args if args is not None else (tool_args if tool_args is not None else {})
    )
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
    agent_name = getattr(tool_context, "agent_name", None) or "dealership_coordinator"

    log_intent(
        action_name=f"before_tool:{tool_name}",
        intended_arguments=dict(active_args),
        agent_name=agent_name,
    )

    numeric_positive_fields = (
        "max_price",
        "proposed_selling_price",
        "requested_discount_amount",
    )
    for field_name in numeric_positive_fields:
        val = active_args.get(field_name)
        if isinstance(val, (int, float)) and val <= 0:
            blocked_response = build_guided_error_response(
                error_code="POLICY_BLOCKED_TOOL_CALL",
                message=(
                    f"Guardrail blocked `{tool_name}`: `{field_name}` must be strictly "
                    f"positive (received {val})."
                ),
                recovery_suggestion=(
                    f"Pass a positive dollar amount (> 0) for `{field_name}` and retry."
                ),
            )
            log_outcome(
                action_name=f"before_tool:{tool_name}",
                status="blocked_by_guardrail",
                actual_outcome=blocked_response,
                agent_name=agent_name,
            )
            return blocked_response

    for key, value in list(active_args.items()):
        if isinstance(value, str) and key not in _NON_REDACTED_ID_FIELDS:
            inspection = inspect_user_prompt(value)
            if inspection["is_blocked"]:
                blocked_response = build_guided_error_response(
                    error_code="POLICY_BLOCKED_TOOL_CALL",
                    message=(
                        f"Guardrail blocked `{tool_name}` due to policy violation "
                        f"({inspection['violation_code']}) in argument `{key}`."
                    ),
                    recovery_suggestion=(
                        "Remove adversarial instructions or policy-bypass directives from tool arguments."
                    ),
                )
                log_outcome(
                    action_name=f"before_tool:{tool_name}",
                    status="blocked_by_guardrail",
                    actual_outcome=blocked_response,
                    agent_name=agent_name,
                )
                return blocked_response
            if inspection["contained_pii"]:
                active_args[key] = inspection["sanitized_text"]

    log_outcome(
        action_name=f"before_tool:{tool_name}",
        status="allowed",
        actual_outcome={"tool_name": tool_name},
        agent_name=agent_name,
    )
    return None


@traced_operation("autodeal.guardrails.after_tool_audit_callback")
def after_tool_audit_callback(
    tool: Optional[BaseTool] = None,
    args: Optional[Dict[str, Any]] = None,
    tool_context: Optional[ToolContext] = None,
    tool_response: Optional[Dict[str, Any]] = None,
    *,
    tool_args: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """ADK `after_tool_callback` that audits tool results and scrubs residual PII.

    Args:
        tool: ADK tool instance or function that completed execution.
        args: Tool arguments passed to the tool.
        tool_context: ADK `ToolContext` for the invocation.
        tool_response: Positional tool output dictionary (used by `LlmAgent`).
        tool_args: Keyword tool arguments (used by `BasePlugin`).
        result: Keyword tool output dictionary (used by `BasePlugin`).
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[Dict[str, Any]]: Sanitized dictionary if PII was scrubbed from
        the tool result, or `None` to keep the original result unchanged.
    """
    _ = (args, tool_args, kwargs)
    active_result = tool_response if tool_response is not None else result
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
    agent_name = getattr(tool_context, "agent_name", None) or "dealership_coordinator"

    if not isinstance(active_result, dict):
        log_outcome(
            action_name=f"after_tool:{tool_name}",
            status="completed",
            actual_outcome={"tool_name": tool_name},
            agent_name=agent_name,
        )
        return None

    sanitized_result: Dict[str, Any] = {}
    modified = False
    for key, value in active_result.items():
        if key in _NON_REDACTED_ID_FIELDS:
            sanitized_result[key] = value
        else:
            scrubbed = DEFAULT_PII_REDACTOR.redact_data(value)
            if scrubbed != value:
                modified = True
            sanitized_result[key] = scrubbed

    log_outcome(
        action_name=f"after_tool:{tool_name}",
        status=str(sanitized_result.get("status", "completed")),
        actual_outcome={
            "tool_name": tool_name,
            "result_status": sanitized_result.get("status", "completed"),
            "pii_scrubbed": modified,
        },
        agent_name=agent_name,
    )

    if modified:
        active_result.clear()
        active_result.update(sanitized_result)
        return sanitized_result
    return None


class PolicyAndPiiPlugin(BasePlugin):
    """Runner-level ADK Plugin enforcing PII redaction, safety guardrails, and self-eval.

    Inherits from `google.adk.plugins.base_plugin.BasePlugin` and intercepts:
    - `on_user_message_callback`: Redacts customer PII from incoming user messages.
    - `before_model_callback`: Blocks prompt injection and scrubs PII in `LlmRequest`.
    - `after_model_callback`: Runs self-evaluation (`evaluate_agent_response`) on
      model outputs to prevent PII leakage and deceptive lending claims.
    - `before_tool_callback`: Validates tool arguments and scrubs PII.
    - `after_tool_callback`: Audits tool outputs and scrubs residual PII.
    """

    def __init__(
        self,
        name: str = "autodeal_policy_and_pii_plugin",
        redactor: Optional[PIIRedactor] = None,
    ) -> None:
        super().__init__(name=name)
        self.redactor = redactor or DEFAULT_PII_REDACTOR

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Scrubs customer PII from incoming user messages at the start of an invocation."""
        _ = invocation_context
        if user_message is None or not getattr(user_message, "parts", None):
            return None

        modified = False
        for part in user_message.parts or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and self.redactor.contains_pii(part_text):
                part.text = self.redactor.redact_text(part_text)
                modified = True

        return user_message if modified else None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        """Intercepts model requests to block prompt injection and redact PII."""
        return before_model_guardrail_callback(
            callback_context=callback_context,
            llm_request=llm_request,
        )

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """Evaluates model responses for PII leakage and Truth-in-Lending compliance."""
        return after_model_self_eval_callback(
            callback_context=callback_context,
            llm_response=llm_response,
        )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[Dict[str, Any]]:
        """Validates tool arguments and scrubs PII prior to tool execution."""
        return before_tool_policy_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
        )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Audits tool results and scrubs any residual PII after execution."""
        return after_tool_audit_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            result=result,
        )


DEFAULT_POLICY_PLUGIN = PolicyAndPiiPlugin()
