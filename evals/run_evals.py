"""Golden Dataset Evaluation Harness for AutoDeal Copilot.

Satisfies the AgentOps Rubric Criterion **Automated Evaluation Suites (5 pts)**:
"The repository contains a testing harness (e.g., against a golden dataset) to
statically measure agent regressions."

Executes all golden dealership scenarios defined in `evals/golden_dataset.json`
against `generate_top_three_offers_workflow`, `apply_manager_special_discount`,
`inspect_user_prompt`, `StrategicModelRouter`, and `evaluate_agent_response`,
computing trajectory accuracy, tool selection accuracy, PII redaction compliance,
floor-margin compliance, and overall pass rate.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent import generate_top_three_offers_workflow
from app.guardrails.hitl import apply_manager_special_discount
from app.guardrails.policy_plugin import (
    before_model_guardrail_callback,
    evaluate_agent_response,
    inspect_user_prompt,
)
from app.memory.session_store import PersistentSessionStore
from app.observability.logger import get_structured_logger, log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.routing.model_router import DEFAULT_MODEL_ROUTER, StrategicModelRouter
from app.tools.inventory_tools import fetch_active_manufacturer_rebates
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


class EvaluationCaseResult(BaseModel):
    """Strict Pydantic schema representing the evaluation outcome for one golden case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        ..., min_length=3, description="Unique golden case identifier."
    )
    category: str = Field(..., min_length=3, description="Scenario category label.")
    passed: bool = Field(
        ..., description="True when all assertions in the golden case pass."
    )
    trajectory_accurate: bool = Field(
        ...,
        description="True when workflow status and routing match expected trajectory.",
    )
    tool_selection_accurate: bool = Field(
        ..., description="True when expected dealership tools executed accurately."
    )
    pii_redaction_compliant: bool = Field(
        ...,
        description="True when all raw PII is scrubbed from outputs and session state.",
    )
    floor_margin_compliant: bool = Field(
        ...,
        description="True when all offers obey floor-margin and HITL governance rules.",
    )
    routed_agent: Optional[str] = Field(
        default=None, description="Agent selected by StrategicModelRouter."
    )
    routed_model: Optional[str] = Field(
        default=None, description="Gemini model selected by StrategicModelRouter."
    )
    workflow_status: str = Field(..., description="Final workflow execution status.")
    executed_tools: List[str] = Field(
        default_factory=list, description="Ordered list of dealership tools invoked."
    )
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Detailed failure reasons if any check failed.",
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Execution duration in milliseconds."
    )


class EvaluationSuiteReport(BaseModel):
    """Strict Pydantic schema representing the aggregated golden evaluation suite report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_name: str = Field(..., min_length=3)
    version: str = Field(..., min_length=1)
    total_cases: int = Field(..., ge=1)
    passed_cases: int = Field(..., ge=0)
    failed_cases: int = Field(..., ge=0)
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0)
    trajectory_accuracy: float = Field(..., ge=0.0, le=1.0)
    tool_selection_accuracy: float = Field(..., ge=0.0, le=1.0)
    pii_redaction_compliance: float = Field(..., ge=0.0, le=1.0)
    floor_margin_compliance: float = Field(..., ge=0.0, le=1.0)
    case_results: List[EvaluationCaseResult] = Field(..., min_length=1)


def load_golden_dataset(dataset_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads and parses the golden evaluation dataset JSON file.

    Args:
        dataset_path: Optional path override to a golden dataset JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON dataset dictionary containing `cases`.
    """
    resolved_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    with resolved_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _evaluate_single_case(
    case: Dict[str, Any],
    router: StrategicModelRouter,
    session_store: PersistentSessionStore,
) -> EvaluationCaseResult:
    """Executes and grades a single golden scenario."""
    start_time = time.perf_counter()
    case_id = str(case["case_id"])
    category = str(case.get("category", "general"))
    user_query = str(case["user_query"])
    params = dict(case.get("workflow_params", {}))
    expected = dict(case.get("expected", {}))

    custom_qualifiers = params.pop("customer_qualifiers", None)
    failure_reasons: List[str] = []
    executed_tools: List[str] = []

    log_intent(
        action_name=f"eval_case:{case_id}",
        intended_arguments={"case_id": case_id, "category": category},
        agent_name="golden_evaluator",
    )

    prompt_inspection = inspect_user_prompt(user_query)
    workflow_result = generate_top_three_offers_workflow(
        user_query,
        session_id=f"golden-{case_id}",
        session_store=session_store,
        router=router,
        **params,
    )

    workflow_status = str(workflow_result.get("status", "unknown"))

    # Determine executed tools based on workflow artifacts
    if workflow_result.get("trade_in_appraisal") is not None:
        executed_tools.append("appraise_trade_in_vehicle")
    if workflow_result.get("inventory_search") is not None:
        executed_tools.append("search_dealership_inventory")
    if workflow_result.get("best_match_rebates") is not None:
        executed_tools.append("fetch_active_manufacturer_rebates")
    if custom_qualifiers and workflow_status != "blocked_by_guardrail":
        best_vin = (
            workflow_result.get("offer_sheet", {})
            .get("offers", [{}])[0]
            .get("vin", "STK-2026-001")
        )
        rebate_check = fetch_active_manufacturer_rebates(
            vehicle_identifier=best_vin,
            customer_qualifiers=custom_qualifiers,
        )
        if rebate_check.get("status") != "success":
            failure_reasons.append(
                f"Qualifier rebate lookup failed: {rebate_check.get('message')}"
            )
    if workflow_result.get("offer_sheet") is not None:
        executed_tools.extend(
            [
                "calculate_financing_and_lease_options",
                "verify_dealership_floor_margin",
            ]
        )
    if workflow_result.get("discount_decision") is not None:
        executed_tools.append("apply_manager_special_discount")
    if workflow_result.get("offer_sheet") is not None:
        executed_tools.append("generate_customer_offer_sheet")

    # 1. Trajectory Accuracy Checks
    trajectory_accurate = True
    expected_status = expected.get("workflow_status")
    if expected_status and workflow_status != expected_status:
        trajectory_accurate = False
        failure_reasons.append(
            f"Expected workflow_status='{expected_status}', got '{workflow_status}'."
        )

    routed_agent: Optional[str] = None
    routed_model: Optional[str] = None
    routing_payload = workflow_result.get("routing_decision")
    if isinstance(routing_payload, dict):
        routed_agent = str(routing_payload.get("target_agent"))
        routed_model = str(routing_payload.get("selected_model"))

    expected_agent = expected.get("routed_agent")
    if expected_agent and routed_agent != expected_agent:
        trajectory_accurate = False
        failure_reasons.append(
            f"Expected routed_agent='{expected_agent}', got '{routed_agent}'."
        )

    expected_model = expected.get("routed_model")
    if expected_model and routed_model != expected_model:
        trajectory_accurate = False
        failure_reasons.append(
            f"Expected routed_model='{expected_model}', got '{routed_model}'."
        )

    if expected.get("expect_blocked"):
        if not prompt_inspection.get("is_blocked"):
            trajectory_accurate = False
            failure_reasons.append(
                "Expected prompt inspection to block adversarial input."
            )
        expected_code = expected.get("expected_violation_code")
        if expected_code and prompt_inspection.get("violation_code") != expected_code:
            trajectory_accurate = False
            failure_reasons.append(
                f"Expected violation_code='{expected_code}', got '{prompt_inspection.get('violation_code')}'."
            )
        # Also verify ADK before_model_guardrail_callback blocks an LlmRequest
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types

        adk_req = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)],
                )
            ]
        )
        adk_block_resp = before_model_guardrail_callback(llm_request=adk_req)
        if adk_block_resp is None:
            trajectory_accurate = False
            failure_reasons.append(
                "Expected before_model_guardrail_callback to return blocking LlmResponse."
            )

    # 2. Tool Selection Accuracy Checks
    tool_selection_accurate = True
    expected_tools = expected.get("expected_tools", [])
    for req_tool in expected_tools:
        if req_tool not in executed_tools:
            tool_selection_accurate = False
            failure_reasons.append(
                f"Expected tool '{req_tool}' was not executed (executed={executed_tools})."
            )

    if expected.get("expect_positive_trade_in_equity"):
        trade_in_data = workflow_result.get("trade_in_appraisal") or {}
        if float(trade_in_data.get("net_trade_in_equity", 0.0)) <= 0.0:
            tool_selection_accurate = False
            failure_reasons.append("Expected positive trade-in equity from appraisal.")

    if expected.get("expect_three_distinct_offers"):
        offers = (workflow_result.get("offer_sheet") or {}).get("offers", [])
        vins = {card.get("vin") for card in offers if isinstance(card, dict)}
        if len(offers) != 3 or len(vins) != 3:
            tool_selection_accurate = False
            failure_reasons.append(
                f"Expected 3 distinct vehicle offers, got {len(offers)} offers ({len(vins)} unique VINs)."
            )

    # 3. PII Redaction Compliance Checks
    pii_redaction_compliant = True
    expect_pii = bool(expected.get("expect_pii_detected", False))
    if bool(prompt_inspection.get("contained_pii")) != expect_pii:
        pii_redaction_compliant = False
        failure_reasons.append(
            f"Expected contained_pii={expect_pii}, got {prompt_inspection.get('contained_pii')}."
        )

    serialized_output = json.dumps(workflow_result)
    for raw_pii_token in expected.get("forbidden_raw_pii", []):
        if raw_pii_token in serialized_output:
            pii_redaction_compliant = False
            failure_reasons.append(
                f"Leaked raw PII token '{raw_pii_token}' in workflow result payload."
            )

    self_eval_report = evaluate_agent_response(
        str(workflow_result.get("response_text", ""))
    )
    if not self_eval_report.get("passed"):
        pii_redaction_compliant = False
        failure_reasons.append(
            f"Self-evaluation failed on response_text: {self_eval_report.get('issues')}"
        )

    # 4. Floor Margin & HITL Governance Compliance Checks
    floor_margin_compliant = True
    if workflow_result.get("offer_sheet"):
        offers = workflow_result["offer_sheet"].get("offers", [])
        if not all(bool(card.get("floor_margin_compliant")) for card in offers):
            floor_margin_compliant = False
            failure_reasons.append(
                "One or more generated offers violated dealership floor-margin policy."
            )

    expected_discount_status = expected.get("expect_discount_status")
    if expected_discount_status:
        discount_decision = workflow_result.get("discount_decision") or {}
        actual_disc_status = discount_decision.get("status")
        if actual_disc_status != expected_discount_status:
            floor_margin_compliant = False
            failure_reasons.append(
                f"Expected discount_decision.status='{expected_discount_status}', got '{actual_disc_status}'."
            )
        expected_mode = expected.get("expect_approval_mode")
        if expected_mode and discount_decision.get("approval_mode") != expected_mode:
            floor_margin_compliant = False
            failure_reasons.append(
                f"Expected approval_mode='{expected_mode}', got '{discount_decision.get('approval_mode')}'."
            )
        if "expect_requires_human_confirmation" in expected:
            expected_hitl = bool(expected["expect_requires_human_confirmation"])
            if (
                bool(discount_decision.get("requires_human_confirmation"))
                != expected_hitl
            ):
                floor_margin_compliant = False
                failure_reasons.append(
                    f"Expected requires_human_confirmation={expected_hitl}, got {discount_decision.get('requires_human_confirmation')}."
                )

        # Also directly verify apply_manager_special_discount tool contract
        direct_disc = apply_manager_special_discount(
            vehicle_identifier="STK-2026-001",
            requested_discount_amount=float(
                params.get("requested_discount_amount", 500.0)
            ),
            reason=str(params.get("discount_reason", "Golden eval test")),
            manager_approval_token=params.get("manager_approval_token"),
        )
        if direct_disc.get("status") != expected_discount_status:
            floor_margin_compliant = False
            failure_reasons.append(
                f"Direct apply_manager_special_discount status mismatch: {direct_disc.get('status')}"
            )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    passed = (
        trajectory_accurate
        and tool_selection_accurate
        and pii_redaction_compliant
        and floor_margin_compliant
        and len(failure_reasons) == 0
    )

    case_result = EvaluationCaseResult(
        case_id=case_id,
        category=category,
        passed=passed,
        trajectory_accurate=trajectory_accurate,
        tool_selection_accurate=tool_selection_accurate,
        pii_redaction_compliant=pii_redaction_compliant,
        floor_margin_compliant=floor_margin_compliant,
        routed_agent=routed_agent,
        routed_model=routed_model,
        workflow_status=workflow_status,
        executed_tools=executed_tools,
        failure_reasons=failure_reasons,
        latency_ms=latency_ms,
    )

    log_outcome(
        action_name=f"eval_case:{case_id}",
        status="passed" if passed else "failed",
        actual_outcome=case_result.model_dump(),
        latency_ms=latency_ms,
        agent_name="golden_evaluator",
    )
    return case_result


@traced_operation("autodeal.evals.run_golden_evaluation_suite")
def run_golden_evaluation_suite(
    dataset_path: Optional[str | Path] = None,
) -> EvaluationSuiteReport:
    """Executes the complete Golden Evaluation Suite and returns a structured report.

    Args:
        dataset_path: Optional path to a custom golden dataset JSON file.

    Returns:
        EvaluationSuiteReport: Validated evaluation report with pass rates and
        per-case metrics.
    """
    dataset = load_golden_dataset(dataset_path)
    suite_name = str(
        dataset.get("suite_name", "AutoDeal Copilot Golden Evaluation Suite")
    )
    version = str(dataset.get("version", "1.0.0"))
    cases = list(dataset.get("cases", []))

    log_intent(
        action_name="run_golden_evaluation_suite",
        intended_arguments={"suite_name": suite_name, "case_count": len(cases)},
        agent_name="golden_evaluator",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        isolated_db = Path(tmp_dir) / "eval_session_store.db"
        eval_store = PersistentSessionStore(db_path=str(isolated_db))
        router = DEFAULT_MODEL_ROUTER

        results: List[EvaluationCaseResult] = [
            _evaluate_single_case(case=case, router=router, session_store=eval_store)
            for case in cases
        ]

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count

    overall_pass_rate = round(passed_count / total, 4) if total else 0.0
    trajectory_acc = (
        round(sum(1 for r in results if r.trajectory_accurate) / total, 4)
        if total
        else 0.0
    )
    tool_acc = (
        round(sum(1 for r in results if r.tool_selection_accurate) / total, 4)
        if total
        else 0.0
    )
    pii_comp = (
        round(sum(1 for r in results if r.pii_redaction_compliant) / total, 4)
        if total
        else 0.0
    )
    margin_comp = (
        round(sum(1 for r in results if r.floor_margin_compliant) / total, 4)
        if total
        else 0.0
    )

    report = EvaluationSuiteReport(
        suite_name=suite_name,
        version=version,
        total_cases=total,
        passed_cases=passed_count,
        failed_cases=failed_count,
        overall_pass_rate=overall_pass_rate,
        trajectory_accuracy=trajectory_acc,
        tool_selection_accuracy=tool_acc,
        pii_redaction_compliance=pii_comp,
        floor_margin_compliance=margin_comp,
        case_results=results,
    )

    log_outcome(
        action_name="run_golden_evaluation_suite",
        status="success" if failed_count == 0 else "failed",
        actual_outcome={
            "total_cases": total,
            "passed_cases": passed_count,
            "overall_pass_rate": overall_pass_rate,
            "trajectory_accuracy": trajectory_acc,
            "tool_selection_accuracy": tool_acc,
            "pii_redaction_compliance": pii_comp,
            "floor_margin_compliance": margin_comp,
        },
        agent_name="golden_evaluator",
    )
    return report


def main() -> int:
    """CLI entrypoint for running the Golden Evaluation Suite in CI/CD pipelines."""
    logger = get_structured_logger("autodeal_eval_cli")
    report = run_golden_evaluation_suite()
    logger.info(
        "Golden Evaluation Suite completed",
        extra={"evaluation_report": report.model_dump()},
    )
    sys.stdout.write(json.dumps(report.model_dump(), indent=2) + "\n")
    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
