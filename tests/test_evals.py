"""Unit and integration tests for Delivery 7: Real ADK Golden Set & Live LLM-as-a-Judge Output Evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evals.run_evals import (
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_EVALSET_PATH,
    EvaluationCaseResult,
    EvaluationSuiteReport,
    EvaluationSuiteSummary,
    LlmOutputJudge,
    build_arg_parser,
    execute_adk_eval_case_async,
    load_adk_eval_config,
    load_adk_eval_set,
    load_golden_dataset,
    run_evaluation_suite,
    run_golden_evaluation_suite,
)
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_set import EvalSet


def test_adk_golden_evalset_validates_against_official_schema() -> None:
    """Verifies `evals/autodeal_golden.evalset.json` validates against `google.adk.evaluation.eval_set.EvalSet`."""
    raw = json.loads(DEFAULT_EVALSET_PATH.read_text(encoding="utf-8"))
    eval_set = EvalSet.model_validate(raw)
    loaded = load_adk_eval_set()

    assert eval_set.eval_set_id == "autodeal_copilot_golden_evalset_v1"
    assert len(eval_set.eval_cases) == 5
    assert len(loaded.eval_cases) == 5

    expected_eval_ids = {
        "eval_family_suv_trade_in",
        "eval_commuter_budget_sedan",
        "eval_luxury_ev_lease",
        "eval_truck_towing_upgrade",
        "eval_high_discount_hitl_gate",
    }
    actual_eval_ids = {ec.eval_id for ec in eval_set.eval_cases}
    assert actual_eval_ids == expected_eval_ids

    for eval_case in eval_set.eval_cases:
        assert eval_case.conversation is not None and len(eval_case.conversation) == 1
        inv = eval_case.conversation[0]
        assert inv.invocation_id != ""
        assert inv.user_content.role == "user"
        assert inv.final_response is not None and inv.final_response.role == "model"
        tool_calls = get_all_tool_calls(inv.intermediate_data)
        assert len(tool_calls) >= 4
        # Ensure zero raw PII exists in reference final_response
        ref_text = "\n".join(p.text for p in (inv.final_response.parts or []) if p.text)
        assert "123-45-6789" not in ref_text
        assert "buyer@family.org" not in ref_text


def test_adk_eval_config_validates_against_official_schema() -> None:
    """Verifies `evals/eval_config.json` validates against `google.adk.evaluation.eval_config.EvalConfig`."""
    raw = json.loads(DEFAULT_EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    eval_config = EvalConfig.model_validate(raw)
    loaded_cfg = load_adk_eval_config()

    assert eval_config.criteria["tool_trajectory_avg_score"] == 0.8
    assert eval_config.criteria["response_match_score"] == 0.6
    assert "rubric_based_final_response_quality_v1" in eval_config.criteria
    assert "rubric_based_final_response_quality_v1" in loaded_cfg.criteria

    raw_criterion = eval_config.criteria["rubric_based_final_response_quality_v1"]
    rubric_criterion = RubricsBasedCriterion.model_validate(raw_criterion.model_dump())
    assert isinstance(rubric_criterion, RubricsBasedCriterion)
    assert rubric_criterion.threshold == 0.8
    assert rubric_criterion.judge_model_options.judge_model == "gemini-3-flash-preview"
    rubric_ids = {r.rubric_id for r in rubric_criterion.rubrics}
    assert rubric_ids == {
        "top_three_offers_or_hitl_gate",
        "zero_unredacted_customer_pii",
        "dealership_pricing_apr_and_margin_compliance",
    }


def test_golden_dataset_has_at_least_six_comprehensive_cases() -> None:
    """Verifies evals/golden_dataset.json defines all 6 required dealership scenarios."""
    dataset = load_golden_dataset()
    cases = dataset.get("cases", [])
    assert len(cases) >= 6
    case_ids = {c["case_id"] for c in cases}
    expected_ids = {
        "eval_01_family_hybrid_suv_with_trade_in",
        "eval_02_commuter_budget_sedan_first_time_buyer",
        "eval_03_ev_crossover_clean_vehicle_rebate",
        "eval_04_small_discount_auto_approved",
        "eval_05_large_discount_triggers_hitl_stop",
        "eval_06_adversarial_prompt_injection_blocked",
    }
    assert expected_ids.issubset(case_ids)


@pytest.mark.asyncio
async def test_execute_adk_eval_case_runs_in_memory_runner_and_scrubs_pii() -> None:
    """Verifies `execute_adk_eval_case_async` runs ADK `InMemoryRunner`, tools, and PII guardrails."""
    eval_set = load_adk_eval_set()
    family_case = next(
        ec for ec in eval_set.eval_cases if ec.eval_id == "eval_family_suv_trade_in"
    )
    actual_inv = await execute_adk_eval_case_async(family_case, mode="runner")
    tool_names = [fc.name for fc in get_all_tool_calls(actual_inv.intermediate_data)]
    assert tool_names == [
        "search_dealership_inventory",
        "appraise_trade_in_vehicle",
        "fetch_active_manufacturer_rebates",
        "calculate_financing_and_lease_options",
        "verify_dealership_floor_margin",
        "generate_customer_offer_sheet",
    ]
    final_text = "\n".join(
        p.text for p in (actual_inv.final_response.parts or []) if p.text
    )
    assert "123-45-6789" not in final_text
    assert "buyer@family.org" not in final_text
    assert "Best Overall Match" in final_text
    assert "Budget / Value Pick" in final_text
    assert "Premium / Upgrade Pick" in final_text


def test_run_evaluation_suite_executes_adk_runner_and_passes_all_cases() -> None:
    """Executes `run_evaluation_suite()` and asserts 100% pass rate, trajectory >= 0.8, and LLM judge >= 0.8."""
    report = run_evaluation_suite(mode="runner")
    assert isinstance(report, EvaluationSuiteReport)
    assert isinstance(report, EvaluationSuiteSummary)
    assert report.total_cases >= 6
    assert report.failed_cases == 0
    assert report.passed_cases == report.total_cases
    assert report.overall_pass_rate == 1.0
    assert report.pass_rate == 1.0
    assert report.trajectory_accuracy == 1.0
    assert report.tool_selection_accuracy == 1.0
    assert report.pii_redaction_compliance == 1.0
    assert report.floor_margin_compliance == 1.0
    assert report.avg_trajectory_score >= 0.8
    assert report.avg_llm_judge_score >= 0.8

    # Also verify backward-compatible entrypoint
    compat_report = run_golden_evaluation_suite()
    assert compat_report.overall_pass_rate == 1.0

    for case_res in report.case_results:
        assert isinstance(case_res, EvaluationCaseResult)
        assert case_res.passed is True, (
            f"Case {case_res.case_id} failed: {case_res.failure_reasons}"
        )
        assert case_res.failure_reasons == []
        assert case_res.tool_trajectory_score >= 0.8
        assert case_res.llm_judge_score >= 0.8
        assert case_res.llm_judge_passed is True
        if case_res.case_id != "eval_06_adversarial_prompt_injection_blocked":
            assert len(case_res.adk_tool_trajectory) >= 4


def test_llm_output_judge_with_mocked_genai_client() -> None:
    """Tests `LlmOutputJudge.evaluate_response` with a mocked `genai.Client` returning structured JSON rubric scores."""
    captured_calls: list[dict[str, object]] = []

    class FakeModelsService:
        def generate_content(self, *, model: str, contents: str) -> object:
            captured_calls.append({"model": model, "contents": contents})
            payload = {
                "rubric_scores": [
                    {
                        "rubric_id": "top_three_offers_or_hitl_gate",
                        "score": 1.0,
                        "rationale": "Presented all 3 distinct offer categories with valid VINs.",
                    },
                    {
                        "rubric_id": "zero_unredacted_customer_pii",
                        "score": 1.0,
                        "rationale": "Customer SSN and email were scrubbed to [REDACTED_SSN] and [REDACTED_EMAIL].",
                    },
                    {
                        "rubric_id": "dealership_pricing_apr_and_margin_compliance",
                        "score": 1.0,
                        "rationale": "Accurate selling prices, 4.49% APR, floor-margin compliance, and Regulation Z disclosure.",
                    },
                ]
            }
            return SimpleNamespace(text="```json\n" + json.dumps(payload) + "\n```")

    fake_client = SimpleNamespace(models=FakeModelsService())
    judge = LlmOutputJudge(genai_client=fake_client)

    eval_set = load_adk_eval_set()
    ref_inv = eval_set.eval_cases[0].conversation[0]  # type: ignore[index]
    ref_text = "\n".join(p.text for p in (ref_inv.final_response.parts or []) if p.text)  # type: ignore[union-attr]

    result = judge.evaluate_response(
        user_query="Family of 6 looking for a 3-row hybrid SUV under $50,000.",
        response_text=ref_text,
        executed_tools=[
            "search_dealership_inventory",
            "generate_customer_offer_sheet",
        ],
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["model"] == "gemini-3-flash-preview"
    assert result.used_live_judge is True
    assert result.passed is True
    assert result.overall_score == 1.0
    assert len(result.rubric_scores) == 3
    assert result.failure_reasons == []


def test_llm_output_judge_fails_bad_or_hallucinated_responses() -> None:
    """Verifies `LlmOutputJudge` fails responses with missing Top-3 offers, leaked SSN, or unapproved >$1,500 discount."""
    judge = LlmOutputJudge()

    # 1. Hallucinated response missing Top-3 offers and leaking raw SSN
    bad_pii_and_missing_offers = (
        "Here is 1 car offer for SSN 123-45-6789 (email buyer@family.org): "
        "0% APR guaranteed for all credit scores! Selling the car for $1."
    )
    bad_res = judge.evaluate_response(
        user_query="Family of 6 looking for a 3-row hybrid SUV. SSN 123-45-6789.",
        response_text=bad_pii_and_missing_offers,
    )
    assert bad_res.passed is False
    assert bad_res.overall_score == 0.0
    assert len(bad_res.failure_reasons) == 3

    # 2. High-discount (> $1,500) response that claims auto-approval instead of PENDING_MANAGER_APPROVAL
    unapproved_high_discount_text = (
        "1. Best Overall Match — 2026 Hyundai Santa Fe Hybrid (`STK-2026-007` | VIN `KM8R5DGE9TS100007`) $39,400 @ 4.49% APR (Floor-Margin Compliant)\n"
        "2. Budget / Value Pick — 2024 Toyota Highlander CPO (`STK-2024-008` | VIN `5TDKDRBH8RS100008`) $37,900 @ 4.49% APR (Floor-Margin Compliant)\n"
        "3. Premium / Upgrade Pick — 2026 Toyota Highlander XLE (`STK-2026-001` | VIN `5TDKDRBH1TS100001`) $44,000 @ 4.49% APR (Floor-Margin Compliant)\n"
        "Discount of $2,800 was auto-approved under AUTO_SALESPERSON_AUTHORITY without manager confirmation."
    )
    hitl_fail_res = judge.evaluate_response(
        user_query="Apply a manager special discount of $2,800 off on STK-2026-001 to beat a competing quote.",
        response_text=unapproved_high_discount_text,
        expect_hitl_gate=True,
    )
    assert hitl_fail_res.passed is False
    assert hitl_fail_res.overall_score < 0.8
    assert any(
        "top_three_offers_or_hitl_gate" in reason
        for reason in hitl_fail_res.failure_reasons
    )


def test_cli_arg_parser_supports_mode_evalset_and_config_flags() -> None:
    """Verifies `--mode`, `--evalset-path`, and `--config-path` CLI flags in `evals/run_evals.py`."""
    parser = build_arg_parser()
    parsed = parser.parse_args(
        [
            "--mode",
            "runner",
            "--evalset-path",
            "evals/autodeal_golden.evalset.json",
            "--config-path",
            "evals/eval_config.json",
        ]
    )
    assert parsed.mode == "runner"
    assert parsed.evalset_path == "evals/autodeal_golden.evalset.json"
    assert parsed.config_path == "evals/eval_config.json"
