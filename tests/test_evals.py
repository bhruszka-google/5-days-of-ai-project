"""Unit tests for the Golden Dataset Evaluation Suite (`evals/run_evals.py`)."""

from __future__ import annotations

from evals.run_evals import (
    EvaluationCaseResult,
    EvaluationSuiteReport,
    load_golden_dataset,
    run_golden_evaluation_suite,
)


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


def test_run_golden_evaluation_suite_achieves_100_percent_pass_rate() -> None:
    """Executes the full golden evaluation suite and asserts 1.0 accuracy across all metrics."""
    report = run_golden_evaluation_suite()
    assert isinstance(report, EvaluationSuiteReport)
    assert report.total_cases >= 6
    assert report.failed_cases == 0
    assert report.passed_cases == report.total_cases
    assert report.overall_pass_rate == 1.0
    assert report.trajectory_accuracy == 1.0
    assert report.tool_selection_accuracy == 1.0
    assert report.pii_redaction_compliance == 1.0
    assert report.floor_margin_compliance == 1.0

    for case_res in report.case_results:
        assert isinstance(case_res, EvaluationCaseResult)
        assert case_res.passed is True, (
            f"Case {case_res.case_id} failed: {case_res.failure_reasons}"
        )
        assert case_res.failure_reasons == []
