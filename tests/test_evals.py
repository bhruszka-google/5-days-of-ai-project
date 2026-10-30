"""Unit tests for Delivery 7: ADK EvalSet, EvalConfig, TrajectoryEvaluator & LlmOutputJudge.

Note: These unit tests validate schemas, trajectory scoring, and LLM-as-a-Judge parsing
without automatically running live LLM golden evaluations.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evals.run_evals import (
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_EVALSET_PATH,
    LlmOutputJudge,
    MissingLiveCredentialsError,
    build_arg_parser,
    compute_response_match_score,
    evaluate_adk_tool_trajectory,
    load_adk_eval_config,
    load_adk_eval_set,
    load_golden_dataset,
    main,
    run_golden_evaluation_suite,
)
from google.adk.evaluation.eval_case import (
    IntermediateData,
    Invocation,
    get_all_tool_calls,
)
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types


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
        assert len(tool_calls) >= 3
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
    """Verifies evals/golden_dataset.json defines all 6 dealership scenarios."""
    dataset = load_golden_dataset()
    cases = dataset.get("cases", [])
    assert len(cases) >= 6


def test_evaluate_adk_tool_trajectory_scores_in_order_and_partial_calls() -> None:
    """Verifies `evaluate_adk_tool_trajectory` uses ADK `TrajectoryEvaluator` accurately."""
    expected_inv = Invocation(
        invocation_id="inv-expected",
        user_content=types.Content(
            role="user", parts=[types.Part.from_text(text="Find me a hybrid SUV")]
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                types.FunctionCall(name="search_dealership_inventory", args={}),
                types.FunctionCall(
                    name="calculate_financing_and_lease_options", args={}
                ),
                types.FunctionCall(name="generate_customer_offer_sheet", args={}),
            ]
        ),
    )

    matching_inv = Invocation(
        invocation_id="inv-actual-match",
        user_content=expected_inv.user_content,
        intermediate_data=IntermediateData(
            tool_uses=[
                types.FunctionCall(name="search_dealership_inventory", args={}),
                types.FunctionCall(
                    name="calculate_financing_and_lease_options", args={}
                ),
                types.FunctionCall(name="generate_customer_offer_sheet", args={}),
            ]
        ),
    )
    assert evaluate_adk_tool_trajectory(matching_inv, expected_inv) == 1.0

    partial_inv = Invocation(
        invocation_id="inv-actual-partial",
        user_content=expected_inv.user_content,
        intermediate_data=IntermediateData(
            tool_uses=[
                types.FunctionCall(name="search_dealership_inventory", args={}),
            ]
        ),
    )
    partial_score = evaluate_adk_tool_trajectory(partial_inv, expected_inv)
    assert 0.0 < partial_score < 0.8


def test_llm_output_judge_with_mocked_genai_client_parses_structured_json() -> None:
    """Verifies `LlmOutputJudge.evaluate_response` parses structured JSON from `genai.Client`."""
    mock_judge_json = json.dumps(
        {
            "rubric_scores": [
                {
                    "rubric_id": "top_three_offers_or_hitl_gate",
                    "score": 1.0,
                    "rationale": "All 3 offer categories and unique VINs present.",
                },
                {
                    "rubric_id": "zero_unredacted_customer_pii",
                    "score": 1.0,
                    "rationale": "Zero raw PII tokens in response.",
                },
                {
                    "rubric_id": "dealership_pricing_apr_and_margin_compliance",
                    "score": 1.0,
                    "rationale": "Pricing, APR, and Regulation Z disclosures accurate.",
                },
            ]
        }
    )

    captured_calls: list[dict[str, str]] = []

    def _mock_generate_content(*, model: str, contents: str) -> SimpleNamespace:
        captured_calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=f"```json\n{mock_judge_json}\n```")

    mock_genai_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_mock_generate_content)
    )

    judge = LlmOutputJudge(genai_client=mock_genai_client)
    eval_set = load_adk_eval_set()
    ref_inv = (eval_set.eval_cases[0].conversation or [])[0]
    ref_text = "\n".join(p.text for p in (ref_inv.final_response.parts or []) if p.text)

    verdict = judge.evaluate_response(
        user_query="Need a 3-row family hybrid SUV under $650/mo",
        response_text=ref_text,
        executed_tools=[
            "search_dealership_inventory",
            "generate_customer_offer_sheet",
        ],
    )
    assert verdict.passed is True
    assert verdict.overall_score == 1.0
    assert len(verdict.rubric_scores) == 3
    assert len(captured_calls) == 1
    assert captured_calls[0]["model"] == "gemini-3-flash-preview"


def test_llm_output_judge_blocks_leaked_pii_even_if_model_passed() -> None:
    """Verifies `LlmOutputJudge` enforces hard PII safety check if SSN leaks into LLM output."""
    mock_judge_json = json.dumps(
        {
            "rubric_scores": [
                {
                    "rubric_id": "top_three_offers_or_hitl_gate",
                    "score": 1.0,
                    "rationale": "Has offers.",
                },
                {
                    "rubric_id": "zero_unredacted_customer_pii",
                    "score": 1.0,
                    "rationale": "Judge missed SSN.",
                },
                {
                    "rubric_id": "dealership_pricing_apr_and_margin_compliance",
                    "score": 1.0,
                    "rationale": "Has APR.",
                },
            ]
        }
    )
    judge = LlmOutputJudge(judge_fn=lambda _: mock_judge_json)
    leaked_pii_verdict = judge.evaluate_response(
        user_query="Customer SSN 123-45-6789 wants an SUV",
        response_text="Confirmed application for SSN 123-45-6789 at 5.4% APR.",
    )
    assert leaked_pii_verdict.passed is False
    pii_score = next(
        s
        for s in leaked_pii_verdict.rubric_scores
        if s.rubric_id == "zero_unredacted_customer_pii"
    )
    assert pii_score.score == 0.0


def test_run_golden_evaluation_suite_requires_live_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies `run_golden_evaluation_suite` refuses to run without live Gemini credentials."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    with pytest.raises(MissingLiveCredentialsError):
        run_golden_evaluation_suite()


def test_cli_default_and_validate_only_do_not_execute_live_eval() -> None:
    """Verifies running `main()` without `--confirm-live-eval` only validates schemas."""
    parser = build_arg_parser()
    parsed = parser.parse_args(["--validate-only"])
    assert parsed.validate_only is True
    assert parsed.confirm_live_eval is False
    assert main(["--validate-only"]) == 0
    assert compute_response_match_score("hybrid suv offer", "hybrid suv offer") == 1.0
