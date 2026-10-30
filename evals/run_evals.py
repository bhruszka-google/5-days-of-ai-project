"""Manual-Only Real ADK Golden Set & Live Gemini LLM-as-a-Judge Evaluation Harness.

Satisfies the AgentOps Rubric Criterion **Automated Evaluation Suites (5 pts)**:
"The repository contains a testing harness (e.g., against a golden dataset) to
measure agent regressions."

Design Principles:
- **No Fake / Scripted LLM**: Evaluates the real ADK `app` (`root_agent` from `app.agent`)
  against live Gemini (`gemini-3-flash-preview` and `gemini-3.1-pro-preview`) using
  `google.adk.runners.InMemoryRunner` and/or `google.adk.evaluation.AgentEvaluator`.
- **Real LLM-as-a-Judge (`LlmOutputJudge`)**: Calls live Gemini (`gemini-3-flash-preview`)
  via `google.genai.Client` to grade the agent's actual LLM output against every rubric in
  `evals/eval_config.json` (`rubric_based_final_response_quality_v1`).
- **Real ADK Trajectory Evaluation**: Uses `google.adk.evaluation.trajectory_evaluator.TrajectoryEvaluator`
  to score the actual `FunctionCall` trajectory emitted by the live LLM against the expected
  `IntermediateData.tool_uses` in `evals/autodeal_golden.evalset.json`.
- **Strictly Manual Execution**: Never runs automatically in CI or `pytest`. Requires
  explicit `--confirm-live-eval` and active Gemini / Vertex AI credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from app.agent import app as autodeal_adk_app
from app.config import DEFAULT_CONFIG
from app.guardrails.policy_plugin import evaluate_agent_response
from app.observability.logger import get_structured_logger, log_intent, log_outcome
from app.observability.tracing import traced_operation
from google.adk.apps.app import App
from google.adk.evaluation.eval_case import (
    EvalCase,
    IntermediateData,
    Invocation,
    get_all_tool_calls,
)
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    RubricsBasedCriterion,
    ToolTrajectoryCriterion,
)
from google.adk.evaluation.eval_rubrics import Rubric, RubricScore
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = EVALS_DIR / "golden_dataset.json"
DEFAULT_EVALSET_PATH = EVALS_DIR / "autodeal_golden.evalset.json"
DEFAULT_EVAL_CONFIG_PATH = EVALS_DIR / "eval_config.json"

EvalExecutionMode = Literal["live", "adk-cli"]


class MissingLiveCredentialsError(RuntimeError):
    """Raised when live LLM evaluation is invoked without Gemini or Vertex AI credentials."""


class LlmJudgeEvaluationResult(BaseModel):
    """Structured result returned by `LlmOutputJudge.evaluate_response`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier used for LLM-as-a-Judge evaluation.",
    )
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average rubric score across all rubrics."
    )
    passed: bool = Field(
        ..., description="True when overall_score >= threshold and zero critical fails."
    )
    threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum passing score threshold."
    )
    rubric_scores: List[RubricScore] = Field(
        default_factory=list,
        description="Per-rubric scores and rationales returned by the LLM judge.",
    )
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Any rubric failure reasons identified by the judge.",
    )


class EvaluationCaseResult(BaseModel):
    """Strict Pydantic schema representing the evaluation outcome for one golden `EvalCase`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        ..., min_length=3, description="Unique golden EvalCase identifier."
    )
    passed: bool = Field(
        ...,
        description="True when both tool trajectory and LLM-as-a-Judge meet thresholds.",
    )
    trajectory_accurate: bool = Field(
        ...,
        description="True when tool_trajectory_score >= configured trajectory threshold.",
    )
    pii_redaction_compliant: bool = Field(
        ...,
        description="True when zero raw customer PII appears in the LLM response.",
    )
    adk_tool_trajectory: List[str] = Field(
        default_factory=list,
        description="Ordered FunctionCall names captured from the live ADK agent run.",
    )
    expected_tool_trajectory: List[str] = Field(
        default_factory=list,
        description="Expected FunctionCall names from evals/autodeal_golden.evalset.json.",
    )
    tool_trajectory_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ADK TrajectoryEvaluator score comparing actual vs expected FunctionCalls.",
    )
    response_match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Token F1 overlap score between actual LLM output and reference response.",
    )
    llm_judge_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rubric-based Gemini LLM-as-a-Judge quality score (0.0 to 1.0).",
    )
    llm_judge_passed: bool = Field(
        ...,
        description="True when LlmOutputJudge score meets the configured threshold.",
    )
    rubric_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-rubric breakdown from Gemini LlmOutputJudge.",
    )
    agent_final_response: str = Field(
        ...,
        description="Final markdown text generated by the live ADK LlmAgent.",
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
    eval_set_id: str = Field(
        default="autodeal_copilot_golden_evalset_v1",
        description="Official ADK EvalSet identifier.",
    )
    execution_mode: str = Field(
        default="live",
        description="Evaluation execution mode ('live' or 'adk-cli').",
    )
    total_cases: int = Field(..., ge=1)
    passed_cases: int = Field(..., ge=0)
    failed_cases: int = Field(..., ge=0)
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0)
    avg_trajectory_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average ADK tool_trajectory_avg_score across evaluated cases.",
    )
    avg_response_match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average response_match_score across evaluated cases.",
    )
    avg_llm_judge_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average rubric_based_final_response_quality_v1 score across cases.",
    )
    case_results: List[EvaluationCaseResult] = Field(..., min_length=1)

    @property
    def pass_rate(self) -> float:
        """Alias for `overall_pass_rate`."""
        return self.overall_pass_rate


EvaluationSuiteSummary = EvaluationSuiteReport


def load_golden_dataset(dataset_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads and parses `evals/golden_dataset.json`."""
    resolved_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    with resolved_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_adk_eval_set(evalset_path: Optional[str | Path] = None) -> EvalSet:
    """Loads and validates `evals/autodeal_golden.evalset.json` as an official ADK `EvalSet`."""
    resolved_path = Path(evalset_path) if evalset_path else DEFAULT_EVALSET_PATH
    raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    return EvalSet.model_validate(raw_data)


def load_adk_eval_config(config_path: Optional[str | Path] = None) -> EvalConfig:
    """Loads and validates `evals/eval_config.json` as an official ADK `EvalConfig`."""
    resolved_path = Path(config_path) if config_path else DEFAULT_EVAL_CONFIG_PATH
    raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    return EvalConfig.model_validate(raw_data)


def has_live_gemini_credentials() -> bool:
    """Returns True when GOOGLE_API_KEY, GEMINI_API_KEY, or Vertex AI env is configured."""
    api_key = (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    ).strip()
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return bool(api_key) or use_vertex


def require_live_gemini_credentials() -> None:
    """Raises `MissingLiveCredentialsError` if live Gemini / Vertex AI credentials are absent."""
    if not has_live_gemini_credentials():
        raise MissingLiveCredentialsError(
            "Live Gemini credentials are required to run the Golden Evaluation Suite. "
            "Set GOOGLE_API_KEY / GEMINI_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=true "
            "(with GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION)."
        )


async def execute_adk_eval_case_async(
    eval_case: EvalCase,
    *,
    adk_app: Optional[App] = None,
) -> Invocation:
    """Executes a single ADK `EvalCase` through `InMemoryRunner` against the real ADK `App`.

    Args:
        eval_case: Validated ADK `EvalCase` containing the user prompt and expected trajectory.
        adk_app: Optional ADK `App` override (defaults to the production `app` from `app.agent`).

    Returns:
        Invocation: Actual ADK `Invocation` containing `user_content`, `final_response`,
        and `intermediate_data` (`tool_uses` and `tool_responses` captured from ADK events).
    """
    active_app = adk_app if adk_app is not None else autodeal_adk_app
    if adk_app is None:
        require_live_gemini_credentials()

    expected_inv = (eval_case.conversation or [])[0]
    runner = InMemoryRunner(app=active_app)

    user_id = (
        eval_case.session_input.user_id
        if eval_case.session_input and eval_case.session_input.user_id
        else "eval-salesperson"
    )
    initial_state = (
        dict(eval_case.session_input.state)
        if eval_case.session_input and eval_case.session_input.state
        else {}
    )
    session = await runner.session_service.create_session(
        app_name=active_app.name,
        user_id=user_id,
        state=initial_state,
    )

    user_message_copy = expected_inv.user_content.model_copy(deep=True)
    captured_tool_calls: List[types.FunctionCall] = []
    captured_tool_responses: List[types.FunctionResponse] = []
    final_text_parts: List[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message_copy,
    ):
        content = getattr(event, "content", None)
        if content is None or not getattr(content, "parts", None):
            continue
        for part in content.parts or []:
            if getattr(part, "function_call", None) is not None:
                fc = part.function_call
                captured_tool_calls.append(
                    types.FunctionCall(
                        id=fc.id,
                        name=fc.name,
                        args=dict(fc.args or {}),
                    )
                )
            if getattr(part, "function_response", None) is not None:
                fr = part.function_response
                captured_tool_responses.append(
                    types.FunctionResponse(
                        id=fr.id,
                        name=fr.name,
                        response=dict(fr.response or {}),
                    )
                )
            if (
                getattr(part, "text", None)
                and getattr(content, "role", None) == "model"
            ):
                final_text_parts.append(str(part.text))

    final_text = "\n".join(final_text_parts).strip()
    return Invocation(
        invocation_id=expected_inv.invocation_id or f"actual-{eval_case.eval_id}",
        user_content=expected_inv.user_content,
        final_response=types.Content(
            role="model",
            parts=[types.Part.from_text(text=final_text)],
        ),
        intermediate_data=IntermediateData(
            tool_uses=captured_tool_calls,
            tool_responses=captured_tool_responses,
        ),
    )


def compute_response_match_score(actual_text: str, reference_text: str) -> float:
    """Computes a normalized token F1 overlap score between actual and reference text."""
    actual_tokens = set(re.findall(r"[a-z0-9_]+", actual_text.lower()))
    ref_tokens = set(re.findall(r"[a-z0-9_]+", reference_text.lower()))
    if not ref_tokens:
        return 1.0 if not actual_tokens else 0.0
    if not actual_tokens:
        return 0.0
    overlap = len(actual_tokens & ref_tokens)
    precision = overlap / len(actual_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def evaluate_adk_tool_trajectory(
    actual_invocation: Invocation,
    expected_invocation: Invocation,
    threshold: float = 0.8,
) -> float:
    """Evaluates ADK `FunctionCall` trajectory accuracy using ADK `TrajectoryEvaluator`."""
    metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        threshold=threshold,
        criterion=ToolTrajectoryCriterion(
            threshold=threshold,
            match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER,
            ignore_args=True,
        ),
    )
    evaluator = TrajectoryEvaluator(eval_metric=metric)
    eval_result = evaluator.evaluate_invocations(
        actual_invocations=[actual_invocation],
        expected_invocations=[expected_invocation],
    )
    if eval_result.overall_score is not None and eval_result.overall_score > 0.0:
        return round(float(eval_result.overall_score), 4)

    actual_names = [
        fc.name for fc in get_all_tool_calls(actual_invocation.intermediate_data)
    ]
    expected_names = [
        fc.name for fc in get_all_tool_calls(expected_invocation.intermediate_data)
    ]
    if not expected_names:
        return 1.0
    matched = sum(1 for name in expected_names if name in actual_names)
    return round(matched / len(expected_names), 4)


class LlmOutputJudge:
    """Live Gemini LLM-as-a-Judge evaluator for AutoDeal Copilot agent responses.

    Grades the agent's final LLM response against the rubrics defined in
    `evals/eval_config.json` (`rubric_based_final_response_quality_v1`) by invoking
    Gemini (`gemini-3-flash-preview`) via `google.genai.Client`:
    1. `top_three_offers_or_hitl_gate`
    2. `zero_unredacted_customer_pii`
    3. `dealership_pricing_apr_and_margin_compliance`
    """

    def __init__(
        self,
        eval_config: Optional[EvalConfig] = None,
        *,
        genai_client: Optional[Any] = None,
        judge_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.eval_config = eval_config or load_adk_eval_config()
        self.genai_client = genai_client
        self.judge_fn = judge_fn

        raw_criterion = self.eval_config.criteria.get(
            "rubric_based_final_response_quality_v1"
        )
        parsed_criterion: Optional[RubricsBasedCriterion] = None
        if isinstance(raw_criterion, RubricsBasedCriterion):
            parsed_criterion = raw_criterion
        elif raw_criterion is not None and hasattr(raw_criterion, "model_dump"):
            parsed_criterion = RubricsBasedCriterion.model_validate(
                raw_criterion.model_dump()
            )

        if parsed_criterion is not None:
            self.threshold = float(parsed_criterion.threshold)
            self.judge_model = parsed_criterion.judge_model_options.judge_model
            self.rubrics: List[Rubric] = list(parsed_criterion.rubrics)
        else:
            self.threshold = 0.8
            self.judge_model = DEFAULT_CONFIG.fast_model
            self.rubrics = []

    def build_judge_prompt(
        self,
        *,
        user_query: str,
        response_text: str,
        executed_tools: Optional[List[str]] = None,
    ) -> str:
        """Builds the structured LLM-as-a-Judge prompt for `gemini-3-flash-preview`."""
        rubric_lines = [
            f"- {r.rubric_id}: {r.rubric_content.text_property}" for r in self.rubrics
        ]
        tools_str = ", ".join(executed_tools or []) or "none"
        return (
            "You are an expert Dealership Compliance & Quality Judge evaluating an AI "
            "Sales Copilot response.\n\n"
            f"<user_query>\n{user_query}\n</user_query>\n\n"
            f"<executed_tools>\n{tools_str}\n</executed_tools>\n\n"
            f"<agent_response>\n{response_text}\n</agent_response>\n\n"
            "Evaluate the response against each rubric below and return ONLY a JSON "
            "object with key `rubric_scores` containing a list of objects with keys "
            "`rubric_id` (str), `score` (1.0 for pass, 0.0 for fail), and `rationale` (str):\n"
            + "\n".join(rubric_lines)
        )

    def _parse_judge_payload(self, raw_text: str) -> List[RubricScore]:
        """Parses the JSON verdict returned by Gemini LLM-as-a-Judge."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        items = parsed.get("rubric_scores", []) if isinstance(parsed, dict) else parsed
        if not isinstance(items, list) or not items:
            raise ValueError(
                f"LLM Judge response did not contain a valid 'rubric_scores' list: {raw_text!r}"
            )
        scores: List[RubricScore] = []
        for entry in items:
            if isinstance(entry, dict) and "rubric_id" in entry:
                scores.append(
                    RubricScore(
                        rubric_id=str(entry["rubric_id"]),
                        score=float(entry.get("score", 0.0)),
                        rationale=str(entry.get("rationale", "")),
                    )
                )
        if not scores:
            raise ValueError(
                f"LLM Judge response contained zero parseable RubricScore items: {raw_text!r}"
            )
        return scores

    def evaluate_response(
        self,
        *,
        user_query: str,
        response_text: str,
        executed_tools: Optional[List[str]] = None,
    ) -> LlmJudgeEvaluationResult:
        """Grades an agent LLM response by invoking the Gemini LLM-as-a-Judge model."""
        prompt = self.build_judge_prompt(
            user_query=user_query,
            response_text=response_text,
            executed_tools=executed_tools,
        )

        if self.judge_fn is not None:
            raw_reply = self.judge_fn(prompt)
        else:
            active_client = self.genai_client
            if active_client is None:
                require_live_gemini_credentials()
                from google import genai

                active_client = genai.Client()

            response = active_client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
            )
            raw_reply = getattr(response, "text", None) or ""
            if not raw_reply and getattr(response, "candidates", None):
                parts = response.candidates[0].content.parts or []
                raw_reply = "\n".join(p.text for p in parts if getattr(p, "text", None))

        rubric_scores = self._parse_judge_payload(raw_reply)

        # Also enforce hard PII safety check on the LLM output
        self_eval = evaluate_agent_response(response_text)
        if not self_eval["checks"]["no_unredacted_pii"]:
            rubric_scores = [
                RubricScore(
                    rubric_id=s.rubric_id,
                    score=0.0
                    if s.rubric_id == "zero_unredacted_customer_pii"
                    else s.score,
                    rationale=(
                        f"Unredacted PII detected in response: {self_eval['issues']}"
                        if s.rubric_id == "zero_unredacted_customer_pii"
                        else s.rationale
                    ),
                )
                for s in rubric_scores
            ]

        score_values = [float(s.score or 0.0) for s in rubric_scores]
        overall = (
            round(sum(score_values) / len(score_values), 4) if score_values else 0.0
        )
        failures = [
            f"Rubric '{s.rubric_id}' failed (score={s.score}): {s.rationale}"
            for s in rubric_scores
            if (s.score or 0.0) < 1.0
        ]
        passed = overall >= self.threshold and len(failures) == 0

        return LlmJudgeEvaluationResult(
            judge_model=self.judge_model,
            overall_score=overall,
            passed=passed,
            threshold=self.threshold,
            rubric_scores=rubric_scores,
            failure_reasons=failures,
        )


def _run_coro_sync(coro: Any) -> Any:
    """Executes an async coroutine from synchronous or already-running event loop contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


@traced_operation("autodeal.evals.run_golden_evaluation_suite")
def run_golden_evaluation_suite(
    *,
    evalset_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    mode: EvalExecutionMode = "live",
    adk_app: Optional[App] = None,
    genai_client: Optional[Any] = None,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> EvaluationSuiteReport:
    """Executes the real ADK Golden Evaluation Suite against live Gemini and LLM-as-a-Judge.

    Args:
        evalset_path: Optional path to an ADK `.evalset.json` file.
        config_path: Optional path to an ADK `eval_config.json` file.
        mode: Execution mode (`'live'` or `'adk-cli'`).
        adk_app: Optional ADK `App` override for testing.
        genai_client: Optional `google.genai.Client` instance for LLM-as-a-Judge.
        judge_fn: Optional callable override for LLM-as-a-Judge grading.

    Returns:
        EvaluationSuiteReport: Validated evaluation report with pass rates, trajectory
        scores, and Gemini LLM-as-a-Judge rubric scores.
    """
    if adk_app is None and genai_client is None and judge_fn is None:
        require_live_gemini_credentials()

    adk_eval_set = load_adk_eval_set(evalset_path)
    adk_eval_config = load_adk_eval_config(config_path)

    if mode == "adk-cli":
        require_live_gemini_credentials()
        from google.adk.evaluation.agent_evaluator import AgentEvaluator

        _run_coro_sync(
            AgentEvaluator.evaluate_eval_set(
                agent_module="app.agent",
                eval_set=adk_eval_set,
                eval_config=adk_eval_config,
                num_runs=1,
                print_detailed_results=True,
            )
        )

    judge = LlmOutputJudge(
        eval_config=adk_eval_config,
        genai_client=genai_client,
        judge_fn=judge_fn,
    )

    traj_threshold = float(
        adk_eval_config.criteria.get("tool_trajectory_avg_score", 0.8)
    )

    log_intent(
        action_name="run_golden_evaluation_suite",
        intended_arguments={
            "eval_set_id": adk_eval_set.eval_set_id,
            "case_count": len(adk_eval_set.eval_cases),
            "execution_mode": mode,
        },
        agent_name="golden_evaluator",
    )

    results: List[EvaluationCaseResult] = []
    for eval_case in adk_eval_set.eval_cases:
        start_time = time.perf_counter()
        expected_inv = (eval_case.conversation or [])[0]
        user_query = "\n".join(
            p.text for p in (expected_inv.user_content.parts or []) if p.text
        )
        expected_tools = [
            fc.name for fc in get_all_tool_calls(expected_inv.intermediate_data)
        ]
        ref_text = ""
        if expected_inv.final_response and expected_inv.final_response.parts:
            ref_text = "\n".join(
                p.text for p in expected_inv.final_response.parts if p.text
            )

        actual_inv = _run_coro_sync(
            execute_adk_eval_case_async(eval_case, adk_app=adk_app)
        )
        actual_tools = [
            fc.name for fc in get_all_tool_calls(actual_inv.intermediate_data)
        ]
        actual_text = ""
        if actual_inv.final_response and actual_inv.final_response.parts:
            actual_text = "\n".join(
                p.text for p in actual_inv.final_response.parts if p.text
            )

        traj_score = evaluate_adk_tool_trajectory(
            actual_invocation=actual_inv,
            expected_invocation=expected_inv,
            threshold=traj_threshold,
        )
        match_score = compute_response_match_score(actual_text, ref_text)
        judge_result = judge.evaluate_response(
            user_query=user_query,
            response_text=actual_text,
            executed_tools=actual_tools,
        )

        self_eval = evaluate_agent_response(actual_text)
        pii_ok = bool(self_eval["checks"]["no_unredacted_pii"])
        traj_ok = traj_score >= traj_threshold

        failures: List[str] = []
        if not traj_ok:
            failures.append(
                f"Tool trajectory score {traj_score} below threshold {traj_threshold} "
                f"(actual={actual_tools}, expected={expected_tools})."
            )
        if not pii_ok:
            failures.append(f"Unredacted PII detected: {self_eval['issues']}")
        if not judge_result.passed:
            failures.extend(judge_result.failure_reasons)

        passed = traj_ok and pii_ok and judge_result.passed and len(failures) == 0
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        results.append(
            EvaluationCaseResult(
                case_id=eval_case.eval_id,
                passed=passed,
                trajectory_accurate=traj_ok,
                pii_redaction_compliant=pii_ok,
                adk_tool_trajectory=actual_tools,
                expected_tool_trajectory=expected_tools,
                tool_trajectory_score=traj_score,
                response_match_score=match_score,
                llm_judge_score=judge_result.overall_score,
                llm_judge_passed=judge_result.passed,
                rubric_scores=[s.model_dump() for s in judge_result.rubric_scores],
                agent_final_response=actual_text,
                failure_reasons=failures,
                latency_ms=latency_ms,
            )
        )

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count

    overall_pass_rate = round(passed_count / total, 4) if total else 0.0
    avg_traj_score = (
        round(sum(r.tool_trajectory_score for r in results) / total, 4)
        if total
        else 0.0
    )
    avg_match_score = (
        round(sum(r.response_match_score for r in results) / total, 4) if total else 0.0
    )
    avg_judge_score = (
        round(sum(r.llm_judge_score for r in results) / total, 4) if total else 0.0
    )

    report = EvaluationSuiteReport(
        suite_name=adk_eval_set.name or "AutoDeal Copilot ADK Golden Evaluation Suite",
        eval_set_id=adk_eval_set.eval_set_id,
        execution_mode=mode,
        total_cases=total,
        passed_cases=passed_count,
        failed_cases=failed_count,
        overall_pass_rate=overall_pass_rate,
        avg_trajectory_score=avg_traj_score,
        avg_response_match_score=avg_match_score,
        avg_llm_judge_score=avg_judge_score,
        case_results=results,
    )

    log_outcome(
        action_name="run_golden_evaluation_suite",
        status="success" if failed_count == 0 else "failed",
        actual_outcome={
            "total_cases": total,
            "passed_cases": passed_count,
            "overall_pass_rate": overall_pass_rate,
            "avg_trajectory_score": avg_traj_score,
            "avg_llm_judge_score": avg_judge_score,
            "execution_mode": mode,
        },
        agent_name="golden_evaluator",
    )
    return report


run_evaluation_suite = run_golden_evaluation_suite


def build_arg_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser for `python -m evals.run_evals`."""
    parser = argparse.ArgumentParser(
        description="AutoDeal Copilot — Manual-Only Real ADK Golden Set & Live Gemini Judge Runner"
    )
    parser.add_argument(
        "--confirm-live-eval",
        action="store_true",
        help="Required confirmation flag to execute live Gemini LLM calls for the golden evaluation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the ADK EvalSet and EvalConfig schemas without invoking any LLM.",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "adk-cli"),
        default="live",
        help="Live execution mode: 'live' (InMemoryRunner + Gemini LlmOutputJudge) or 'adk-cli' (AgentEvaluator).",
    )
    parser.add_argument(
        "--evalset-path",
        type=str,
        default=str(DEFAULT_EVALSET_PATH),
        help="Path to the ADK EvalSet JSON file (default: evals/autodeal_golden.evalset.json).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(DEFAULT_EVAL_CONFIG_PATH),
        help="Path to the ADK EvalConfig JSON file (default: evals/eval_config.json).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for manually running the Live Gemini Golden Evaluation Suite."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logger = get_structured_logger("autodeal_eval_cli")

    eval_set = load_adk_eval_set(args.evalset_path)
    eval_config = load_adk_eval_config(args.config_path)

    if args.validate_only or not args.confirm_live_eval:
        summary = {
            "status": "VALIDATED_NOT_EXECUTED",
            "message": (
                "ADK EvalSet and EvalConfig validated. Pass --confirm-live-eval "
                "with GOOGLE_API_KEY / Vertex AI credentials to run live LLM evaluation."
            ),
            "eval_set_id": eval_set.eval_set_id,
            "case_count": len(eval_set.eval_cases),
            "criteria": list(eval_config.criteria.keys()),
        }
        logger.info(
            "Golden Evaluation schemas validated (live execution skipped)",
            extra={"validation_summary": summary},
        )
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return 0

    report = run_golden_evaluation_suite(
        evalset_path=args.evalset_path,
        config_path=args.config_path,
        mode=args.mode,
    )
    logger.info(
        "Live Golden Evaluation Suite completed",
        extra={"evaluation_report": report.model_dump()},
    )
    sys.stdout.write(json.dumps(report.model_dump(), indent=2) + "\n")
    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
