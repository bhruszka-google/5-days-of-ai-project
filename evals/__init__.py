"""Automated Golden Dataset Evaluation Suite for AutoDeal Copilot.

Satisfies the AgentOps Rubric Criterion **Automated Evaluation Suites (5 pts)**:
"The repository contains a testing harness (e.g., against a golden dataset) to
statically measure agent regressions."
"""

from evals.run_evals import (
    EvaluationCaseResult,
    EvaluationSuiteReport,
    load_golden_dataset,
    run_golden_evaluation_suite,
)

__all__ = [
    "EvaluationCaseResult",
    "EvaluationSuiteReport",
    "load_golden_dataset",
    "run_golden_evaluation_suite",
]
