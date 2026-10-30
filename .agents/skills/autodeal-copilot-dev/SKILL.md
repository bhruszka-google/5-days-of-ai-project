---
name: autodeal-copilot-dev
description: >-
  Development, testing, evaluation, and deployment workflow for the AutoDeal
  Copilot ADK multi-agent system. Use this skill whenever writing or modifying
  tools, agents, guardrails, memory stores, observability components, pytest
  unit tests, or Terraform infrastructure in this repository.
---

# AutoDeal Copilot Development & AgentOps Rubric Skill

Follow these architectural invariants and verification steps for every delivery in `AutoDeal Copilot`.

## 1. Model Generation & Strategic Routing Invariants
Always use the newest 3rd-generation Gemini models configured in `app/config.py`:
- **Fast / High-Throughput Tasks (`FAST_MODEL`)**: `gemini-3-flash-preview`
  - Assigned to: `dealership_coordinator`, `inventory_specialist`, `deal_desk_closer`, and `LlmEventSummarizer` (context compaction).
- **Complex Multi-Constraint Reasoning (`PRO_MODEL`)**: `gemini-3.1-pro-preview`
  - Assigned to: `top_offers_strategist` (ranks vehicles, calculates APR/lease math, checks floor margin, and structures the Top 3 Offer Cards: *Best Overall Match*, *Budget/Value Pick*, and *Premium/Upgrade Pick*).

## 2. AgentOps 95-Point Code Review Checklist (Must Hold in Every File)
1. **Tool & Interface Design (20 pts)**:
   - Every tool function MUST have a complete Google-style docstring (`Args`, `Returns`, `Raises`, example).
   - Tool names MUST be descriptive verb-noun pairs (`search_dealership_inventory`, `appraise_trade_in_vehicle`, `fetch_active_manufacturer_rebates`, `calculate_financing_and_lease_options`, `verify_dealership_floor_margin`, `apply_manager_special_discount`, `generate_customer_offer_sheet`).
   - All tool inputs and outputs MUST be validated with strict `pydantic.BaseModel` schemas (`extra="forbid"`).
   - Tools MUST NEVER crash on invalid LLM arguments; return structured dicts with `status="error"`, `error_code`, and `recovery_suggestion`.
2. **Context & Memory (20 pts)**:
   - System prompts in `app/prompts/constitution.py` define explicit persona, domain rules, and constraints.
   - History compaction uses both ADK's `EventsCompactionConfig` / `ContextCacheConfig` and `ContextCompactor` sliding-window summarization.
   - Session & vehicle catalog state persists via `PersistentSessionStore` (SQLite + Firestore compatible).
   - Expensive customer preference extraction runs asynchronously via `AsyncMemoryConsolidator` (`asyncio.create_task`).
3. **Orchestration & Logic (20 pts)**:
   - Multi-agent Coordinator + Specialist delegation (`app/agent.py`).
   - `StrategicModelRouter` (`app/routing/model_router.py`) routes requests between `gemini-3-flash-preview` and `gemini-3.1-pro-preview`.
   - `PolicyAndPiiPlugin` (`BasePlugin`) and agent callbacks enforce prompt-injection guardrails, floor-margin rules, and self-evaluation.
   - High-stakes discounts (`apply_manager_special_discount` > $1,500 or near invoice) trigger an explicit Human-in-the-Loop (HITL) Sales Manager confirmation stop (`require_confirmation`).
4. **Observability & Tracing (20 pts)**:
   - Zero `print()` statements in `app/`. All logs use structured JSON via `app/observability/logger.py`.
   - Every tool and agent action logs `event="INTENT"` before execution and `event="OUTCOME"` after execution.
   - OpenTelemetry spans (`app/observability/tracing.py`) trace requests end-to-end.
   - `PIIRedactor` (`app/guardrails/pii_redactor.py`) scrubs SSNs, driver's licenses, credit cards, phone numbers, and emails before logging or memory storage.
5. **Infrastructure & CI/CD (15 pts)**:
   - Golden evaluation harness in `evals/run_evals.py` + `pytest` suites in `tests/unit/`.
   - Terraform IaC in `deployment/terraform/` provisioning Cloud Run, Secret Manager, Firestore, Cloud Trace, and SRE PromQL alerts.
   - Zero hardcoded API keys; `SecretManagerClient` in `app/config.py` retrieves credentials from Google Cloud Secret Manager.

## 3. Running Linting, Formatting, Unit Tests & Evaluations
Every source file under `app/<subpackage>/<module>.py` MUST have a corresponding 1:1 mirrored test file under `tests/<subpackage>/test_<module>.py`.
After every delivery (and automatically via `.pre-commit-config.yaml` on `git commit`), run `ruff` and `pytest`:
```bash
.venv/bin/ruff check --fix .
.venv/bin/ruff format --check .
.venv/bin/pytest -v
```
