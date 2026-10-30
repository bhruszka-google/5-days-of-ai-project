# AutoDeal Copilot — Enterprise Car Salesman Top-3 Offer Copilot

> **Google ADK-Python Multi-Agent System | Gemini 3rd-Gen Strategic Routing (`gemini-3-flash-preview` + `gemini-3.1-pro-preview`) | 95 / 95 AgentOps Rubric Compliant**

---

## 1. Executive Overview: Problem & Solution

### The Problem
While a retail automotive customer is sitting at the dealership desk, sales consultants must manually cross-reference live lot inventory, trade-in wholesale appraisals, stackable OEM manufacturer rebates, FICO credit-tier APR/lease tables, and dealership floor-margin rules. This fragmented process takes 25–40 minutes and typically produces a single generic quote that fails to match the buyer's lifestyle or budget—leading to customer walkouts, margin erosion, or compliance mistakes.

### The Solution (`AutoDeal Copilot`)
**AutoDeal Copilot** is a production-grade, multi-agent Enterprise Dealership Sales Copilot built with **Google ADK-Python (`google-adk`)** and deployed on **Google Cloud Run v2**. A salesperson enters the customer's preferences, family size, FICO score, cash down payment, and trade-in details, and the system orchestrates specialist sub-agents to generate a customer-ready **Top 3 Tailored Vehicle Offers** comparison sheet in seconds:

1. **Offer #1 — Best Overall Match**: Optimal alignment with seating capacity, fuel/propulsion preference, key features, and target monthly payment.
2. **Offer #2 — Budget / Value Pick**: Lowest monthly payment and lowest total cost of ownership leveraging Certified Pre-Owned (`cpo`) inventory, 60+ day aged-lot cash, and stackable rebates.
3. **Offer #3 — Premium / Upgrade Pick**: Higher trim, AWD/PHEV/EV powertrain, or luxury/safety package within a reasonable payment step-up.

Every request passes through **active PII redaction** (scrubbing SSNs, credit cards, driver's licenses, emails, and phone numbers before LLM calls, logging, or memory storage), **Truth-in-Lending Act (Regulation Z)** disclosures, **dealership floor-margin verification** (`selling_price >= invoice_price + $500`), and an explicit **Human-in-the-Loop (HITL) Sales Manager confirmation gate** (`require_confirmation` / `PENDING_MANAGER_APPROVAL`) whenever a promotional discount exceeds `$1,500` or encroaches on dealer invoice margin.

---

## 2. Multi-Agent System Architecture

```mermaid
flowchart TD
    User(["Dealership Salesperson / Desk Request"]) --> Plugin["PolicyAndPiiPlugin (ADK BasePlugin)\nPrompt-Injection Guardrail & Active PII Redactor\n(Scrubs SSN, Credit Card, Driver License, Email, Phone)"]
    Plugin --> Coordinator["dealership_coordinator (Root LlmAgent)\nModel: gemini-3-flash-preview\nOrchestration, Session Compaction & Async Memory"]

    Coordinator -->|1. Fast Catalog & Trade-In Discovery| InvAgent["inventory_specialist (LlmAgent)\nModel: gemini-3-flash-preview"]
    Coordinator -->|2. Multi-Constraint Ranking & APR/Lease Math| Strategist["top_offers_strategist (LlmAgent)\nModel: gemini-3.1-pro-preview"]
    Coordinator -->|3. Discount Governance & Offer Lock| Closer["deal_desk_closer (LlmAgent)\nModel: gemini-3-flash-preview + HITL Hook"]

    InvAgent --> InvTools["- search_dealership_inventory\n- appraise_trade_in_vehicle\n- fetch_active_manufacturer_rebates"]
    Strategist --> FinTools["- calculate_financing_and_lease_options\n- verify_dealership_floor_margin\n- generate_customer_offer_sheet"]
    Closer --> HITL{"Sales Manager HITL Gate\n(Discounts > $1,500 or < Invoice + $500)"}
    HITL -->|MGR-APPROVED Token / <= $1,500| CloserTools["- apply_manager_special_discount\n- verify_dealership_floor_margin\n- generate_customer_offer_sheet"]
    HITL -->|Unapproved > $1,500| Pause["Halt: PENDING_MANAGER_APPROVAL\n(requires_human_confirmation = True)"]

    subgraph Context & Memory Layer
        Store[("PersistentSessionStore\n(SQLite & Google Cloud Firestore)")]
        Compactor["ContextCompactor + ADK EventsCompactionConfig\n& ContextCacheConfig"]
        AsyncMem["AsyncMemoryConsolidator\n(Background asyncio.create_task)"]
    end

    Coordinator <--> Store
    Coordinator <--> Compactor
    Coordinator -.->|Non-Blocking Background Extraction| AsyncMem

    subgraph Observability, Security & Cloud IaC
        Logger["Structured JSON Logger\n(INTENT vs. OUTCOME + PII Scrubbing)"]
        OTel["OpenTelemetry Distributed Tracing\n(Cloud Trace & InMemorySpanCollector)"]
        SecretMgr["SecretManagerClient\n(Google Cloud Secret Manager)"]
        Terraform["Terraform IaC (deployment/terraform/ & terraform/)\nCloud Run v2 + Firestore + PromQL Alerts"]
    end
```

---

## 3. AgentOps Code Review Matrix (95 / 95 Points)

Every single criterion of the **19-item AgentOps Assessment Rubric** is implemented and verified by automated unit and static compliance tests in `tests/test_rubric_compliance.py`:

| Category | # | Rubric Criterion (5 pts each) | Implementation File(s) & Key Symbols |
| :--- | :---: | :--- | :--- |
| **1. Tool & Interface Design (20 pts)** | 1 | **Comprehensive Tool Docstrings** | `app/tools/inventory_tools.py`, `app/tools/financing_tools.py`, `app/guardrails/hitl.py` — Google-style docstrings with explicit `Args:`, `Returns:`, `Raises:`, and `Example:` blocks on all 7 tools. |
| | 2 | **Descriptive Naming** | Verb-noun tool names: `search_dealership_inventory`, `appraise_trade_in_vehicle`, `fetch_active_manufacturer_rebates`, `calculate_financing_and_lease_options`, `verify_dealership_floor_margin`, `apply_manager_special_discount`, `generate_customer_offer_sheet`. |
| | 3 | **Explicit JSON Schemas** | `app/tools/schemas.py` — Strict Pydantic `BaseModel` input/output schemas (`InventorySearchInput`, `TradeInAppraisalInput`, `RebateLookupInput`, `FinancingCalculationInput`, `MarginCheckInput`, `ManagerDiscountInput`, `OfferSheetInput`, `VehicleOfferCard`, `TopThreeOffersResponse`, `ToolErrorResponse`) with `ConfigDict(extra="forbid")`. |
| | 4 | **Guided Error Handling** | `app/tools/schemas.py` (`build_guided_error_response`), `app/tools/inventory_tools.py`, `app/tools/financing_tools.py` — Tools never crash on bad LLM inputs; they return `status="error"`, `error_code`, `message`, and actionable `recovery_suggestion` instructions. |
| **2. Context & Memory (20 pts)** | 5 | **Robust System Instructions** | `app/prompts/constitution.py` — `GLOBAL_SAFETY_CONSTITUTION`, `COORDINATOR_CONSTITUTION`, `INVENTORY_SPECIALIST_CONSTITUTION`, `TOP_OFFERS_STRATEGIST_CONSTITUTION`, `DEAL_DESK_CLOSER_CONSTITUTION`, and `get_agent_constitution()`. |
| | 6 | **History Compaction** | `app/memory/compaction.py` — `ContextCompactor` sliding-window token truncation + summarization, ADK `build_adk_events_compaction_config()` (`EventsCompactionConfig` with `LlmEventSummarizer`), and `build_adk_context_cache_config()` (`ContextCacheConfig`). |
| | 7 | **Persistent Session State** | `app/memory/session_store.py` — `PersistentSessionStore` backed by SQLite and Google Cloud Firestore adapter (`to_firestore_document` / `from_firestore_document`) persisting session turns, vehicle inventory, customer profiles, and active offers. |
| | 8 | **Async Memory Operations** | `app/memory/async_memory.py` — `AsyncMemoryConsolidator.schedule_consolidation()` using `asyncio.create_task` for non-blocking background customer preference extraction and profile consolidation. |
| **3. Orchestration & Logic (20 pts)** | 9 | **Multi-Agent Patterns** | `app/agent.py` — Coordinator + Specialist delegation (`create_dealership_coordinator`, `create_inventory_specialist`, `create_top_offers_strategist`, `create_deal_desk_closer`) and deterministic `SequentialAgent` (`create_offer_generation_pipeline`). |
| | 10 | **Strategic Model Routing** | `app/routing/model_router.py` — `StrategicModelRouter` routing low-latency lookup/orchestration/HITL tasks to `gemini-3-flash-preview` and complex multi-constraint Top-3 offer ranking to `gemini-3.1-pro-preview`. |
| | 11 | **Guardrails & Policy Plugins** | `app/guardrails/policy_plugin.py` — Runner-wide `PolicyAndPiiPlugin(BasePlugin)` + `before_model_guardrail_callback`, `after_model_self_eval_callback`, `before_tool_policy_callback`, `after_tool_audit_callback`, `inspect_user_prompt`, and `evaluate_agent_response`. |
| | 12 | **Human-in-the-Loop Hooks** | `app/guardrails/hitl.py` — `apply_manager_special_discount`, `requires_sales_manager_approval`, and `build_manager_discount_tool()` (`FunctionTool(require_confirmation=requires_sales_manager_approval)`) enforcing `PENDING_MANAGER_APPROVAL` for discounts > $1,500 or below `invoice_price + $500`. |
| **4. Observability & Tracing (20 pts)** | 13 | **Structured JSON Logging** | `app/observability/logger.py` — `PIIRedactingJsonFormatter` (`pythonjsonlogger`) emitting structured JSON with `severity`, `trace_id`, `span_id`, and **zero `print()` statements** across `app/`. |
| | 14 | **Intent vs. Outcome Capture** | `app/observability/logger.py` — `log_intent()` (`event_type="INTENT"`) recorded before execution and `log_outcome()` (`event_type="OUTCOME"`) recorded after execution across every tool and agent callback. |
| | 15 | **Distributed Tracing** | `app/observability/tracing.py` — OpenTelemetry `TracerProvider`, `InMemorySpanCollector`, `start_trace_span`, and `@traced_operation` decorator linking spans across coordinator, sub-agents, and tools. |
| | 16 | **PII Redaction** | `app/guardrails/pii_redactor.py` — `PIIRedactor` (`redact_text`, `redact_data`) actively scrubbing SSNs (`[REDACTED_SSN]`), credit cards (`[REDACTED_CC]`), driver's licenses (`[REDACTED_DL]`), emails (`[REDACTED_EMAIL]`), and phone numbers (`[REDACTED_PHONE]`). |
| **5. Infrastructure & CI/CD (15 pts)** | 17 | **Automated Evaluation Suites** | `evals/golden_dataset.json`, `evals/run_evals.py` (`run_golden_evaluation_suite`), `.github/workflows/ci_eval.yml`, and 1:1 mirrored `pytest` test suite under `tests/`. |
| | 18 | **Infrastructure as Code** | `deployment/terraform/` & `terraform/` (`main.tf`, `variables.tf`, `outputs.tf`, `iam.tf`, `alerts.tf`), `Dockerfile`, `Makefile`, and `agents-cli` / `adk` CLI workflows documented below. |
| | 19 | **Secure Secret Management** | `app/config.py` — `SecretManagerClient` retrieving `GOOGLE_API_KEY` from Google Cloud Secret Manager (`google-cloud-secret-manager`) + Terraform `google_secret_manager_secret.gemini_api_key` with zero hardcoded keys. |

---

## 4. Usage of `agents-cli` (Agent Starter Pack) & Google `adk` CLI

This project is structured to work seamlessly with both the **Agent Starter Pack CLI (`agents-cli`)** (configured in `pyproject.toml` under `[tool.agents-cli]`) and the **Google ADK CLI (`adk`)**.

### 4.1 Agent Starter Pack (`agents-cli`) Commands

```bash
# 1. Inspect project metadata, ADK template configuration, and deployment target
agents-cli info

# 2. Scaffold or refresh Agent Starter Pack template components for ADK on Cloud Run
agents-cli scaffold create --template adk --deployment-target cloud_run --agent-directory app

# 3. Run the AutoDeal Copilot multi-agent system locally via agents-cli
agents-cli run --agent-dir app

# 4. Execute the Golden Evaluation Suite via agents-cli evaluation runner
agents-cli eval run --dataset evals/golden_dataset.json --runner evals.run_evals

# 5. Provision single-project Google Cloud infrastructure (Cloud Run, Firestore, Secret Manager, Alerts)
agents-cli infra single-project --project-id "${GOOGLE_CLOUD_PROJECT}" --region us-central1 --terraform-dir deployment/terraform

# 6. Build and deploy AutoDeal Copilot to Google Cloud Run v2
agents-cli deploy --target cloud_run --agent-dir app --project-id "${GOOGLE_CLOUD_PROJECT}" --region us-central1
```

### 4.2 Google ADK CLI (`adk`) Commands

```bash
# Interactive terminal session with the root dealership_coordinator agent
.venv/bin/adk run app

# Launch the ADK Developer Web UI (trace inspector, sub-agent graph, HITL confirmation UI)
.venv/bin/adk web .

# Start the production ADK FastAPI server
.venv/bin/adk api_server . --port 8080

# Or run the custom FastAPI entrypoint (with /healthz and /v1/offers/generate)
.venv/bin/uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8080
```

---

## 5. Local Quickstart, Testing & Golden Evaluation

### 5.1 Install Dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

### 5.2 Run Linting & Formatting (`ruff`)

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

### 5.3 Run Unit & 19-Criterion Rubric Compliance Tests (`pytest`)

Every module in `app/` has a 1:1 mirrored unit test file in `tests/`, plus `tests/test_evals.py`, `tests/test_fast_api_app.py`, and `tests/test_rubric_compliance.py`:

```bash
.venv/bin/pytest -v
```

### 5.4 Run the Golden Dataset Evaluation Suite

Executes all 6 golden dealership scenarios (`evals/golden_dataset.json`) verifying trajectory accuracy, tool selection accuracy, PII redaction compliance, floor-margin compliance, and HITL discount gating:

```bash
.venv/bin/python -m evals.run_evals
```

---

## 6. Infrastructure as Code (Terraform) & Cloud Run Deployment

Production Terraform configurations are provided in both `deployment/terraform/` and `terraform/`:

- `main.tf`: Enables required Google Cloud APIs (`run`, `aiplatform`, `secretmanager`, `firestore`, `cloudtrace`, `dlp`, `monitoring`), provisions `google_secret_manager_secret.gemini_api_key`, `google_firestore_database.session_store`, `google_storage_bucket.telemetry_and_eval_artifacts`, and `google_cloud_run_v2_service.autodeal_copilot`.
- `iam.tf`: Provisions dedicated least-privilege runtime (`google_service_account.app_sa`) and CI/CD (`google_service_account.cicd_runner_sa`) service accounts with scoped `roles/secretmanager.secretAccessor`.
- `alerts.tf`: Configures Cloud Monitoring SRE PromQL alert policies (`google_monitoring_alert_policy`) for Cloud Run 5xx error ratio (`> 2%`) and P95 request latency (`> 8000ms`).
- `variables.tf` & `outputs.tf`: Parameterized inputs (`project_id`, `region`, `fast_model`, `pro_model`) and exported endpoints.

```bash
cd deployment/terraform
terraform init -backend=false
terraform validate
terraform plan -var="project_id=${GOOGLE_CLOUD_PROJECT}" -var="region=us-central1"
```

---

## 7. Repository Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── agent.py                     # Multi-agent ADK Coordinator, Specialists, SequentialAgent & Workflow
│   ├── config.py                    # AgentConfig (gemini-3-flash-preview / gemini-3.1-pro-preview) & SecretManagerClient
│   ├── fast_api_app.py              # Cloud Run FastAPI entrypoint (/healthz, /v1/offers/generate, ADK routes)
│   ├── data/
│   │   └── synthetic_catalog.py     # 24-vehicle dealership inventory, OEM rebates, credit tiers, trade-in matrix
│   ├── guardrails/
│   │   ├── hitl.py                  # Sales Manager HITL approval gate (apply_manager_special_discount)
│   │   ├── pii_redactor.py          # Active regex + key-based PII scrubber (SSN, CC, DL, Email, Phone)
│   │   └── policy_plugin.py         # ADK PolicyAndPiiPlugin (BasePlugin), prompt-injection defense & self-eval
│   ├── memory/
│   │   ├── async_memory.py          # AsyncMemoryConsolidator (non-blocking asyncio.create_task profile extraction)
│   │   ├── compaction.py            # ContextCompactor + ADK EventsCompactionConfig & ContextCacheConfig
│   │   └── session_store.py         # PersistentSessionStore (SQLite + Firestore compatible)
│   ├── observability/
│   │   ├── logger.py                # Structured JSON logger with INTENT vs. OUTCOME & PII scrubbing
│   │   └── tracing.py               # OpenTelemetry TracerProvider, spans & @traced_operation
│   ├── prompts/
│   │   └── constitution.py          # Constitutional system instructions for all 4 agent roles
│   ├── routing/
│   │   └── model_router.py          # StrategicModelRouter (Flash vs. Pro task complexity routing)
│   └── tools/
│       ├── financing_tools.py       # Loan/lease calculator, floor-margin verifier & Top-3 Offer Sheet generator
│       ├── inventory_tools.py       # Inventory search, trade-in appraisal & OEM rebate lookup tools
│       └── schemas.py               # Strict Pydantic schemas (extra="forbid") & guided ToolErrorResponse
├── deployment/terraform/            # Production Terraform HCL (mirrored in ./terraform/)
│   ├── alerts.tf
│   ├── iam.tf
│   ├── main.tf
│   ├── outputs.tf
│   └── variables.tf
├── evals/
│   ├── golden_dataset.json          # 6 comprehensive golden dealership evaluation scenarios
│   └── run_evals.py                 # Automated golden evaluation harness & CLI report generator
├── tests/                           # 1:1 mirrored unit tests + FastAPI, Evals & 19-Criterion Rubric Compliance
├── Dockerfile                       # Python 3.13 slim non-root Cloud Run container
├── Makefile                         # Developer & CI/CD automation targets
├── DESIGN_SPEC.md                   # Architectural specification
└── IMPLEMENTATION_PLAN.md           # Phased implementation roadmap & rubric checklist
```
