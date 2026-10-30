# DESIGN_SPEC.md — AutoDeal Copilot (Car Salesman Top-3 Offer Copilot)

## Overview
`AutoDeal Copilot` is an enterprise multi-agent assistant built with the **Google Agent Development Kit (ADK-Python)** and **Gemini 3rd-generation models**. It assists dealership salespeople while a customer is at the desk by searching lot inventory, appraising trade-ins, matching manufacturer rebates, calculating APR/lease payments, enforcing dealer floor-margin policies, and generating **Top 3 Tailored Vehicle Offers** (*Best Overall Match*, *Budget/Value Pick*, and *Premium/Upgrade Pick*).

## Model Selection (Newest of Class)
- **Fast / High-Throughput Tasks (`FAST_MODEL`)**: `gemini-3-flash-preview` (configurable to `gemini-3.8-flash` / `gemini-3-flash-preview` via `AUTODEAL_FAST_MODEL`) — powers `dealership_coordinator`, `inventory_specialist`, `deal_desk_closer`, and `LlmEventSummarizer`.
- **Deep Multi-Constraint Reasoning (`PRO_MODEL`)**: `gemini-3.1-pro-preview` (configurable via `AUTODEAL_PRO_MODEL`) — powers `top_offers_strategist` for ranking vehicles across customer budget, family size, fuel preference, rebates, and dealer gross margin.

## Example Use Cases
1. **Family SUV Buyer with Trade-In**:
   - *Input*: "Customer needs a 3-row hybrid SUV under $650/mo, 740 credit score, $4,000 down, trading in a 2021 Honda CR-V with 45k miles in good condition. Customer email: john.doe@example.com, SSN: 123-45-6789."
   - *Expected Behavior*: Scrubs SSN and email before logging/memory storage, appraises the 2021 CR-V, searches 3-row SUVs, computes monthly payments at Tier 1 APR with active rebates, and returns 3 structured offer cards.
2. **High-Stakes Manager Discount Request (HITL)**:
   - *Input*: "Apply a $2,500 manager special discount on the 2026 Toyota Highlander Hybrid XLE to close the deal today."
   - *Expected Behavior*: Triggers the `apply_manager_special_discount` Human-in-the-Loop (HITL) Sales Manager approval gate (`PENDING_MANAGER_APPROVAL`) before modifying the offer below standard margin thresholds.

## Tools Required
| Tool Name | Purpose | Schema Validation & Error Recovery |
| :--- | :--- | :--- |
| `search_dealership_inventory` | Searches synthetic lot inventory by body style, max price, fuel type, seating, and condition. | Validated via `InventorySearchInput`; suggests broader price/style filters if 0 matches. |
| `appraise_trade_in_vehicle` | Calculates trade-in allowance based on year, make, model, mileage, and condition. | Validated via `TradeInAppraisalInput`; returns guided error if year/mileage out of bounds. |
| `fetch_active_manufacturer_rebates` | Finds stackable manufacturer incentives (EV/Hybrid, Loyalty, Aged Inventory, First-Time Buyer). | Validated via `RebateLookupInput`. |
| `calculate_financing_and_lease_options` | Computes exact monthly loan payment and 36-month lease payment by credit tier, down payment, and trade-in equity. | Validated via `FinancingCalculationInput`; guides LLM to valid loan terms (`24, 36, 48, 60, 72`). |
| `verify_dealership_floor_margin` | Audits proposed selling price against vehicle `invoice_price` and minimum required gross margin. | Validated via `MarginCheckInput`. |
| `apply_manager_special_discount` | Applies promotional discount; enforces HITL Sales Manager confirmation for discounts > $1,500 or near invoice. | Validated via `ManagerDiscountInput` + HITL gate. |
| `generate_customer_offer_sheet` | Compiles the final Top 3 Offers into a structured, compliant customer-ready presentation sheet. | Validated via `OfferSheetInput`. |

## Constraints & Safety Rules
1. **Zero Hardcoded Secrets**: All API keys and credentials are fetched via `SecretManagerClient` from Google Cloud Secret Manager (`google-cloud-secret-manager`).
2. **Active PII Redaction**: Customer SSNs, driver's license numbers, credit card numbers, emails, and phone numbers are actively scrubbed from all logs and memory stores.
3. **Floor Margin Guardrail**: No offer may be finalized below dealer invoice minus holdback without explicit Sales Manager HITL approval.
4. **Structured Observability**: Zero `print()` statements in `app/`. All tool/agent actions emit structured JSON logs with `INTENT` and `OUTCOME` events linked to OpenTelemetry trace spans.

## Success Criteria
- 100% pass rate on `pytest` unit and rubric compliance test suites (`tests/unit/`).
- 100% pass rate on the Golden Dataset evaluation harness (`evals/run_evals.py`).
- Full 95/95 score across all 19 AgentOps Code Review Matrix criteria.
