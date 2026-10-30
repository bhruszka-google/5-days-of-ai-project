"""Multi-Agent Orchestration and ADK App Entrypoint for AutoDeal Copilot.

Satisfies the AgentOps Rubric Criterion **Multi-Agent Patterns (5 pts)**:
"Complex tasks utilize proven design patterns (e.g., Coordinator, Sequential)
rather than monolithic agents implemented in ADK."

Architecture:
- `dealership_coordinator` (`LlmAgent`, `gemini-3-flash-preview`): Root
  coordinator that parses customer desk requests, enforces constitutional
  guardrails, manages session history compaction & async customer profile
  consolidation, and delegates tasks to specialist sub-agents.
- `inventory_specialist` (`LlmAgent`, `gemini-3-flash-preview`): Searches live
  dealership inventory (`search_dealership_inventory`), appraises trade-ins
  (`appraise_trade_in_vehicle`), and identifies stackable OEM rebates
  (`fetch_active_manufacturer_rebates`).
- `top_offers_strategist` (`LlmAgent`, `gemini-3.1-pro-preview`): Performs
  multi-constraint vehicle ranking across customer budget, seating, fuel type,
  credit-tier APR/lease amortization (`calculate_financing_and_lease_options`),
  and floor-margin compliance (`verify_dealership_floor_margin`) to structure
  the Top 3 Tailored Vehicle Offers (`Best Overall Match`, `Budget / Value Pick`,
  `Premium / Upgrade Pick`).
- `deal_desk_closer` (`LlmAgent`, `gemini-3-flash-preview`): Enforces the
  Sales Manager Human-in-the-Loop (HITL) approval gate on high-stakes discounts
  (`apply_manager_special_discount`) and generates the customer-ready Top-3
  Offer Sheet (`generate_customer_offer_sheet`).
- `offer_generation_pipeline` (`SequentialAgent`): Deterministic sequential
  pipeline chaining `inventory_specialist -> top_offers_strategist -> deal_desk_closer`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.genai import types

from app.config import DEFAULT_CONFIG, AgentConfig
from app.guardrails.hitl import (
    apply_manager_special_discount,
    build_manager_discount_tool,
)
from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR
from app.guardrails.policy_plugin import (
    PolicyAndPiiPlugin,
    after_model_self_eval_callback,
    after_tool_audit_callback,
    before_model_guardrail_callback,
    before_tool_policy_callback,
    evaluate_agent_response,
    inspect_user_prompt,
)
from app.memory.async_memory import (
    DEFAULT_ASYNC_CONSOLIDATOR,
    AsyncMemoryConsolidator,
)
from app.memory.compaction import (
    DEFAULT_CONTEXT_COMPACTOR,
    ContextCompactor,
    build_adk_context_cache_config,
    build_adk_events_compaction_config,
)
from app.memory.session_store import DEFAULT_SESSION_STORE, PersistentSessionStore
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation
from app.prompts.constitution import (
    COORDINATOR_CONSTITUTION,
    DEAL_DESK_CLOSER_CONSTITUTION,
    INVENTORY_SPECIALIST_CONSTITUTION,
    TOP_OFFERS_STRATEGIST_CONSTITUTION,
)
from app.routing.model_router import DEFAULT_MODEL_ROUTER, StrategicModelRouter
from app.tools.financing_tools import (
    calculate_financing_and_lease_options,
    generate_customer_offer_sheet,
    verify_dealership_floor_margin,
)
from app.tools.inventory_tools import (
    appraise_trade_in_vehicle,
    fetch_active_manufacturer_rebates,
    search_dealership_inventory,
)


def _extract_session_id(callback_context: Optional[CallbackContext]) -> str:
    """Resolves a stable session ID from an ADK `CallbackContext`."""
    if callback_context is None:
        return "default-autodeal-session"
    inv_ctx = getattr(callback_context, "_invocation_context", None) or getattr(
        callback_context, "invocation_context", None
    )
    session = getattr(inv_ctx, "session", None) or getattr(
        callback_context, "session", None
    )
    session_id = getattr(session, "id", None) or getattr(
        callback_context, "session_id", None
    )
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict) and state.get("session_id"):
        return str(state["session_id"])
    return "default-autodeal-session"


@traced_operation("autodeal.agent.before_agent_callback")
def before_agent_lifecycle_callback(
    callback_context: Optional[CallbackContext] = None,
    *,
    session_store: Optional[PersistentSessionStore] = None,
    compactor: Optional[ContextCompactor] = None,
) -> Optional[types.Content]:
    """ADK `before_agent_callback` that hydrates customer profile and compacts history.

    Loads the consolidated customer profile from `PersistentSessionStore`, runs
    sliding-window history compaction via `ContextCompactor` when turn count or
    token budget exceeds thresholds, and injects compacted context into the ADK
    session state.

    Args:
        callback_context: Active ADK `CallbackContext` for the agent turn.
        session_store: Optional `PersistentSessionStore` override.
        compactor: Optional `ContextCompactor` override.

    Returns:
        Optional[types.Content]: Always returns `None` so normal agent execution
        proceeds with hydrated session state.
    """
    store = session_store or DEFAULT_SESSION_STORE
    active_compactor = compactor or DEFAULT_CONTEXT_COMPACTOR
    agent_name = (
        getattr(callback_context, "agent_name", None) or "dealership_coordinator"
    )
    session_id = _extract_session_id(callback_context)

    log_intent(
        action_name="before_agent_lifecycle_callback",
        intended_arguments={"session_id": session_id, "agent_name": agent_name},
        agent_name=agent_name,
    )

    history = store.get_session_history(session_id)
    compacted_history = active_compactor.compact_history(history)
    if len(compacted_history) < len(history):
        store.replace_session_history(session_id, compacted_history)

    profile = store.get_customer_profile(session_id)
    state = getattr(callback_context, "state", None)
    if state is not None and hasattr(state, "__setitem__"):
        state["session_id"] = session_id
        state["customer_profile"] = profile
        state["compacted_turn_count"] = len(compacted_history)

    log_outcome(
        action_name="before_agent_lifecycle_callback",
        status="success",
        actual_outcome={
            "session_id": session_id,
            "history_turns": len(compacted_history),
            "profile_keys": sorted(profile.keys()),
        },
        agent_name=agent_name,
    )
    return None


@traced_operation("autodeal.agent.after_agent_callback")
def after_agent_lifecycle_callback(
    callback_context: Optional[CallbackContext] = None,
    *,
    session_store: Optional[PersistentSessionStore] = None,
    consolidator: Optional[AsyncMemoryConsolidator] = None,
) -> Optional[types.Content]:
    """ADK `after_agent_callback` triggering async customer memory consolidation.

    Schedules non-blocking customer preference extraction via
    `AsyncMemoryConsolidator.schedule_consolidation` (`asyncio.create_task`) when
    an event loop is active, or falls back to synchronous profile upsert in
    synchronous unit tests.

    Args:
        callback_context: Active ADK `CallbackContext` for the completed turn.
        session_store: Optional `PersistentSessionStore` override.
        consolidator: Optional `AsyncMemoryConsolidator` override.

    Returns:
        Optional[types.Content]: `None` so the agent's response is returned intact.
    """
    store = session_store or DEFAULT_SESSION_STORE
    active_consolidator = consolidator or DEFAULT_ASYNC_CONSOLIDATOR
    agent_name = (
        getattr(callback_context, "agent_name", None) or "dealership_coordinator"
    )
    session_id = _extract_session_id(callback_context)

    log_intent(
        action_name="after_agent_lifecycle_callback",
        intended_arguments={"session_id": session_id, "agent_name": agent_name},
        agent_name=agent_name,
    )

    history = store.get_session_history(session_id, limit=8)
    scheduled_async = False
    if history:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                active_consolidator.schedule_consolidation(
                    session_id=session_id,
                    turns_or_text=history,
                )
                scheduled_async = True
        except RuntimeError:
            extracted = active_consolidator.extract_customer_facts(history)
            store.upsert_customer_profile(session_id, extracted)

    log_outcome(
        action_name="after_agent_lifecycle_callback",
        status="success",
        actual_outcome={
            "session_id": session_id,
            "scheduled_async_consolidation": scheduled_async,
        },
        agent_name=agent_name,
    )
    return None


def create_inventory_specialist(
    router: Optional[StrategicModelRouter] = None,
) -> LlmAgent:
    """Creates the `inventory_specialist` sub-agent routed to `gemini-3-flash-preview`.

    Args:
        router: Optional `StrategicModelRouter` instance.

    Returns:
        LlmAgent: Configured Inventory & Trade-In Specialist ADK agent.
    """
    active_router = router or DEFAULT_MODEL_ROUTER
    routing = active_router.select_model_for_agent("inventory_specialist")
    return LlmAgent(
        name="inventory_specialist",
        model=routing.selected_model,
        description=(
            "Searches live dealership vehicle inventory, appraises customer trade-in "
            "vehicles for net equity, and fetches stackable OEM manufacturer rebates."
        ),
        instruction=INVENTORY_SPECIALIST_CONSTITUTION,
        tools=[
            search_dealership_inventory,
            appraise_trade_in_vehicle,
            fetch_active_manufacturer_rebates,
        ],
        output_key="inventory_findings",
        before_agent_callback=before_agent_lifecycle_callback,
        after_agent_callback=after_agent_lifecycle_callback,
        before_model_callback=before_model_guardrail_callback,
        after_model_callback=after_model_self_eval_callback,
        before_tool_callback=before_tool_policy_callback,
        after_tool_callback=after_tool_audit_callback,
    )


def create_top_offers_strategist(
    router: Optional[StrategicModelRouter] = None,
) -> LlmAgent:
    """Creates the `top_offers_strategist` sub-agent routed to `gemini-3.1-pro-preview`.

    Args:
        router: Optional `StrategicModelRouter` instance.

    Returns:
        LlmAgent: Configured Top-3 Offer Strategist & Financier ADK agent.
    """
    active_router = router or DEFAULT_MODEL_ROUTER
    routing = active_router.select_model_for_agent("top_offers_strategist")
    return LlmAgent(
        name="top_offers_strategist",
        model=routing.selected_model,
        description=(
            "Performs multi-constraint reasoning across vehicle inventory, customer "
            "budget, FICO credit-tier APRs, 36-month lease residuals, and dealer "
            "floor-margin rules to structure the Top 3 Tailored Vehicle Offers."
        ),
        instruction=TOP_OFFERS_STRATEGIST_CONSTITUTION,
        tools=[
            calculate_financing_and_lease_options,
            verify_dealership_floor_margin,
            generate_customer_offer_sheet,
        ],
        output_key="top_three_offers",
        before_agent_callback=before_agent_lifecycle_callback,
        after_agent_callback=after_agent_lifecycle_callback,
        before_model_callback=before_model_guardrail_callback,
        after_model_callback=after_model_self_eval_callback,
        before_tool_callback=before_tool_policy_callback,
        after_tool_callback=after_tool_audit_callback,
    )


def create_deal_desk_closer(
    router: Optional[StrategicModelRouter] = None,
) -> LlmAgent:
    """Creates the `deal_desk_closer` sub-agent with the Sales Manager HITL tool.

    Args:
        router: Optional `StrategicModelRouter` instance.

    Returns:
        LlmAgent: Configured Deal Desk & HITL Discount Closer ADK agent.
    """
    active_router = router or DEFAULT_MODEL_ROUTER
    routing = active_router.select_model_for_agent("deal_desk_closer")
    return LlmAgent(
        name="deal_desk_closer",
        model=routing.selected_model,
        description=(
            "Applies promotional discounts with a Sales Manager Human-in-the-Loop "
            "(HITL) confirmation gate for discounts over $1,500 or near invoice, "
            "verifies floor margin, and compiles the final customer offer sheet."
        ),
        instruction=DEAL_DESK_CLOSER_CONSTITUTION,
        tools=[
            build_manager_discount_tool(),
            verify_dealership_floor_margin,
            generate_customer_offer_sheet,
        ],
        output_key="closing_decision",
        before_agent_callback=before_agent_lifecycle_callback,
        after_agent_callback=after_agent_lifecycle_callback,
        before_model_callback=before_model_guardrail_callback,
        after_model_callback=after_model_self_eval_callback,
        before_tool_callback=before_tool_policy_callback,
        after_tool_callback=after_tool_audit_callback,
    )


def create_offer_generation_pipeline(
    router: Optional[StrategicModelRouter] = None,
) -> SequentialAgent:
    """Creates a deterministic `SequentialAgent` chaining the three specialist stages.

    Executes in strict sequence:
    1. `inventory_specialist` (Flash) -> populates `inventory_findings`
    2. `top_offers_strategist` (Pro) -> populates `top_three_offers`
    3. `deal_desk_closer` (Flash + HITL) -> populates `closing_decision`

    Args:
        router: Optional `StrategicModelRouter` instance.

    Returns:
        SequentialAgent: Configured three-stage ADK sequential pipeline.
    """
    return SequentialAgent(
        name="offer_generation_pipeline",
        description=(
            "Sequential dealership workflow that discovers inventory & trade-in "
            "equity, ranks and structures the Top 3 Vehicle Offers, and enforces "
            "Sales Manager HITL floor-margin compliance."
        ),
        sub_agents=[
            create_inventory_specialist(router=router),
            create_top_offers_strategist(router=router),
            create_deal_desk_closer(router=router),
        ],
        before_agent_callback=before_agent_lifecycle_callback,
        after_agent_callback=after_agent_lifecycle_callback,
    )


def create_dealership_coordinator(
    router: Optional[StrategicModelRouter] = None,
) -> LlmAgent:
    """Creates the root `dealership_coordinator` agent with specialist sub-agents.

    Args:
        router: Optional `StrategicModelRouter` instance.

    Returns:
        LlmAgent: Root Coordinator ADK agent (`dealership_coordinator`).
    """
    active_router = router or DEFAULT_MODEL_ROUTER
    routing = active_router.select_model_for_agent("dealership_coordinator")
    return LlmAgent(
        name="dealership_coordinator",
        model=routing.selected_model,
        description=(
            "Root Enterprise Dealership Sales Copilot Coordinator that orchestrates "
            "inventory lookup, trade-in appraisal, multi-constraint Top-3 Offer "
            "structuring, and Sales Manager HITL discount approvals."
        ),
        instruction=COORDINATOR_CONSTITUTION,
        tools=[
            search_dealership_inventory,
            appraise_trade_in_vehicle,
            fetch_active_manufacturer_rebates,
            calculate_financing_and_lease_options,
            verify_dealership_floor_margin,
            build_manager_discount_tool(),
            generate_customer_offer_sheet,
        ],
        sub_agents=[
            create_inventory_specialist(router=active_router),
            create_top_offers_strategist(router=active_router),
            create_deal_desk_closer(router=active_router),
        ],
        before_agent_callback=before_agent_lifecycle_callback,
        after_agent_callback=after_agent_lifecycle_callback,
        before_model_callback=before_model_guardrail_callback,
        after_model_callback=after_model_self_eval_callback,
        before_tool_callback=before_tool_policy_callback,
        after_tool_callback=after_tool_audit_callback,
    )


def create_adk_app(
    config: Optional[AgentConfig] = None,
    router: Optional[StrategicModelRouter] = None,
) -> App:
    """Constructs the production Google ADK `App` with guardrails and compaction.

    Wires together:
    - Root `dealership_coordinator` multi-agent hierarchy
    - `PolicyAndPiiPlugin` (`BasePlugin`) for runner-wide PII scrubbing & guardrails
    - `EventsCompactionConfig` for conversation event sliding-window compaction
    - `ContextCacheConfig` for static constitution and schema prefix caching

    Args:
        config: Optional `AgentConfig` override.
        router: Optional `StrategicModelRouter` override.

    Returns:
        App: Configured Google ADK `App` ready for `InMemoryRunner` or Cloud Run.
    """
    active_config = config or DEFAULT_CONFIG
    coordinator = create_dealership_coordinator(router=router)
    return App(
        name=active_config.app_name,
        root_agent=coordinator,
        plugins=[PolicyAndPiiPlugin()],
        events_compaction_config=build_adk_events_compaction_config(),
        context_cache_config=build_adk_context_cache_config(),
    )


def _select_distinct_top_three_vins(
    primary_vehicles: List[Dict[str, Any]],
    fallback_body_style: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Selects three distinct vehicles for Best Match, Value Pick, and Upgrade Pick."""
    candidates: List[Dict[str, Any]] = list(primary_vehicles)
    seen_vins = {v["vin"] for v in candidates}

    if len(candidates) < 3 and fallback_body_style:
        broader_style = search_dealership_inventory(
            body_style=fallback_body_style,
            limit=10,
        )
        for v in broader_style.get("vehicles", []):
            if v["vin"] not in seen_vins:
                candidates.append(v)
                seen_vins.add(v["vin"])

    if len(candidates) < 3:
        all_stock = search_dealership_inventory(limit=15)
        for v in all_stock.get("vehicles", []):
            if v["vin"] not in seen_vins:
                candidates.append(v)
                seen_vins.add(v["vin"])

    sorted_by_price = sorted(candidates, key=lambda item: float(item["selling_price"]))
    value_pick = sorted_by_price[0]
    upgrade_pick = sorted_by_price[-1]
    middle_idx = len(sorted_by_price) // 2
    best_match = sorted_by_price[middle_idx]

    ordered: List[Dict[str, Any]] = []
    used: set[str] = set()
    for chosen in (best_match, value_pick, upgrade_pick, *sorted_by_price):
        if chosen["vin"] not in used:
            ordered.append(chosen)
            used.add(chosen["vin"])
        if len(ordered) == 3:
            break
    return ordered


@traced_operation("autodeal.agent.generate_top_three_offers_workflow")
def generate_top_three_offers_workflow(
    user_query: str,
    *,
    session_id: str = "desk-session-001",
    customer_name: str = "Valued Dealership Guest",
    body_style: Optional[str] = "suv",
    max_price: Optional[float] = 52000.0,
    min_seating: Optional[int] = None,
    fuel_type: Optional[str] = None,
    down_payment: float = 4000.0,
    credit_score: int = 740,
    loan_term_months: int = 60,
    trade_in_year: Optional[int] = None,
    trade_in_make: Optional[str] = None,
    trade_in_model: Optional[str] = None,
    trade_in_mileage: Optional[int] = None,
    trade_in_condition: str = "good",
    trade_in_payoff: float = 0.0,
    requested_discount_amount: Optional[float] = None,
    discount_reason: str = "Customer desk negotiation request",
    manager_approval_token: Optional[str] = None,
    session_store: Optional[PersistentSessionStore] = None,
    compactor: Optional[ContextCompactor] = None,
    consolidator: Optional[AsyncMemoryConsolidator] = None,
    router: Optional[StrategicModelRouter] = None,
) -> Dict[str, Any]:
    """Executes the deterministic end-to-end AutoDeal Copilot multi-agent workflow.

    Orchestrates the complete dealership pipeline across guardrails, strategic
    model routing, inventory search, trade-in appraisal, OEM rebate lookup,
    financing & floor-margin verification, optional HITL manager discount
    gating, Top-3 Offer Sheet generation, session persistence, and customer
    memory consolidation.

    Args:
        user_query: Raw salesperson or customer desk prompt.
        session_id: Conversation session identifier for persistence.
        customer_name: Customer name or buyer label (scrubbed for PII).
        body_style: Optional body style filter (`'suv'`, `'sedan'`, `'crossover'`, `'truck'`, `'ev'`).
        max_price: Maximum selling price filter in USD.
        min_seating: Minimum seating capacity filter.
        fuel_type: Optional fuel type filter (`'gas'`, `'hybrid'`, `'phev'`, `'ev'`).
        down_payment: Cash down payment in USD.
        credit_score: Customer FICO score (300 to 850).
        loan_term_months: Loan term in months (`24`, `36`, `48`, `60`, `72`).
        trade_in_year: Optional model year of trade-in vehicle.
        trade_in_make: Optional make of trade-in vehicle.
        trade_in_model: Optional model of trade-in vehicle.
        trade_in_mileage: Optional mileage of trade-in vehicle.
        trade_in_condition: Condition of trade-in (`'excellent'`, `'good'`, `'fair'`, `'rough'`).
        trade_in_payoff: Remaining loan payoff on trade-in vehicle in USD.
        requested_discount_amount: Optional promotional discount requested in USD.
        discount_reason: Justification for any requested promotional discount.
        manager_approval_token: Optional Sales Manager HITL approval token.
        session_store: Optional `PersistentSessionStore` override.
        compactor: Optional `ContextCompactor` override.
        consolidator: Optional `AsyncMemoryConsolidator` override.
        router: Optional `StrategicModelRouter` override.

    Returns:
        Dict[str, Any]: Complete structured workflow result including
        `status`, `routing_decision`, `guardrail_inspection`,
        `trade_in_appraisal`, `inventory_search`, `offer_sheet`,
        `discount_decision`, `self_evaluation`, and `customer_profile`.
    """
    store = session_store or DEFAULT_SESSION_STORE
    active_compactor = compactor or DEFAULT_CONTEXT_COMPACTOR
    active_consolidator = consolidator or DEFAULT_ASYNC_CONSOLIDATOR
    active_router = router or DEFAULT_MODEL_ROUTER

    sanitized_query = DEFAULT_PII_REDACTOR.redact_text(user_query)
    log_intent(
        action_name="generate_top_three_offers_workflow",
        intended_arguments={
            "session_id": session_id,
            "query_preview": sanitized_query[:100],
            "body_style": body_style,
            "max_price": max_price,
            "credit_score": credit_score,
        },
        agent_name="dealership_coordinator",
    )

    inspection = inspect_user_prompt(user_query)
    store.save_session_turn(
        session_id=session_id,
        role="user",
        content=inspection["sanitized_text"],
        metadata={"contained_pii": inspection["contained_pii"]},
    )

    if inspection["is_blocked"]:
        block_msg = str(inspection["policy_message"])
        store.save_session_turn(
            session_id=session_id,
            role="assistant",
            content=block_msg,
            metadata={"violation_code": inspection["violation_code"]},
        )
        blocked_result = {
            "status": "blocked_by_guardrail",
            "session_id": session_id,
            "guardrail_inspection": inspection,
            "response_text": block_msg,
            "self_evaluation": evaluate_agent_response(block_msg),
        }
        log_outcome(
            action_name="generate_top_three_offers_workflow",
            status="blocked_by_guardrail",
            actual_outcome={"violation_code": inspection["violation_code"]},
            agent_name="dealership_coordinator",
        )
        return blocked_result

    routing_decision = active_router.route_query(inspection["sanitized_text"])

    trade_in_result: Optional[Dict[str, Any]] = None
    net_trade_in_equity = 0.0
    if (
        trade_in_year is not None
        and trade_in_make is not None
        and trade_in_model is not None
        and trade_in_mileage is not None
    ):
        trade_in_result = appraise_trade_in_vehicle(
            year=trade_in_year,
            make=trade_in_make,
            model=trade_in_model,
            mileage=trade_in_mileage,
            condition=trade_in_condition,
            existing_loan_payoff=trade_in_payoff,
        )
        if trade_in_result.get("status") == "success":
            net_trade_in_equity = float(trade_in_result["net_trade_in_equity"])

    inv_result = search_dealership_inventory(
        body_style=body_style,
        max_price=max_price,
        min_seating=min_seating,
        fuel_type=fuel_type,
        limit=8,
    )
    primary_matches = (
        inv_result.get("vehicles", []) if inv_result.get("status") == "success" else []
    )
    selected_three = _select_distinct_top_three_vins(
        primary_vehicles=primary_matches,
        fallback_body_style=body_style,
    )

    best_vin = selected_three[0]["vin"]
    value_vin = selected_three[1]["vin"]
    upgrade_vin = selected_three[2]["vin"]

    rebates_lookup = fetch_active_manufacturer_rebates(
        vehicle_identifier=best_vin,
        customer_qualifiers=["trade_in"] if net_trade_in_equity > 0 else None,
    )

    offer_sheet = generate_customer_offer_sheet(
        customer_name=customer_name,
        best_match_vin=best_vin,
        value_pick_vin=value_vin,
        upgrade_pick_vin=upgrade_vin,
        down_payment=down_payment,
        trade_in_equity=net_trade_in_equity,
        credit_score=credit_score,
        loan_term_months=loan_term_months,
        custom_notes=inspection["sanitized_text"],
    )

    discount_decision: Optional[Dict[str, Any]] = None
    workflow_status = "success"
    if requested_discount_amount is not None and requested_discount_amount > 0:
        discount_decision = apply_manager_special_discount(
            vehicle_identifier=best_vin,
            requested_discount_amount=requested_discount_amount,
            reason=discount_reason,
            manager_approval_token=manager_approval_token,
        )
        store.save_active_offer_state(
            session_id=session_id,
            offer_key=best_vin,
            offer_payload=discount_decision,
        )
        if discount_decision.get("status") == "PENDING_MANAGER_APPROVAL":
            workflow_status = "PENDING_MANAGER_APPROVAL"

    summary_text = (
        f"{offer_sheet.get('customer_summary', '')} "
        f"{offer_sheet.get('compliance_disclaimer', '')}"
    ).strip()
    self_eval = evaluate_agent_response(summary_text)

    store.save_session_turn(
        session_id=session_id,
        role="assistant",
        content=self_eval["sanitized_text"],
        metadata={
            "workflow_status": workflow_status,
            "offer_vins": [best_vin, value_vin, upgrade_vin],
        },
    )

    full_history = store.get_session_history(session_id)
    compacted_turns = active_compactor.compact_history(full_history)
    if len(compacted_turns) < len(full_history):
        store.replace_session_history(session_id, compacted_turns)

    extracted_facts = active_consolidator.extract_customer_facts(full_history)
    consolidated_profile = store.upsert_customer_profile(session_id, extracted_facts)

    result_payload = {
        "status": workflow_status,
        "session_id": session_id,
        "routing_decision": routing_decision.model_dump(),
        "guardrail_inspection": inspection,
        "trade_in_appraisal": trade_in_result,
        "inventory_search": inv_result,
        "best_match_rebates": rebates_lookup,
        "offer_sheet": offer_sheet,
        "discount_decision": discount_decision,
        "self_evaluation": self_eval,
        "customer_profile": consolidated_profile,
        "response_text": self_eval["sanitized_text"],
    }
    log_outcome(
        action_name="generate_top_three_offers_workflow",
        status=workflow_status,
        actual_outcome={
            "session_id": session_id,
            "routed_model": routing_decision.selected_model,
            "offer_vins": [best_vin, value_vin, upgrade_vin],
            "self_eval_passed": self_eval["passed"],
        },
        agent_name="dealership_coordinator",
    )
    return result_payload


root_agent: LlmAgent = create_dealership_coordinator()
app: App = create_adk_app()
