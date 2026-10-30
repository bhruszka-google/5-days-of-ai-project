"""Asynchronous Background Customer Memory Extraction and Consolidation.

Executes expensive customer preference extraction (budget limits, FICO credit
score, seating/family size, preferred body styles & fuel types, and trade-in
mentions) in non-blocking background tasks via `asyncio.create_task` so the
salesperson's main conversational thread never stalls.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor
from app.memory.session_store import DEFAULT_SESSION_STORE, PersistentSessionStore
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation


class AsyncMemoryConsolidator:
    """Non-blocking background customer profile extractor and consolidator."""

    _MONTHLY_PAYMENT_RE = re.compile(
        r"\$\s?(\d{3,4}(?:,\d{3})?)\s*(?:/\s*mo(?:nth)?|per\s*month|monthly)",
        re.IGNORECASE,
    )
    _DOWN_PAYMENT_RE = re.compile(
        r"\$\s?(\d{1,3}(?:,\d{3})+|\d{3,5})\s*down",
        re.IGNORECASE,
    )
    _MAX_PRICE_RE = re.compile(
        r"(?:under|below|max|budget(?:\s*of)?)\s*\$\s?(\d{2,3}(?:,\d{3})+|\d{5})",
        re.IGNORECASE,
    )
    _CREDIT_SCORE_RE = re.compile(
        r"\b([3-8]\d{2})\s*(?:credit\s*score|fico|credit)\b|\b(?:credit\s*score|fico)\s*(?:of|is|:)?\s*([3-8]\d{2})\b",
        re.IGNORECASE,
    )
    _THREE_ROW_RE = re.compile(
        r"\b(3-row|three-row|7\s*seat|8\s*seat|family\s*of\s*[6-8])\b",
        re.IGNORECASE,
    )
    _TRADE_IN_RE = re.compile(
        r"(?:trad(?:e|ing)\s*(?:in)?\s*(?:a|my)?\s*)?(20[1-2]\d)\s+([A-Za-z]+)\s+([A-Za-z0-9-]+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        session_store: Optional[PersistentSessionStore] = None,
        redactor: Optional[PIIRedactor] = None,
    ) -> None:
        self.session_store = session_store or DEFAULT_SESSION_STORE
        self.redactor = redactor or DEFAULT_PII_REDACTOR
        self._pending_tasks: Set[asyncio.Task[Dict[str, Any]]] = set()

    @traced_operation("autodeal.memory.extract_customer_facts")
    def extract_customer_facts(
        self,
        turns_or_text: Union[str, Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Extracts structured, PII-redacted customer preference facts from turns.

        Args:
            turns_or_text: Raw transcript string or sequence of turn dicts.

        Returns:
            Dictionary of extracted customer attributes (`max_monthly_payment`,
            `down_payment`, `max_budget_price`, `credit_score`,
            `min_seating_capacity`, `preferred_body_styles`,
            `preferred_fuel_types`, `trade_in_mentioned`, `trade_in_vehicle`,
            and `sanitized_notes`).
        """
        if isinstance(turns_or_text, str):
            raw_text = turns_or_text
        else:
            raw_text = " \n ".join(
                str(t.get("content", "")) for t in turns_or_text if t.get("content")
            )

        clean_text = self.redactor.redact_text(raw_text)
        lower_text = clean_text.lower()

        facts: Dict[str, Any] = {}

        monthly_match = self._MONTHLY_PAYMENT_RE.search(clean_text)
        if monthly_match:
            facts["max_monthly_payment"] = float(
                monthly_match.group(1).replace(",", "")
            )

        down_match = self._DOWN_PAYMENT_RE.search(clean_text)
        if down_match:
            facts["down_payment"] = float(down_match.group(1).replace(",", ""))

        price_match = self._MAX_PRICE_RE.search(clean_text)
        if price_match:
            facts["max_budget_price"] = float(price_match.group(1).replace(",", ""))

        credit_match = self._CREDIT_SCORE_RE.search(clean_text)
        if credit_match:
            score_str = credit_match.group(1) or credit_match.group(2)
            facts["credit_score"] = int(score_str)

        if self._THREE_ROW_RE.search(clean_text):
            facts["min_seating_capacity"] = 7

        body_styles: List[str] = []
        for style in ("suv", "crossover", "sedan", "truck", "ev"):
            if re.search(rf"\b{style}s?\b", lower_text):
                body_styles.append(style)
        if body_styles:
            facts["preferred_body_styles"] = body_styles

        fuel_types: List[str] = []
        for fuel in ("hybrid", "phev", "ev", "gas"):
            if re.search(rf"\b{fuel}\b", lower_text):
                fuel_types.append(fuel)
        if "plug-in" in lower_text and "phev" not in fuel_types:
            fuel_types.append("phev")
        if "electric" in lower_text and "ev" not in fuel_types:
            fuel_types.append("ev")
        if fuel_types:
            facts["preferred_fuel_types"] = fuel_types

        if (
            "trade-in" in lower_text
            or "trade in" in lower_text
            or "trading in" in lower_text
        ):
            facts["trade_in_mentioned"] = True
            trade_match = self._TRADE_IN_RE.search(clean_text)
            if trade_match:
                facts["trade_in_vehicle"] = (
                    f"{trade_match.group(1)} {trade_match.group(2)} {trade_match.group(3)}"
                )

        facts["sanitized_notes"] = clean_text[:280].strip()
        return dict(self.redactor.redact_data(facts))

    async def _consolidate_async(
        self,
        session_id: str,
        turns_or_text: Union[str, Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Internal coroutine that extracts facts and persists them to SQLite."""
        log_intent(
            action_name="async_memory_consolidate",
            intended_arguments={"session_id": session_id},
            agent_name="async_memory_consolidator",
        )
        await asyncio.sleep(0)
        extracted = self.extract_customer_facts(turns_or_text)
        merged_profile = self.session_store.upsert_customer_profile(
            session_id=session_id,
            profile_updates=extracted,
        )
        log_outcome(
            action_name="async_memory_consolidate",
            status="success",
            actual_outcome={
                "session_id": session_id,
                "consolidated_keys": sorted(merged_profile.keys()),
            },
            agent_name="async_memory_consolidator",
        )
        return merged_profile

    def schedule_consolidation(
        self,
        session_id: str,
        turns_or_text: Union[str, Sequence[Dict[str, Any]]],
    ) -> asyncio.Task[Dict[str, Any]]:
        """Schedules non-blocking customer memory extraction via `asyncio.create_task`.

        Args:
            session_id: Unique conversation session identifier.
            turns_or_text: Conversation transcript or turn list to analyze.

        Returns:
            The scheduled `asyncio.Task` running in the background event loop.
        """
        task: asyncio.Task[Dict[str, Any]] = asyncio.create_task(
            self._consolidate_async(session_id=session_id, turns_or_text=turns_or_text)
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def wait_for_pending_tasks(
        self, timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Awaits completion of all active background memory consolidation tasks.

        Args:
            timeout: Maximum seconds to wait for background tasks.

        Returns:
            List of consolidated profile dictionaries from completed tasks.
        """
        if not self._pending_tasks:
            return []
        tasks_snapshot = list(self._pending_tasks)
        done, _ = await asyncio.wait(tasks_snapshot, timeout=timeout)
        results: List[Dict[str, Any]] = []
        for completed_task in done:
            if not completed_task.cancelled() and completed_task.exception() is None:
                results.append(completed_task.result())
        return results


DEFAULT_ASYNC_CONSOLIDATOR = AsyncMemoryConsolidator()
