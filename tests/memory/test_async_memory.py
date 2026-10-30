"""Unit tests for app/memory/async_memory.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.memory.async_memory import AsyncMemoryConsolidator
from app.memory.session_store import PersistentSessionStore


class TestAsyncMemoryConsolidator:
    """Tests for non-blocking background customer fact extraction and PII scrubbing."""

    @pytest.mark.asyncio
    async def test_async_schedule_consolidation_extracts_facts_and_scrubs_pii(
        self, tmp_path: Path
    ) -> None:
        db_file = str(tmp_path / "async_memory.db")
        store = PersistentSessionStore(db_path=db_file, auto_seed_vehicles=False)
        consolidator = AsyncMemoryConsolidator(session_store=store)

        session_id = "sess-async-001"
        turns = [
            {
                "role": "user",
                "content": (
                    "Customer needs a 3-row hybrid SUV under $650/mo with $4,000 down, "
                    "budget under $50,000, 740 credit score, trading in a 2021 Honda CR-V. "
                    "SSN: 123-45-6789, Email: vip@customer.com."
                ),
            }
        ]

        task = consolidator.schedule_consolidation(
            session_id=session_id, turns_or_text=turns
        )
        assert task is not None

        completed_profiles = await consolidator.wait_for_pending_tasks(timeout=5.0)
        assert len(completed_profiles) == 1

        stored_profile = store.get_customer_profile(session_id)
        assert stored_profile["max_monthly_payment"] == 650.0
        assert stored_profile["down_payment"] == 4000.0
        assert stored_profile["max_budget_price"] == 50000.0
        assert stored_profile["credit_score"] == 740
        assert stored_profile["min_seating_capacity"] == 7
        assert "suv" in stored_profile["preferred_body_styles"]
        assert "hybrid" in stored_profile["preferred_fuel_types"]
        assert stored_profile["trade_in_mentioned"] is True
        assert stored_profile["trade_in_vehicle"] == "2021 Honda CR-V"
        assert "123-45-6789" not in stored_profile["sanitized_notes"]
        assert "vip@customer.com" not in stored_profile["sanitized_notes"]
        assert "[REDACTED_SSN]" in stored_profile["sanitized_notes"]
        assert "[REDACTED_EMAIL]" in stored_profile["sanitized_notes"]
