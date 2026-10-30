"""Unit tests for app/memory/session_store.py."""

from __future__ import annotations

from pathlib import Path

from app.memory.session_store import PersistentSessionStore


class TestPersistentSessionStore:
    """Tests for SQLite persistence across store instances, vehicle queries, and PII scrubbing."""

    def test_vehicle_seeding_and_filtering(self, tmp_path: Path) -> None:
        db_file = str(tmp_path / "test_inventory.db")
        store = PersistentSessionStore(db_path=db_file)

        hybrid_suvs = store.query_vehicles(
            body_style="suv",
            fuel_type="hybrid",
            min_seating=7,
            max_price=50000.0,
        )
        assert len(hybrid_suvs) >= 2
        for v in hybrid_suvs:
            assert v.body_style == "suv"
            assert v.fuel_type == "hybrid"
            assert v.seating_capacity >= 7
            assert v.selling_price <= 50000.0

        cpo_units = store.query_vehicles(condition="cpo")
        assert len(cpo_units) >= 2
        assert all(v.condition == "cpo" for v in cpo_units)

        captain_chair_units = store.query_vehicles(feature_keyword="captain chairs")
        assert len(captain_chair_units) >= 2

    def test_session_persistence_across_store_instances_with_pii_redaction(
        self, tmp_path: Path
    ) -> None:
        db_file = str(tmp_path / "persistent_sessions.db")
        session_id = "sess-family-suv-001"

        store_instance_1 = PersistentSessionStore(db_path=db_file)
        store_instance_1.save_session_turn(
            session_id=session_id,
            role="user",
            content=(
                "Customer needs a 3-row hybrid SUV under $650/mo. "
                "Email: buyer@family.org, SSN: 123-45-6789, Phone: 415-555-0199."
            ),
            metadata={"ssn": "123-45-6789", "salesperson": "Alex"},
        )
        store_instance_1.upsert_customer_profile(
            session_id=session_id,
            profile_updates={
                "max_monthly_payment": 650.0,
                "min_seating_capacity": 7,
                "contact_email": "buyer@family.org",
                "driver_license": "DL: CA99887766",
            },
        )
        store_instance_1.save_active_offer_state(
            session_id=session_id,
            offer_key="STK-2026-001",
            offer_payload={
                "vin": "5TDKDRBH1TS100001",
                "proposed_discount": 2500.0,
                "approval_status": "PENDING_MANAGER_APPROVAL",
                "customer_phone": "415-555-0199",
            },
        )

        # Re-open a fresh PersistentSessionStore pointing at the same SQLite file
        store_instance_2 = PersistentSessionStore(
            db_path=db_file, auto_seed_vehicles=False
        )
        history = store_instance_2.get_session_history(session_id)
        assert len(history) == 1
        assert "123-45-6789" not in history[0]["content"]
        assert "buyer@family.org" not in history[0]["content"]
        assert "415-555-0199" not in history[0]["content"]
        assert "[REDACTED_SSN]" in history[0]["content"]
        assert "[REDACTED_EMAIL]" in history[0]["content"]
        assert "[REDACTED_PHONE]" in history[0]["content"]
        assert history[0]["metadata"]["ssn"] == "[REDACTED_SSN]"

        profile = store_instance_2.get_customer_profile(session_id)
        assert profile["max_monthly_payment"] == 650.0
        assert profile["min_seating_capacity"] == 7
        assert profile["contact_email"] == "[REDACTED_EMAIL]"
        assert profile["driver_license"] == "[REDACTED_DRIVER_LICENSE]"

        offer_state = store_instance_2.get_active_offer_state(
            session_id, "STK-2026-001"
        )
        assert offer_state is not None
        assert offer_state["approval_status"] == "PENDING_MANAGER_APPROVAL"
        assert offer_state["customer_phone"] == "[REDACTED_PHONE]"
