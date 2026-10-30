"""Persistent Session & Vehicle Catalogue Store (SQLite + Firestore compatible).

Provides durable persistence across turns and process restarts for:
1. Synthetic Dealership Vehicle Inventory (`vehicles` table seeded automatically)
2. Customer Conversational History (`session_turns` table with active PII scrubbing)
3. Consolidated Customer Preference Profiles (`customer_profiles` table)
4. Active Deal Desk / HITL Offer State (`offer_states` table)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import DEFAULT_CONFIG
from app.data.synthetic_catalog import SYNTHETIC_VEHICLES, VehicleRecord
from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation


class PersistentSessionStore:
    """Durable SQLite-backed session and vehicle catalogue store with PII redaction.

    Designed with a clean document/collection interface compatible with Google
    Cloud Firestore and Vertex AI Search backends while executing deterministically
    via local SQLite in unit tests and evaluation suites.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        redactor: Optional[PIIRedactor] = None,
        auto_seed_vehicles: bool = True,
    ) -> None:
        self.db_path = db_path or DEFAULT_CONFIG.sqlite_db_path
        self.redactor = redactor or DEFAULT_PII_REDACTOR
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self.db_path != ":memory:":
            parent_dir = os.path.dirname(os.path.abspath(self.db_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

        self._init_schema()
        if auto_seed_vehicles:
            self.seed_vehicles(SYNTHETIC_VEHICLES)

    def _connect(self) -> sqlite3.Connection:
        """Returns an active SQLite connection with row factory configured."""
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Creates SQLite tables for vehicles, session turns, profiles, and offers."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vehicles (
                        vin TEXT PRIMARY KEY,
                        stock_number TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        make TEXT NOT NULL,
                        model TEXT NOT NULL,
                        trim TEXT NOT NULL,
                        body_style TEXT NOT NULL,
                        fuel_type TEXT NOT NULL,
                        drivetrain TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        mileage INTEGER NOT NULL,
                        msrp REAL NOT NULL,
                        invoice_price REAL NOT NULL,
                        selling_price REAL NOT NULL,
                        seating_capacity INTEGER NOT NULL,
                        mpg_combined INTEGER NOT NULL,
                        safety_rating REAL NOT NULL,
                        key_features_json TEXT NOT NULL,
                        days_on_lot INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_turns (
                        turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_turns_session_id
                    ON session_turns(session_id, turn_id)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS customer_profiles (
                        session_id TEXT PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS offer_states (
                        session_id TEXT NOT NULL,
                        offer_key TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, offer_key)
                    )
                    """
                )
        finally:
            if self._memory_conn is None:
                conn.close()

    @traced_operation("autodeal.memory.seed_vehicles")
    def seed_vehicles(self, vehicles: List[VehicleRecord]) -> int:
        """Upserts synthetic vehicle records into the persistent `vehicles` table.

        Args:
            vehicles: List of validated `VehicleRecord` objects to seed.

        Returns:
            Total number of vehicles stored in the catalog table.
        """
        conn = self._connect()
        try:
            with conn:
                for v in vehicles:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO vehicles (
                            vin, stock_number, year, make, model, trim,
                            body_style, fuel_type, drivetrain, condition,
                            mileage, msrp, invoice_price, selling_price,
                            seating_capacity, mpg_combined, safety_rating,
                            key_features_json, days_on_lot
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            v.vin,
                            v.stock_number,
                            v.year,
                            v.make,
                            v.model,
                            v.trim,
                            v.body_style,
                            v.fuel_type,
                            v.drivetrain,
                            v.condition,
                            v.mileage,
                            v.msrp,
                            v.invoice_price,
                            v.selling_price,
                            v.seating_capacity,
                            v.mpg_combined,
                            v.safety_rating,
                            json.dumps(v.key_features),
                            v.days_on_lot,
                        ),
                    )
                row = conn.execute("SELECT COUNT(*) AS cnt FROM vehicles").fetchone()
                return int(row["cnt"]) if row else 0
        finally:
            if self._memory_conn is None:
                conn.close()

    @traced_operation("autodeal.memory.query_vehicles")
    def query_vehicles(
        self,
        body_style: Optional[str] = None,
        max_price: Optional[float] = None,
        min_seating: Optional[int] = None,
        fuel_type: Optional[str] = None,
        condition: Optional[str] = None,
        make: Optional[str] = None,
        feature_keyword: Optional[str] = None,
        limit: int = 25,
    ) -> List[VehicleRecord]:
        """Queries the persistent vehicle store using multi-attribute filters.

        Args:
            body_style: Optional filter (`'sedan'`, `'suv'`, `'crossover'`, `'truck'`, `'ev'`).
            max_price: Maximum `selling_price` in USD.
            min_seating: Minimum required `seating_capacity`.
            fuel_type: Optional fuel filter (`'gas'`, `'hybrid'`, `'phev'`, `'ev'`).
            condition: Optional condition filter (`'new'`, `'cpo'`).
            make: Optional manufacturer brand filter (case-insensitive).
            feature_keyword: Optional keyword matched against features or model name.
            limit: Maximum number of records returned (default 25).

        Returns:
            List of matching `VehicleRecord` objects sorted by `selling_price` ascending.
        """
        log_intent(
            action_name="query_vehicles",
            intended_arguments={
                "body_style": body_style,
                "max_price": max_price,
                "min_seating": min_seating,
                "fuel_type": fuel_type,
                "condition": condition,
                "make": make,
                "feature_keyword": feature_keyword,
                "limit": limit,
            },
            agent_name="inventory_specialist",
        )

        clauses: List[str] = ["1=1"]
        params: List[Any] = []

        if body_style and body_style.strip().lower() != "any":
            clauses.append("LOWER(body_style) = ?")
            params.append(body_style.strip().lower())

        if max_price is not None:
            clauses.append("selling_price <= ?")
            params.append(float(max_price))

        if min_seating is not None:
            clauses.append("seating_capacity >= ?")
            params.append(int(min_seating))

        if fuel_type and fuel_type.strip().lower() != "any":
            clauses.append("LOWER(fuel_type) = ?")
            params.append(fuel_type.strip().lower())

        if condition and condition.strip().lower() != "any":
            clauses.append("LOWER(condition) = ?")
            params.append(condition.strip().lower())

        if make and make.strip().lower() != "any":
            clauses.append("LOWER(make) = ?")
            params.append(make.strip().lower())

        sql = (
            "SELECT * FROM vehicles WHERE "
            + " AND ".join(clauses)
            + " ORDER BY selling_price ASC"
        )

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        results: List[VehicleRecord] = []
        keyword_norm = feature_keyword.strip().lower() if feature_keyword else None

        for row in rows:
            record = VehicleRecord(
                vin=row["vin"],
                stock_number=row["stock_number"],
                year=row["year"],
                make=row["make"],
                model=row["model"],
                trim=row["trim"],
                body_style=row["body_style"],
                fuel_type=row["fuel_type"],
                drivetrain=row["drivetrain"],
                condition=row["condition"],
                mileage=row["mileage"],
                msrp=row["msrp"],
                invoice_price=row["invoice_price"],
                selling_price=row["selling_price"],
                seating_capacity=row["seating_capacity"],
                mpg_combined=row["mpg_combined"],
                safety_rating=row["safety_rating"],
                key_features=json.loads(row["key_features_json"]),
                days_on_lot=row["days_on_lot"],
            )
            if keyword_norm:
                searchable_blob = (
                    f"{record.display_name} {' '.join(record.key_features)}".lower()
                )
                if keyword_norm not in searchable_blob:
                    continue
            results.append(record)
            if len(results) >= limit:
                break

        log_outcome(
            action_name="query_vehicles",
            status="success",
            actual_outcome={"matched_count": len(results)},
            agent_name="inventory_specialist",
        )
        return results

    @traced_operation("autodeal.memory.save_session_turn")
    def save_session_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persists a single conversation turn after scrubbing all customer PII.

        Args:
            session_id: Unique conversation session identifier.
            role: Speaker role (`'user'`, `'assistant'`, `'system'`).
            content: Raw message text (automatically scrubbed via `PIIRedactor`).
            metadata: Optional structured turn metadata (automatically scrubbed).

        Returns:
            Dictionary representing the sanitized, persisted turn record.
        """
        sanitized_content = self.redactor.redact_text(content)
        sanitized_metadata = self.redactor.redact_data(metadata or {})
        now_iso = datetime.now(timezone.utc).isoformat()

        log_intent(
            action_name="save_session_turn",
            intended_arguments={
                "session_id": session_id,
                "role": role,
                "content_preview": sanitized_content[:80],
            },
            agent_name="dealership_coordinator",
        )

        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO session_turns (session_id, role, content, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        sanitized_content,
                        json.dumps(sanitized_metadata),
                        now_iso,
                    ),
                )
                turn_id = cursor.lastrowid
        finally:
            if self._memory_conn is None:
                conn.close()

        saved_turn = {
            "turn_id": turn_id,
            "session_id": session_id,
            "role": role,
            "content": sanitized_content,
            "metadata": sanitized_metadata,
            "created_at": now_iso,
        }
        log_outcome(
            action_name="save_session_turn",
            status="success",
            actual_outcome={"session_id": session_id, "turn_id": turn_id},
            agent_name="dealership_coordinator",
        )
        return saved_turn

    @traced_operation("autodeal.memory.get_session_history")
    def get_session_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves chronological conversation turns for a session.

        Args:
            session_id: Unique conversation session identifier.
            limit: Optional maximum number of most-recent turns to return.

        Returns:
            Chronological list of turn dictionaries.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT turn_id, session_id, role, content, metadata_json, created_at
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id ASC
                """,
                (session_id,),
            ).fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        turns = [
            {
                "turn_id": row["turn_id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        if limit is not None and limit > 0:
            return turns[-limit:]
        return turns

    @traced_operation("autodeal.memory.replace_session_history")
    def replace_session_history(
        self, session_id: str, turns: List[Dict[str, Any]]
    ) -> None:
        """Replaces stored session turns with a compacted turn list.

        Args:
            session_id: Unique conversation session identifier.
            turns: Compacted list of turn dictionaries (`role`, `content`, `metadata`).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM session_turns WHERE session_id = ?", (session_id,)
                )
                for turn in turns:
                    clean_content = self.redactor.redact_text(
                        str(turn.get("content", ""))
                    )
                    clean_meta = self.redactor.redact_data(turn.get("metadata", {}))
                    conn.execute(
                        """
                        INSERT INTO session_turns (session_id, role, content, metadata_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            str(turn.get("role", "user")),
                            clean_content,
                            json.dumps(clean_meta),
                            str(turn.get("created_at", now_iso)),
                        ),
                    )
        finally:
            if self._memory_conn is None:
                conn.close()

    @traced_operation("autodeal.memory.get_customer_profile")
    def get_customer_profile(self, session_id: str) -> Dict[str, Any]:
        """Fetches the consolidated customer preference profile for a session.

        Args:
            session_id: Unique conversation session identifier.

        Returns:
            Dictionary of customer facts and preferences (or empty dict if none).
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT profile_json FROM customer_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()

        if row is None:
            return {}
        return dict(json.loads(row["profile_json"]))

    @traced_operation("autodeal.memory.upsert_customer_profile")
    def upsert_customer_profile(
        self, session_id: str, profile_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merges and persists customer preference facts after PII redaction.

        Args:
            session_id: Unique conversation session identifier.
            profile_updates: New or updated preference attributes to merge.

        Returns:
            The merged, PII-scrubbed customer profile dictionary.
        """
        sanitized_updates = self.redactor.redact_data(profile_updates)
        log_intent(
            action_name="upsert_customer_profile",
            intended_arguments={
                "session_id": session_id,
                "updated_keys": sorted(sanitized_updates.keys()),
            },
            agent_name="dealership_coordinator",
        )

        existing = self.get_customer_profile(session_id)
        merged = {**existing, **sanitized_updates}
        merged = self.redactor.redact_data(merged)
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO customer_profiles (session_id, profile_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, json.dumps(merged), now_iso),
                )
        finally:
            if self._memory_conn is None:
                conn.close()

        log_outcome(
            action_name="upsert_customer_profile",
            status="success",
            actual_outcome={
                "session_id": session_id,
                "profile_keys": sorted(merged.keys()),
            },
            agent_name="dealership_coordinator",
        )
        return merged

    @traced_operation("autodeal.memory.save_active_offer_state")
    def save_active_offer_state(
        self, session_id: str, offer_key: str, offer_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persists deal-desk offer state or HITL manager approval status.

        Args:
            session_id: Unique conversation session identifier.
            offer_key: Identifier for the offer or discount request (e.g., VIN).
            offer_payload: Offer state dictionary (scrubbed for PII before saving).

        Returns:
            Sanitized offer state dictionary.
        """
        sanitized_payload = self.redactor.redact_data(offer_payload)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO offer_states (session_id, offer_key, payload_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, offer_key, json.dumps(sanitized_payload), now_iso),
                )
        finally:
            if self._memory_conn is None:
                conn.close()
        return dict(sanitized_payload)

    @traced_operation("autodeal.memory.get_active_offer_state")
    def get_active_offer_state(
        self, session_id: str, offer_key: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a persisted deal-desk offer or HITL approval state.

        Args:
            session_id: Unique conversation session identifier.
            offer_key: Identifier for the offer or discount request.

        Returns:
            Stored offer payload dictionary if present, else `None`.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT payload_json FROM offer_states
                WHERE session_id = ? AND offer_key = ?
                """,
                (session_id, offer_key),
            ).fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()

        if row is None:
            return None
        return dict(json.loads(row["payload_json"]))


DEFAULT_SESSION_STORE = PersistentSessionStore()
