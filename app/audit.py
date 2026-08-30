from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(__file__).resolve().parent.parent / "recoverai.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                received_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                processing_result TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_entries (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                payment_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                failure_reason TEXT,
                model_probabilities TEXT,
                candidate_actions TEXT,
                selected_action TEXT,
                policy_decision TEXT,
                execution_status TEXT,
                outcome TEXT,
                recovered_amount REAL,
                recovery_time_minutes INTEGER,
                human_override TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def has_processed_event(event_id: str) -> bool:
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM webhook_events WHERE event_id = ? AND status IN ('processed', 'duplicate')",
            (event_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def record_webhook_event(event_id: str, event_type: str, payload: Dict[str, Any], status: str, processing_result: Optional[Dict[str, Any]] = None) -> None:
    ensure_schema()
    conn = _connect()
    try:
        payload_hash = compute_payload_hash(payload)
        conn.execute(
            """
            INSERT OR REPLACE INTO webhook_events (event_id, event_type, received_at, processed_at, status, payload_hash, processing_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                status,
                payload_hash,
                json.dumps(processing_result or {}, sort_keys=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def append_audit_entry(entry: Dict[str, Any]) -> None:
    ensure_schema()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO audit_entries (
                event_id, payment_id, timestamp, failure_reason, model_probabilities,
                candidate_actions, selected_action, policy_decision, execution_status,
                outcome, recovered_amount, recovery_time_minutes, human_override
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("event_id"),
                entry.get("payment_id"),
                entry.get("timestamp"),
                entry.get("failure_reason"),
                json.dumps(entry.get("model_probabilities") or {}, sort_keys=True),
                json.dumps(entry.get("candidate_actions") or [], sort_keys=True),
                entry.get("selected_action"),
                json.dumps(entry.get("policy_decision") or {}, sort_keys=True),
                entry.get("execution_status"),
                entry.get("outcome"),
                entry.get("recovered_amount"),
                entry.get("recovery_time_minutes"),
                entry.get("human_override"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_audit_records(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    ensure_schema()
    conn = _connect()
    try:
        query = "SELECT * FROM audit_entries"
        params: List[Any] = []
        clauses = []
        filters = filters or {}
        if filters.get("payment_id"):
            clauses.append("payment_id = ?")
            params.append(filters["payment_id"])
        if filters.get("action"):
            clauses.append("selected_action = ?")
            params.append(filters["action"])
        if filters.get("status"):
            clauses.append("execution_status = ?")
            params.append(filters["status"])
        if filters.get("recovered") is not None:
            clauses.append("outcome = ?")
            params.append("RECOVERED" if bool(filters["recovered"]) else "FAILED")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY audit_id DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
