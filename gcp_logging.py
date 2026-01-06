"""BigQuery logging with SQLite buffer for Gemini API calls."""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

import config

_LOGGER = logging.getLogger(__name__)

# Thread locks for global state
_client_lock = threading.Lock()
_id_lock = threading.Lock()

# BigQuery client (initialized lazily)
_bq_client = None

# Conversation tracking (protected by _id_lock)
_current_conversation_id: Optional[str] = None
_current_event_id: Optional[str] = None
_last_event_timestamp: Optional[datetime] = None

# SQLite connection (protected by _client_lock)
_sqlite_conn: Optional[sqlite3.Connection] = None


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class LogEntry:
    """A single log entry for BigQuery."""
    time_generated: datetime
    event_id: str
    conversation_id: str
    role: str  # 'user' or 'assistant'
    message: str
    camera_name: str
    event_type: str
    image_uri: Optional[str]
    model: Optional[str]
    risk_tier: Optional[str]
    recommended_action: Optional[str]
    person_detected: Optional[bool]
    person_count: Optional[int]
    time_of_day_apparent: Optional[str]
    summary: Optional[str]
    activity_observed: Optional[list[str]]
    potential_concerns: Optional[list[str]]
    context_clues: Optional[list[str]]
    weapon_detected: Optional[bool]
    weapon_confidence: Optional[float]
    weapon_description: Optional[str]


# =============================================================================
# SQLite Buffer
# =============================================================================

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS log_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_generated TEXT NOT NULL,
    event_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    camera_name TEXT,
    event_type TEXT,
    image_uri TEXT,
    model TEXT,
    risk_tier TEXT,
    recommended_action TEXT,
    person_detected INTEGER,
    person_count INTEGER,
    time_of_day_apparent TEXT,
    summary TEXT,
    activity_observed TEXT,
    potential_concerns TEXT,
    context_clues TEXT,
    weapon_detected INTEGER,
    weapon_confidence REAL,
    weapon_description TEXT,
    synced INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_synced ON log_buffer(synced);
CREATE INDEX IF NOT EXISTS idx_time_generated ON log_buffer(time_generated);
CREATE INDEX IF NOT EXISTS idx_event_id ON log_buffer(event_id);
CREATE INDEX IF NOT EXISTS idx_conversation_id ON log_buffer(conversation_id);
"""


def _get_sqlite_conn() -> sqlite3.Connection:
    """Get or create SQLite connection (thread-safe)."""
    global _sqlite_conn

    # Fast path without lock
    if _sqlite_conn is not None:
        return _sqlite_conn

    with _client_lock:
        # Double-check after acquiring lock
        if _sqlite_conn is not None:
            return _sqlite_conn

        db_path = config.SQLITE_BUFFER_FILE
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _sqlite_conn.row_factory = sqlite3.Row
        _sqlite_conn.executescript(SQLITE_SCHEMA)
        _sqlite_conn.commit()

        _LOGGER.info("SQLite buffer initialized: %s", db_path)
        return _sqlite_conn


def _insert_to_sqlite(entry: LogEntry) -> bool:
    """Insert a log entry to SQLite buffer."""
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO log_buffer (
                time_generated, event_id, conversation_id, role, message,
                camera_name, event_type, image_uri, model,
                risk_tier, recommended_action, person_detected, person_count,
                time_of_day_apparent, summary,
                activity_observed, potential_concerns, context_clues,
                weapon_detected, weapon_confidence, weapon_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.time_generated.isoformat(),
            entry.event_id,
            entry.conversation_id,
            entry.role,
            entry.message,
            entry.camera_name,
            entry.event_type,
            entry.image_uri,
            entry.model,
            entry.risk_tier,
            entry.recommended_action,
            1 if entry.person_detected else 0 if entry.person_detected is not None else None,
            entry.person_count,
            entry.time_of_day_apparent,
            entry.summary,
            json.dumps(entry.activity_observed) if entry.activity_observed else None,
            json.dumps(entry.potential_concerns) if entry.potential_concerns else None,
            json.dumps(entry.context_clues) if entry.context_clues else None,
            1 if entry.weapon_detected else 0 if entry.weapon_detected is not None else None,
            entry.weapon_confidence,
            entry.weapon_description,
        ))

        conn.commit()
        return True

    except Exception as e:
        _LOGGER.error("Failed to insert to SQLite: %s", e)
        return False


def _get_unsynced_entries(limit: int = 1000) -> list[dict]:
    """Get unsynced entries from SQLite."""
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM log_buffer
            WHERE synced = 0
            ORDER BY time_generated ASC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    except Exception as e:
        _LOGGER.error("Failed to get unsynced entries: %s", e)
        return []


def _mark_entries_synced(ids: list[int]) -> bool:
    """Mark entries as synced in SQLite."""
    if not ids:
        return True

    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(ids))
        cursor.execute(f"UPDATE log_buffer SET synced = 1 WHERE id IN ({placeholders})", ids)
        conn.commit()
        return True

    except Exception as e:
        _LOGGER.error("Failed to mark entries as synced: %s", e)
        return False


def _delete_old_synced_entries(days: int = None) -> int:
    """Delete old synced entries from SQLite."""
    if days is None:
        days = config.LOG_RETENTION_DAYS

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM log_buffer
            WHERE synced = 1 AND time_generated < ?
        """, (cutoff.isoformat(),))

        deleted = cursor.rowcount
        conn.commit()

        if deleted > 0:
            _LOGGER.info("Deleted %d old synced entries from SQLite", deleted)

        return deleted

    except Exception as e:
        _LOGGER.error("Failed to delete old entries: %s", e)
        return 0


# =============================================================================
# BigQuery
# =============================================================================

BQ_SCHEMA = [
    {"name": "IngestionTime", "type": "TIMESTAMP", "mode": "NULLABLE"},
    {"name": "TimeGenerated", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "EventId", "type": "STRING", "mode": "REQUIRED"},
    {"name": "ConversationId", "type": "STRING", "mode": "REQUIRED"},
    {"name": "Role", "type": "STRING", "mode": "REQUIRED"},
    {"name": "Message", "type": "STRING", "mode": "REQUIRED"},
    {"name": "CameraName", "type": "STRING", "mode": "NULLABLE"},
    {"name": "EventType", "type": "STRING", "mode": "NULLABLE"},
    {"name": "ImageUri", "type": "STRING", "mode": "NULLABLE"},
    {"name": "Model", "type": "STRING", "mode": "NULLABLE"},
    {"name": "RiskTier", "type": "STRING", "mode": "NULLABLE"},
    {"name": "RecommendedAction", "type": "STRING", "mode": "NULLABLE"},
    {"name": "PersonDetected", "type": "BOOL", "mode": "NULLABLE"},
    {"name": "PersonCount", "type": "INT64", "mode": "NULLABLE"},
    {"name": "TimeOfDayApparent", "type": "STRING", "mode": "NULLABLE"},
    {"name": "Summary", "type": "STRING", "mode": "NULLABLE"},
    {"name": "ActivityObserved", "type": "STRING", "mode": "REPEATED"},
    {"name": "PotentialConcerns", "type": "STRING", "mode": "REPEATED"},
    {"name": "ContextClues", "type": "STRING", "mode": "REPEATED"},
    {"name": "WeaponDetected", "type": "BOOL", "mode": "NULLABLE"},
    {"name": "WeaponConfidence", "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "WeaponDescription", "type": "STRING", "mode": "NULLABLE"},
]


def _get_bq_client():
    """Get or create BigQuery client (thread-safe)."""
    global _bq_client

    # Fast path without lock
    if _bq_client is not None:
        return _bq_client

    with _client_lock:
        # Double-check after acquiring lock
        if _bq_client is not None:
            return _bq_client

        try:
            from google.cloud import bigquery

            # Use service account file if specified (local development)
            if config.GCP_SERVICE_ACCOUNT_FILE and os.path.exists(config.GCP_SERVICE_ACCOUNT_FILE):
                _bq_client = bigquery.Client.from_service_account_json(
                    config.GCP_SERVICE_ACCOUNT_FILE
                )
                _LOGGER.info("BigQuery client initialized with service account file")
            else:
                # Use Application Default Credentials (production/GCE)
                _bq_client = bigquery.Client(project=config.GCP_PROJECT_ID or None)
                _LOGGER.info("BigQuery client initialized with ADC")

            return _bq_client

        except Exception as e:
            _LOGGER.error("Failed to initialize BigQuery client: %s", e)
            return None


def _ensure_dataset_and_table() -> bool:
    """Ensure BigQuery dataset and table exist."""
    client = _get_bq_client()
    if not client:
        return False

    try:
        from google.cloud import bigquery
        from google.api_core.exceptions import NotFound

        project = client.project
        dataset_id = f"{project}.{config.BQ_DATASET}"
        table_id = f"{dataset_id}.{config.BQ_TABLE}"

        # Create dataset if not exists
        try:
            client.get_dataset(dataset_id)
        except NotFound:
            _LOGGER.info("Creating BigQuery dataset: %s", dataset_id)
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            client.create_dataset(dataset)

        # Create table if not exists
        try:
            client.get_table(table_id)
        except NotFound:
            _LOGGER.info("Creating BigQuery table: %s", table_id)
            schema = [bigquery.SchemaField(**field) for field in BQ_SCHEMA]
            table = bigquery.Table(table_id, schema=schema)

            # Add partitioning and clustering
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="TimeGenerated",
            )
            table.clustering_fields = ["EventId", "ConversationId"]

            client.create_table(table)
            _LOGGER.info("BigQuery table created with partitioning and clustering")

        return True

    except Exception as e:
        _LOGGER.error("Failed to ensure BigQuery dataset/table: %s", e)
        return False


def _sqlite_row_to_bq_row(row: dict) -> dict:
    """Convert SQLite row to BigQuery row format."""
    # Parse arrays from JSON strings (handle null values gracefully)
    def parse_array(val):
        if not val:
            return []
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []

    activity = parse_array(row["activity_observed"])
    concerns = parse_array(row["potential_concerns"])
    clues = parse_array(row["context_clues"])

    return {
        "IngestionTime": datetime.now(timezone.utc).isoformat(),
        "TimeGenerated": row["time_generated"],
        "EventId": row["event_id"],
        "ConversationId": row["conversation_id"],
        "Role": row["role"],
        "Message": row["message"],
        "CameraName": row["camera_name"],
        "EventType": row["event_type"],
        "ImageUri": row["image_uri"],
        "Model": row["model"],
        "RiskTier": row["risk_tier"],
        "RecommendedAction": row["recommended_action"],
        "PersonDetected": bool(row["person_detected"]) if row["person_detected"] is not None else None,
        "PersonCount": row["person_count"],
        "TimeOfDayApparent": row["time_of_day_apparent"],
        "Summary": row["summary"],
        "ActivityObserved": activity,
        "PotentialConcerns": concerns,
        "ContextClues": clues,
        "WeaponDetected": bool(row["weapon_detected"]) if row["weapon_detected"] is not None else None,
        "WeaponConfidence": row["weapon_confidence"],
        "WeaponDescription": row["weapon_description"],
    }


def sync_to_bigquery() -> tuple[int, int]:
    """
    Sync unsynced entries from SQLite to BigQuery.

    Returns (synced_count, failed_count).
    """
    client = _get_bq_client()
    if not client:
        _LOGGER.debug("BigQuery not available, skipping sync")
        return 0, 0

    # Ensure table exists
    if not _ensure_dataset_and_table():
        return 0, 0

    # Get unsynced entries
    entries = _get_unsynced_entries(limit=1000)
    if not entries:
        return 0, 0

    _LOGGER.info("Syncing %d entries to BigQuery", len(entries))

    try:
        from google.cloud import bigquery

        table_id = f"{client.project}.{config.BQ_DATASET}.{config.BQ_TABLE}"

        # Convert to BigQuery format
        rows = [_sqlite_row_to_bq_row(entry) for entry in entries]

        # Insert rows
        errors = client.insert_rows_json(table_id, rows)

        if errors:
            # Parse errors to identify which rows failed
            # Each error dict contains 'index' field indicating failed row position
            failed_indices = set()
            for error in errors:
                if 'index' in error:
                    failed_indices.add(error['index'])

            # Separate successful and failed entry IDs
            successful_ids = []
            failed_count = 0

            for idx, entry in enumerate(entries):
                if idx in failed_indices:
                    failed_count += 1
                else:
                    successful_ids.append(entry["id"])

            # Mark only successful entries as synced
            if successful_ids:
                _mark_entries_synced(successful_ids)
                _LOGGER.info("Synced %d entries, %d failed (will retry)", len(successful_ids), failed_count)
            else:
                _LOGGER.error("All %d entries failed to sync: %s", len(entries), errors)

            return len(successful_ids), failed_count

        # All rows succeeded - mark all as synced
        ids = [entry["id"] for entry in entries]
        _mark_entries_synced(ids)

        _LOGGER.info("Successfully synced %d entries to BigQuery", len(entries))
        return len(entries), 0

    except Exception as e:
        _LOGGER.error("Failed to sync to BigQuery: %s", e)
        return 0, len(entries)


# =============================================================================
# Event & Conversation ID Management
# =============================================================================

def get_or_create_event_id(timestamp: datetime = None) -> str:
    """
    Get current event ID or create new one (thread-safe).

    Events are grouped if cameras trigger within EVENT_GROUPING_SECONDS.
    This is global - multiple cameras triggering within the window share the same event.
    """
    global _current_event_id, _last_event_timestamp

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    with _id_lock:
        # Check if we should create a new event
        if _last_event_timestamp is None:
            create_new = True
        else:
            elapsed = (timestamp - _last_event_timestamp).total_seconds()
            create_new = elapsed > config.EVENT_GROUPING_SECONDS

        if create_new:
            _current_event_id = f"evt_{uuid.uuid4().hex[:12]}"
            _LOGGER.debug("Created new event ID: %s", _current_event_id)

        _last_event_timestamp = timestamp
        return _current_event_id


def get_or_create_conversation_id(timestamp: datetime = None) -> str:
    """
    Get current conversation ID or create new one (thread-safe).

    Conversations are grouped within CONVERSATION_TIMEOUT_SECONDS.
    A new conversation starts after a gap longer than the timeout.
    """
    global _current_conversation_id

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    with _id_lock:
        # Use global last event time from event ID tracking
        last_time = _last_event_timestamp

        # Check if we should create a new conversation
        # Create new if: no existing ID, no last time, or timeout exceeded
        if _current_conversation_id is None:
            create_new = True
        elif last_time is None:
            create_new = True
        else:
            elapsed = (timestamp - last_time).total_seconds()
            create_new = elapsed > config.CONVERSATION_TIMEOUT_SECONDS

        if create_new:
            _current_conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            _LOGGER.debug("Created new conversation ID: %s", _current_conversation_id)

        return _current_conversation_id


# =============================================================================
# Public Logging API
# =============================================================================

async def log_user_prompt(
    prompt: str,
    camera_name: str,
    event_type: str,
    event_id: str,
    conversation_id: str,
    image_uri: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """
    Log a user prompt (our request to Gemini).

    This is non-blocking - writes to SQLite buffer.
    """
    entry = LogEntry(
        time_generated=datetime.now(timezone.utc),
        event_id=event_id,
        conversation_id=conversation_id,
        role="user",
        message=prompt,
        camera_name=camera_name,
        event_type=event_type,
        image_uri=image_uri,
        model=model,
        risk_tier=None,
        recommended_action=None,
        person_detected=None,
        person_count=None,
        time_of_day_apparent=None,
        summary=None,
        activity_observed=None,
        potential_concerns=None,
        context_clues=None,
        weapon_detected=None,
        weapon_confidence=None,
        weapon_description=None,
    )

    # Run in thread to not block
    return await asyncio.to_thread(_insert_to_sqlite, entry)


async def log_assistant_response(
    response_text: str,
    camera_name: str,
    event_type: str,
    event_id: str,
    conversation_id: str,
    model: str,
    risk_tier: Optional[str] = None,
    recommended_action: Optional[str] = None,
    person_detected: Optional[bool] = None,
    person_count: Optional[int] = None,
    time_of_day_apparent: Optional[str] = None,
    summary: Optional[str] = None,
    activity_observed: Optional[list[str]] = None,
    potential_concerns: Optional[list[str]] = None,
    context_clues: Optional[list[str]] = None,
    weapon_detected: Optional[bool] = None,
    weapon_confidence: Optional[float] = None,
    weapon_description: Optional[str] = None,
) -> bool:
    """
    Log an assistant response (Gemini's reply).

    This is non-blocking - writes to SQLite buffer.
    """
    entry = LogEntry(
        time_generated=datetime.now(timezone.utc),
        event_id=event_id,
        conversation_id=conversation_id,
        role="assistant",
        message=response_text,
        camera_name=camera_name,
        event_type=event_type,
        image_uri=None,  # Image already logged with user prompt
        model=model,
        risk_tier=risk_tier,
        recommended_action=recommended_action,
        person_detected=person_detected,
        person_count=person_count,
        time_of_day_apparent=time_of_day_apparent,
        summary=summary,
        activity_observed=activity_observed,
        potential_concerns=potential_concerns,
        context_clues=context_clues,
        weapon_detected=weapon_detected,
        weapon_confidence=weapon_confidence,
        weapon_description=weapon_description,
    )

    return await asyncio.to_thread(_insert_to_sqlite, entry)


async def run_sync() -> tuple[int, int, int]:
    """
    Run the sync process: upload to BigQuery and cleanup.

    Returns (synced_to_bq, failed_bq, deleted_old).
    """
    # Sync to BigQuery
    synced, failed = await asyncio.to_thread(sync_to_bigquery)

    # Delete old synced entries from SQLite
    deleted = await asyncio.to_thread(_delete_old_synced_entries)

    return synced, failed, deleted


# =============================================================================
# Testing
# =============================================================================

def test_bigquery_connection() -> tuple[bool, str]:
    """
    Test BigQuery connectivity.

    Returns (success, message).
    """
    if not config.GCP_PROJECT_ID and not config.GCP_SERVICE_ACCOUNT_FILE:
        return False, "GCP_PROJECT_ID or GCP_SERVICE_ACCOUNT_FILE not configured"

    try:
        client = _get_bq_client()
        if not client:
            return False, "Failed to create BigQuery client"

        # Try to ensure dataset/table exists
        if _ensure_dataset_and_table():
            table_id = f"{client.project}.{config.BQ_DATASET}.{config.BQ_TABLE}"
            return True, f"Connected to table: {table_id}"
        else:
            return False, "Failed to create dataset/table"

    except Exception as e:
        return False, f"BigQuery error: {e}"


def get_buffer_stats() -> dict:
    """Get statistics about the SQLite buffer."""
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM log_buffer WHERE synced = 0")
        unsynced = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM log_buffer WHERE synced = 1")
        synced = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM log_buffer")
        total = cursor.fetchone()[0]

        return {
            "total": total,
            "synced": synced,
            "unsynced": unsynced,
        }

    except Exception as e:
        _LOGGER.error("Failed to get buffer stats: %s", e)
        return {"total": 0, "synced": 0, "unsynced": 0}
