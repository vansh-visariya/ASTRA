"""
Unified Database Module for ASTRA Platform.

Consolidates all database operations into a single SQLite file (astra.db).
Tables: users, trust_scores, join_requests, secure_messages, used_tokens,
        experiments, metrics, fl_clients, notifications, groups, trained_models.
"""

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Thread-local storage for connections
_local = threading.local()


class AstraDB:
    """Unified database for the ASTRA platform.

    Manages all tables in a single astra.db file with thread-safe connections.
    On first startup, migrates data from legacy .db files if they exist.
    """

    def __init__(self, db_path: str = "./astra.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()
        self._migrate_legacy_dbs()
        self._ensure_default_admin()
        logger.info("AstraDB init: path=%s WAL=on", os.path.abspath(db_path))

    @contextmanager
    def connection(self):
        """Thread-safe context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ========================================================================
    # Schema
    # ========================================================================

    def _init_schema(self):
        """Create all tables if they don't exist."""
        with self.connection() as conn:
            c = conn.cursor()

            # --- Users & Auth ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'client', 'observer')),
                    email TEXT,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS trust_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    group_id TEXT,
                    score REAL DEFAULT 1.0,
                    quarantined BOOLEAN DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, group_id)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS join_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    status TEXT DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'rejected', 'activated')),
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolved_by INTEGER REFERENCES users(id),
                    token_delivered BOOLEAN DEFAULT 0,
                    token_delivered_at TIMESTAMP,
                    request_nonce TEXT UNIQUE NOT NULL,
                    metadata_json TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS used_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT UNIQUE NOT NULL,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)

            # --- Experiments & Training ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    config_json TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    group_id TEXT,
                    step INTEGER,
                    timestamp TEXT,
                    metrics_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS fl_clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT UNIQUE,
                    user_id INTEGER REFERENCES users(id),
                    group_id TEXT,
                    experiment_id TEXT,
                    status TEXT DEFAULT 'active',
                    trust_score REAL DEFAULT 1.0,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT,
                    local_accuracy REAL DEFAULT 0.0,
                    local_loss REAL DEFAULT 0.0,
                    updates_count INTEGER DEFAULT 0,
                    gradient_norm REAL DEFAULT 0.0,
                    last_update TEXT
                )
            """)

            # Migrate: add missing columns to existing fl_clients table
            for col, col_type in [
                ("local_accuracy", "REAL DEFAULT 0.0"),
                ("local_loss", "REAL DEFAULT 0.0"),
                ("updates_count", "INTEGER DEFAULT 0"),
                ("gradient_norm", "REAL DEFAULT 0.0"),
                ("last_update", "TEXT"),
            ]:
                with contextlib.suppress(sqlite3.OperationalError):
                    c.execute(
                        f"ALTER TABLE fl_clients ADD COLUMN {col} {col_type}"
                    )

            # --- Notifications ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notification_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    user_id INTEGER,
                    group_id TEXT,
                    data_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    read_at TIMESTAMP,
                    read BOOLEAN DEFAULT 0
                )
            """)

            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_notifications_user
                ON notifications(user_id, read, created_at)
            """)

            # --- Groups (NEW - persistent) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT UNIQUE NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT DEFAULT 'IDLE',
                    join_token TEXT,
                    config_json TEXT,
                    window_size INTEGER DEFAULT 5,
                    time_limit INTEGER DEFAULT 300,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER REFERENCES users(id)
                )
            """)

            # --- Trained Models (NEW - model persistence) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    model_type TEXT NOT NULL CHECK(model_type IN ('global', 'client')),
                    client_id TEXT,
                    version INTEGER DEFAULT 1,
                    file_path TEXT NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    num_clients INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT
                )
            """)

            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_trained_models_group
                ON trained_models(group_id, model_type, version)
            """)

            # --- Event Logs (persistent server event log) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS event_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    group_id TEXT,
                    details_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_logs_group
                ON event_logs(group_id, timestamp)
            """)

            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_logs_type
                ON event_logs(event_type, timestamp)
            """)

            # --- Model Registry (persist external model registrations across restarts) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    architecture_path TEXT,
                    total_params INTEGER,
                    trainable_params INTEGER,
                    is_peft INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'external',
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            logger.info("[DB] Schema initialized in %s", self.db_path)

        # Forward-compat migrations: extend CHECK constraints that newer
        # code paths need (e.g. join_requests.status='activated').
        # CREATE TABLE IF NOT EXISTS only runs on first init, so existing
        # databases need explicit ALTER TABLE.
        self._migrate_join_requests_status()

    def _migrate_join_requests_status(self):
        """Allow 'activated' as a join_requests.status value.

        Pre-existing CHECK constraint only allowed
        ('pending', 'approved', 'rejected'). The activate endpoint writes
        'activated' after a client joins the FL group; without this
        migration the update silently fails with a CHECK constraint error.
        """
        import sqlite3

        with self.connection() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='join_requests'"
            ).fetchone()
            if not row:
                return
            create_sql = row[0] or ""
            if "'activated'" in create_sql:
                return  # already migrated

            # SQLite can't ALTER a CHECK constraint in place. Recreate
            # the table with the new constraint and copy data over.
            try:
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("ALTER TABLE join_requests RENAME TO _join_requests__old")
                conn.execute(
                    """
                    CREATE TABLE join_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id TEXT NOT NULL,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        status TEXT DEFAULT 'pending'
                            CHECK(status IN ('pending', 'approved', 'rejected', 'activated')),
                        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP,
                        resolved_by INTEGER REFERENCES users(id),
                        token_delivered BOOLEAN DEFAULT 0,
                        token_delivered_at TIMESTAMP,
                        request_nonce TEXT UNIQUE NOT NULL,
                        metadata_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO join_requests
                        (id, group_id, user_id, status, requested_at, resolved_at,
                         resolved_by, token_delivered, token_delivered_at,
                         request_nonce, metadata_json)
                    SELECT id, group_id, user_id, status, requested_at, resolved_at,
                           resolved_by, token_delivered, token_delivered_at,
                           request_nonce, metadata_json
                    FROM _join_requests__old
                    """
                )
                conn.execute("DROP TABLE _join_requests__old")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.commit()
                logger.info("[DB] Migrated join_requests.status to allow 'activated'")
            except sqlite3.OperationalError as e:
                logger.warning("[DB] join_requests migration skipped: %s", e)

    # ========================================================================
    # Migration from legacy databases
    # ========================================================================

    def _migrate_legacy_dbs(self):
        """Migrate data from legacy .db files if they exist."""
        base_dir = os.path.dirname(self.db_path) or "."

        self._migrate_legacy_db(
            os.path.join(base_dir, "users.db"),
            ["users", "trust_scores", "join_requests", "secure_messages", "used_tokens"],
        )
        self._migrate_legacy_db(
            os.path.join(base_dir, "experiments.db"),
            ["experiments", "metrics"],
            rename_map={"clients": "fl_clients"},
        )
        self._migrate_legacy_db(os.path.join(base_dir, "notifications.db"), ["notifications"])

    def _migrate_legacy_db(
        self, old_path: str, tables: list[str], rename_map: dict[str, str] | None = None
    ):
        """Migrate tables from a legacy database file."""
        if not os.path.exists(old_path):
            return

        rename_map = rename_map or {}

        try:
            old_conn = sqlite3.connect(old_path)
            old_conn.row_factory = sqlite3.Row

            with self.connection() as new_conn:
                for table in tables:
                    self._copy_table(old_conn, new_conn, table, table)

                for old_table, new_table in rename_map.items():
                    self._copy_table(old_conn, new_conn, old_table, new_table)

                new_conn.commit()

            old_conn.close()

            # Rename old file to .bak
            bak_path = old_path + ".bak"
            if not os.path.exists(bak_path):
                shutil.move(old_path, bak_path)
                logger.info("[DB] Migrated %s → %s (backup: %s)", old_path, self.db_path, bak_path)
            else:
                os.remove(old_path)
                logger.info(
                    "[DB] Migrated %s → %s (backup already existed)", old_path, self.db_path
                )

        except Exception as e:
            logger.warning("[DB] Migration from %s failed: %s", old_path, e)

    def _copy_table(self, src_conn, dst_conn, src_table: str, dst_table: str):
        """Copy rows from a source table to a destination table."""
        try:
            src_cursor = src_conn.cursor()
            src_cursor.execute(f"SELECT * FROM {src_table}")
            rows = src_cursor.fetchall()

            if not rows:
                return

            # Get column names from source
            src_cols = [desc[0] for desc in src_cursor.description]

            # Get column names from destination
            dst_cursor = dst_conn.cursor()
            dst_cursor.execute(f"PRAGMA table_info({dst_table})")
            dst_cols = {row[1] for row in dst_cursor.fetchall()}

            # Only copy columns that exist in both
            common_cols = [c for c in src_cols if c in dst_cols]
            if not common_cols:
                return

            col_list = ", ".join(common_cols)
            placeholders = ", ".join(["?"] * len(common_cols))

            for row in rows:
                values = [row[c] for c in common_cols]
                with contextlib.suppress(sqlite3.IntegrityError):
                    dst_cursor.execute(
                        f"INSERT OR IGNORE INTO {dst_table} ({col_list}) VALUES ({placeholders})",
                        values,
                    )

        except Exception as e:
            logger.warning("[DB] Could not copy table %s → %s: %s", src_table, dst_table, e)

    # ========================================================================
    # Default admin
    # ========================================================================

    def _ensure_default_admin(self):
        """Create default admin user if not exists."""
        import bcrypt

        default_password = os.getenv("ASTRA_DEFAULT_ADMIN_PASSWORD", "adminpass")
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
            if not cursor.fetchone():
                pw = bcrypt.hashpw(default_password.encode("utf-8"), bcrypt.gensalt()).decode(
                    "utf-8"
                )
                cursor.execute(
                    "INSERT INTO users (username, password_hash,"
                    " role, full_name) VALUES (?, ?, ?, ?)",
                    ("admin", pw, "admin", "System Admin"),
                )
                conn.commit()
                logger.info("[DB] Default admin user created")

    # ========================================================================
    # Experiment methods (replaces ExperimentDB)
    # ========================================================================

    def create_experiment(self, experiment_id: str, config: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments"
                " (experiment_id, config_json, status, start_time)"
                " VALUES (?, ?, ?, ?)",
                (experiment_id, json.dumps(config), "pending", datetime.now().isoformat()),
            )
            conn.commit()

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        end_time = datetime.now().isoformat() if status in ("completed", "failed") else None
        with self.connection() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, end_time = ? WHERE experiment_id = ?",
                (status, end_time, experiment_id),
            )
            conn.commit()

    def log_metrics(
        self, experiment_id: str, step: int, metrics: dict, group_id: str | None = None
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO metrics"
                " (experiment_id, group_id, step, timestamp, metrics_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (experiment_id, group_id, step, datetime.now().isoformat(), json.dumps(metrics)),
            )
            conn.commit()

    def get_experiment_metrics(self, experiment_id: str) -> list[dict]:
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT step, timestamp, metrics_json"
                " FROM metrics WHERE experiment_id = ?"
                " ORDER BY step",
                (experiment_id,),
            )
            return [
                {
                    "step": row["step"],
                    "timestamp": row["timestamp"],
                    **json.loads(row["metrics_json"]),
                }
                for row in cursor.fetchall()
            ]

    # ====================================================================================
    # Event Logs
    # ====================================================================================

    def log_event(
        self,
        event_type: str,
        message: str,
        timestamp: float | None = None,
        group_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO event_logs"
                " (timestamp, event_type, message, group_id, details_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp or time.time(),
                    event_type,
                    message,
                    group_id,
                    json.dumps(details) if details else None,
                ),
            )
            conn.commit()

    def get_logs(
        self, limit: int = 100, event_type: str | None = None, group_id: str | None = None
    ) -> list[dict]:
        query = (
            "SELECT id, timestamp, event_type, message, group_id, details_json, created_at "
            "FROM event_logs WHERE 1=1"
        )
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if group_id:
            query += " AND group_id = ?"
            params.append(group_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        # Return most recent first (DB returns DESC, reverse to get ASC for UI)
        result = []
        for row in rows:
            result.append({
                "timestamp": row["timestamp"],
                "type": row["event_type"],
                "message": row["message"],
                "group_id": row["group_id"],
                "details": json.loads(row["details_json"]) if row["details_json"] else {},
            })
        return result[::-1]

    # ====================================================================================
    # FL Client methods (replaces ExperimentDB client methods)
    # ====================================================================================

    def register_fl_client(
        self,
        client_id: str,
        experiment_id: str,
        user_id: int | None = None,
        group_id: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fl_clients
                   (client_id, user_id, group_id, experiment_id, status, trust_score, last_seen)
                   VALUES (?, ?, ?, ?, 'active', 1.0, ?)""",
                (client_id, user_id, group_id, experiment_id, datetime.now().isoformat()),
            )
            conn.commit()

    def update_fl_client_metrics(
        self,
        client_id: str,
        local_accuracy: float | None = None,
        local_loss: float | None = None,
        updates_count: int | None = None,
        gradient_norm: float | None = None,
        status: str | None = None,
    ) -> None:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            updates: list[str] = []
            values: list[Any] = []

            if local_accuracy is not None:
                updates.append("local_accuracy = ?")
                values.append(local_accuracy)
            if local_loss is not None:
                updates.append("local_loss = ?")
                values.append(local_loss)
            if updates_count is not None:
                updates.append("updates_count = ?")
                values.append(updates_count)
            if gradient_norm is not None:
                updates.append("gradient_norm = ?")
                values.append(gradient_norm)
            if status is not None:
                updates.append("status = ?")
                values.append(status)

            updates.append("last_update = ?")
            values.append(now)
            updates.append("last_seen = ?")
            values.append(now)

            values.append(client_id)

            cursor.execute(
                f"UPDATE fl_clients SET {', '.join(updates)} WHERE client_id = ?",
                values,
            )
            conn.commit()

    # ========================================================================
    # Group methods (NEW - persistent groups)
    # ========================================================================

    def create_group(
        self,
        group_id: str,
        model_id: str,
        config: dict | None = None,
        join_token: str | None = None,
        window_size: int = 5,
        time_limit: int = 300,
        created_by: int | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO groups
                   (group_id, model_id, status, join_token,
                    config_json, window_size, time_limit,
                    created_by, updated_at)
                   VALUES (?, ?, 'IDLE', ?, ?, ?, ?, ?, ?)""",
                (
                    group_id,
                    model_id,
                    join_token,
                    json.dumps(config or {}),
                    window_size,
                    time_limit,
                    created_by,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def update_group_status(self, group_id: str, status: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE groups SET status = ?, updated_at = ? WHERE group_id = ?",
                (status, datetime.now().isoformat(), group_id),
            )
            conn.commit()

    def get_group(self, group_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM groups WHERE group_id = ?", (group_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_all_groups(self) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM groups ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ========================================================================
    # Model Registry persistence
    # ========================================================================

    def save_model_registration(
        self,
        model_id: str,
        architecture: str,
        architecture_path: str | None = None,
        config_json: str = "{}",
        is_huggingface: bool = False,
    ):
        """Save a model registration that survives server restarts.

        For existing registries, stores just the essentials needed to reload.
        """
        import json as jsonlib

        config = jsonlib.loads(config_json) if isinstance(config_json, str) else config_json
        source = "huggingface" if is_huggingface else "external"
        model_type = config.get("model_type", "unknown")
        total_params = config.get("total_params", 0)
        trainable_params = config.get("trainable_params", 0)
        is_peft = 1 if config.get("use_peft") else 0

        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_registry
                   (model_id, model_type, architecture, architecture_path,
                    total_params, trainable_params, is_peft, source, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_id,
                    model_type,
                    architecture,
                    architecture_path,
                    total_params,
                    trainable_params,
                    is_peft,
                    source,
                    jsonlib.dumps(config),
                ),
            )
            conn.commit()

    def load_model_registrations(self) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM model_registry ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_group(self, group_id: str) -> bool:
        with self.connection() as conn:
            conn.execute("DELETE FROM join_requests WHERE group_id = ?", (group_id,))
            conn.execute("DELETE FROM fl_clients WHERE group_id = ?", (group_id,))
            cursor = conn.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ========================================================================
    # Trained model methods (NEW - model persistence tracking)
    # ========================================================================

    def save_model_record(
        self,
        group_id: str,
        model_type: str,
        file_path: str,
        version: int = 1,
        client_id: str | None = None,
        accuracy: float | None = None,
        loss: float | None = None,
        num_clients: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO trained_models
                   (group_id, model_type, client_id, version,
                    file_path, accuracy, loss, num_clients,
                    metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    group_id,
                    model_type,
                    client_id,
                    version,
                    file_path,
                    accuracy,
                    loss,
                    num_clients,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_model(self, group_id: str, model_type: str = "global") -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT * FROM trained_models
                   WHERE group_id = ? AND model_type = ?
                   ORDER BY version DESC LIMIT 1""",
                (group_id, model_type),
            ).fetchone()
            return dict(row) if row else None

    def get_model_history(self, group_id: str, model_type: str = "global") -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM trained_models
                   WHERE group_id = ? AND model_type = ?
                   ORDER BY version DESC""",
                (group_id, model_type),
            ).fetchall()
            return [dict(r) for r in rows]


# ============================================================================
# Global singleton
# ============================================================================

_db: AstraDB | None = None


def get_db() -> AstraDB:
    """Get the global AstraDB instance.

    Honors the `ASTRA_DB_PATH` environment variable so tests can redirect
    the DB to a temporary file without touching the developer's real
    `astra.db`. If the singleton was already initialized under a different
    path, it is re-initialized.
    """
    global _db
    env_path = os.environ.get("ASTRA_DB_PATH")
    if _db is None:
        _db = AstraDB(env_path or "./astra.db")
    elif env_path and env_path != _db.db_path:
        # Singleton was created with the default path but the env now asks
        # for a different one — re-init. This happens when tests redirect
        # the DB after `get_db()` has already been called.
        _db = AstraDB(env_path)
    return _db
