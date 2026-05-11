"""
Unified Database Module for ASTRA Platform.

Consolidates all database operations into a single SQLite file (astra.db).
Tables: users, trust_scores, join_requests, secure_messages, used_tokens,
        experiments, metrics, fl_clients, notifications, groups, trained_models.
"""

import json
import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


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
            c.execute('''
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
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS trust_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    group_id TEXT,
                    score REAL DEFAULT 1.0,
                    quarantined BOOLEAN DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, group_id)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS join_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolved_by INTEGER REFERENCES users(id),
                    token_delivered BOOLEAN DEFAULT 0,
                    token_delivered_at TIMESTAMP,
                    request_nonce TEXT UNIQUE NOT NULL,
                    metadata_json TEXT
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS secure_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    to_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    group_id TEXT,
                    message_type TEXT NOT NULL,
                    encrypted_content BLOB NOT NULL,
                    nonce BLOB NOT NULL,
                    signature BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    read_at TIMESTAMP,
                    UNIQUE(from_user_id, to_user_id, group_id, message_type)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS used_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT UNIQUE NOT NULL,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            ''')
            
            # --- Experiments & Training ---
            c.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    config_json TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    group_id TEXT,
                    step INTEGER,
                    timestamp TEXT,
                    metrics_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS fl_clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT UNIQUE,
                    user_id INTEGER REFERENCES users(id),
                    group_id TEXT,
                    experiment_id TEXT,
                    status TEXT DEFAULT 'active',
                    trust_score REAL DEFAULT 1.0,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT
                )
            ''')
            
            # --- Notifications ---
            c.execute('''
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
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_notifications_user
                ON notifications(user_id, read, created_at)
            ''')
            
            # --- Groups (NEW - persistent) ---
            c.execute('''
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
            ''')
            
            # --- Trained Models (NEW - model persistence) ---
            c.execute('''
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
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_trained_models_group
                ON trained_models(group_id, model_type, version)
            ''')
            
            conn.commit()
            logger.info("[DB] Schema initialized in %s", self.db_path)
    
    # ========================================================================
    # Migration from legacy databases
    # ========================================================================
    
    def _migrate_legacy_dbs(self):
        """Migrate data from legacy .db files if they exist."""
        base_dir = os.path.dirname(self.db_path) or "."
        
        self._migrate_legacy_db(
            os.path.join(base_dir, "users.db"),
            ["users", "trust_scores", "join_requests", "secure_messages", "used_tokens"]
        )
        self._migrate_legacy_db(
            os.path.join(base_dir, "experiments.db"),
            ["experiments", "metrics"],
            rename_map={"clients": "fl_clients"}
        )
        self._migrate_legacy_db(
            os.path.join(base_dir, "notifications.db"),
            ["notifications"]
        )
    
    def _migrate_legacy_db(
        self,
        old_path: str,
        tables: List[str],
        rename_map: Optional[Dict[str, str]] = None
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
                logger.info("[DB] Migrated %s → %s (backup already existed)", old_path, self.db_path)
                
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
                try:
                    dst_cursor.execute(
                        f"INSERT OR IGNORE INTO {dst_table} ({col_list}) VALUES ({placeholders})",
                        values
                    )
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates
                    
        except Exception as e:
            logger.warning("[DB] Could not copy table %s → %s: %s", src_table, dst_table, e)
    
    # ========================================================================
    # Default admin
    # ========================================================================
    
    def _ensure_default_admin(self):
        """Create default admin user if not exists."""
        import bcrypt
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
            if not cursor.fetchone():
                pw = bcrypt.hashpw(b"adminpass", bcrypt.gensalt()).decode("utf-8")
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                    ("admin", pw, "admin", "System Admin")
                )
                conn.commit()
                logger.info("[DB] Default admin user created")
    
    # ========================================================================
    # Experiment methods (replaces ExperimentDB)
    # ========================================================================
    
    def create_experiment(self, experiment_id: str, config: Dict) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments (experiment_id, config_json, status, start_time) VALUES (?, ?, ?, ?)",
                (experiment_id, json.dumps(config), "pending", datetime.now().isoformat())
            )
            conn.commit()
    
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        end_time = datetime.now().isoformat() if status in ("completed", "failed") else None
        with self.connection() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, end_time = ? WHERE experiment_id = ?",
                (status, end_time, experiment_id)
            )
            conn.commit()
    
    def log_metrics(self, experiment_id: str, step: int, metrics: Dict, group_id: str = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO metrics (experiment_id, group_id, step, timestamp, metrics_json) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, group_id, step, datetime.now().isoformat(), json.dumps(metrics))
            )
            conn.commit()
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT step, timestamp, metrics_json FROM metrics WHERE experiment_id = ? ORDER BY step",
                (experiment_id,)
            )
            return [
                {"step": row["step"], "timestamp": row["timestamp"], **json.loads(row["metrics_json"])}
                for row in cursor.fetchall()
            ]
    
    # ========================================================================
    # FL Client methods (replaces ExperimentDB client methods)
    # ========================================================================
    
    def register_fl_client(self, client_id: str, experiment_id: str, user_id: int = None, group_id: str = None) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fl_clients 
                   (client_id, user_id, group_id, experiment_id, status, trust_score, last_seen)
                   VALUES (?, ?, ?, ?, 'active', 1.0, ?)""",
                (client_id, user_id, group_id, experiment_id, datetime.now().isoformat())
            )
            conn.commit()
    
    def update_fl_client(self, client_id: str, trust_score: float, status: str = "active") -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE fl_clients SET trust_score = ?, status = ?, last_seen = ? WHERE client_id = ?",
                (trust_score, status, datetime.now().isoformat(), client_id)
            )
            conn.commit()
    
    # ========================================================================
    # Group methods (NEW - persistent groups)
    # ========================================================================
    
    def create_group(self, group_id: str, model_id: str, config: Dict = None,
                     join_token: str = None, window_size: int = 5,
                     time_limit: int = 300, created_by: int = None) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO groups 
                   (group_id, model_id, status, join_token, config_json, window_size, time_limit, created_by, updated_at)
                   VALUES (?, ?, 'IDLE', ?, ?, ?, ?, ?, ?)""",
                (group_id, model_id, join_token, json.dumps(config or {}),
                 window_size, time_limit, created_by, datetime.now().isoformat())
            )
            conn.commit()
    
    def update_group_status(self, group_id: str, status: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE groups SET status = ?, updated_at = ? WHERE group_id = ?",
                (status, datetime.now().isoformat(), group_id)
            )
            conn.commit()
    
    def get_group(self, group_id: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM groups WHERE group_id = ?", (group_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_groups(self) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM groups ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]
    
    def delete_group(self, group_id: str) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # ========================================================================
    # Trained model methods (NEW - model persistence tracking)
    # ========================================================================
    
    def save_model_record(self, group_id: str, model_type: str, file_path: str,
                          version: int = 1, client_id: str = None,
                          accuracy: float = None, loss: float = None,
                          num_clients: int = None, metadata: Dict = None) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO trained_models 
                   (group_id, model_type, client_id, version, file_path, accuracy, loss, num_clients, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (group_id, model_type, client_id, version, file_path,
                 accuracy, loss, num_clients, json.dumps(metadata or {}))
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_model(self, group_id: str, model_type: str = "global") -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT * FROM trained_models 
                   WHERE group_id = ? AND model_type = ?
                   ORDER BY version DESC LIMIT 1""",
                (group_id, model_type)
            ).fetchone()
            return dict(row) if row else None
    
    def get_model_history(self, group_id: str, model_type: str = "global") -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM trained_models
                   WHERE group_id = ? AND model_type = ?
                   ORDER BY version DESC""",
                (group_id, model_type)
            ).fetchall()
            return [dict(r) for r in rows]


# ============================================================================
# Global singleton
# ============================================================================

_db: Optional[AstraDB] = None


def get_db() -> AstraDB:
    """Get the global AstraDB instance."""
    global _db
    if _db is None:
        _db = AstraDB()
    return _db


def init_db(db_path: str = "./astra.db") -> AstraDB:
    """Initialize the global AstraDB instance."""
    global _db
    _db = AstraDB(db_path)
    return _db
