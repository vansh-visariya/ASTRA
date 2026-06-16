"""
Unit tests for AstraDB: CRUD operations on all tables.
"""

import json
import os
import sqlite3
import tempfile

import pytest

from astra.app.database import AstraDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = AstraDB(db_path=path)
    yield db
    os.unlink(path)


class TestUsers:
    def test_create_and_get_user(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("testuser", "hash123", "client"),
            )
            conn.commit()

            c.execute("SELECT * FROM users WHERE username = ?", ("testuser",))
            row = c.fetchone()
            assert row is not None
            assert row["username"] == "testuser"
            assert row["role"] == "client"

    def test_unique_username(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("dupe", "hash", "client"),
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                c.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    ("dupe", "hash2", "admin"),
                )
                conn.commit()

    def test_role_constraint(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            with pytest.raises(sqlite3.IntegrityError):
                c.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    ("badrole", "hash", "invalid_role"),
                )
                conn.commit()

    def test_default_admin_exists(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
            row = c.fetchone()
            assert row is not None
            assert row["role"] == "admin"


class TestGroups:
    def test_create_group(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            config = json.dumps({"window_size": 5, "time_limit": 300})
            c.execute(
                "INSERT INTO groups (group_id, model_id, status, config_json, window_size) VALUES (?, ?, ?, ?, ?)",
                ("grp1", "simple_cnn_mnist", "ACTIVE", config, 5),
            )
            conn.commit()

            c.execute("SELECT * FROM groups WHERE group_id = ?", ("grp1",))
            row = c.fetchone()
            assert row is not None
            assert row["model_id"] == "simple_cnn_mnist"
            assert row["status"] == "ACTIVE"

    def test_unique_group_id(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO groups (group_id, model_id, status) VALUES (?, ?, ?)",
                ("same_id", "simple_cnn_mnist", "IDLE"),
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                c.execute(
                    "INSERT INTO groups (group_id, model_id, status) VALUES (?, ?, ?)",
                    ("same_id", "other_model", "IDLE"),
                )
                conn.commit()

    def test_update_group_status(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO groups (group_id, model_id, status) VALUES (?, ?, ?)",
                ("grp2", "simple_cnn_mnist", "IDLE"),
            )
            conn.commit()

            c.execute(
                "UPDATE groups SET status = ? WHERE group_id = ?",
                ("TRAINING", "grp2"),
            )
            conn.commit()

            c.execute("SELECT status FROM groups WHERE group_id = ?", ("grp2",))
            assert c.fetchone()["status"] == "TRAINING"

    def test_delete_group(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO groups (group_id, model_id, status) VALUES (?, ?, ?)",
                ("del_me", "simple_cnn_mnist", "IDLE"),
            )
            conn.commit()

            c.execute("DELETE FROM groups WHERE group_id = ?", ("del_me",))
            conn.commit()

            c.execute("SELECT * FROM groups WHERE group_id = ?", ("del_me",))
            assert c.fetchone() is None


class TestJoinRequests:
    def test_create_request(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("client99", "hash", "client"),
            )
            uid = c.lastrowid
            conn.commit()

            c.execute(
                "INSERT INTO join_requests (group_id, user_id, status, request_nonce) VALUES (?, ?, ?, ?)",
                ("grp1", uid, "pending", "test_nonce_abc"),
            )
            conn.commit()

            c.execute(
                "SELECT * FROM join_requests WHERE group_id = ? AND user_id = ?",
                ("grp1", uid),
            )
            row = c.fetchone()
            assert row is not None
            assert row["status"] == "pending"

    def test_approve_request(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("approve_me", "hash", "client"),
            )
            uid = c.lastrowid
            conn.commit()

            c.execute(
                "INSERT INTO join_requests (group_id, user_id, status, request_nonce) VALUES (?, ?, ?, ?)",
                ("grp1", uid, "pending", "test_nonce_def"),
            )
            req_id = c.lastrowid
            conn.commit()

            c.execute(
                "UPDATE join_requests SET status = 'approved' WHERE id = ?",
                (req_id,),
            )
            conn.commit()

            c.execute("SELECT status FROM join_requests WHERE id = ?", (req_id,))
            assert c.fetchone()["status"] == "approved"

    def test_reject_request(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("reject_me", "hash", "client"),
            )
            uid = c.lastrowid
            conn.commit()

            c.execute(
                "INSERT INTO join_requests (group_id, user_id, status, request_nonce) VALUES (?, ?, ?, ?)",
                ("grp1", uid, "pending", "test_nonce_ghi"),
            )
            req_id = c.lastrowid
            conn.commit()

            c.execute(
                "UPDATE join_requests SET status = 'rejected' WHERE id = ?",
                (req_id,),
            )
            conn.commit()

            c.execute("SELECT status FROM join_requests WHERE id = ?", (req_id,))
            assert c.fetchone()["status"] == "rejected"


class TestFLCLients:
    def test_register_client(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO fl_clients (client_id, group_id, status) VALUES (?, ?, ?)",
                ("c1", "grp1", "active"),
            )
            conn.commit()

            c.execute("SELECT * FROM fl_clients WHERE client_id = ?", ("c1",))
            row = c.fetchone()
            assert row is not None
            assert row["group_id"] == "grp1"
            assert row["trust_score"] == 1.0

    def test_unique_client_id(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO fl_clients (client_id, group_id) VALUES (?, ?)",
                ("unique_c", "grp1"),
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                c.execute(
                    "INSERT INTO fl_clients (client_id, group_id) VALUES (?, ?)",
                    ("unique_c", "grp2"),
                )
                conn.commit()


class TestTrustScores:
    def test_insert_and_update(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("trust_me", "hash", "client"),
            )
            uid = c.lastrowid
            conn.commit()

            c.execute(
                "INSERT INTO trust_scores (user_id, group_id, score) VALUES (?, ?, ?)",
                (uid, "grp1", 0.8),
            )
            conn.commit()

            c.execute(
                "SELECT score FROM trust_scores WHERE user_id = ? AND group_id = ?",
                (uid, "grp1"),
            )
            assert c.fetchone()["score"] == 0.8

    def test_quarantine_flag(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("quar", "hash", "client"),
            )
            uid = c.lastrowid
            conn.commit()

            c.execute(
                "INSERT INTO trust_scores (user_id, group_id, score, quarantined) VALUES (?, ?, ?, ?)",
                (uid, "grp1", 0.2, 1),
            )
            conn.commit()

            c.execute(
                "SELECT quarantined FROM trust_scores WHERE user_id = ?",
                (uid,),
            )
            assert c.fetchone()["quarantined"] == 1


class TestNotifications:
    def test_create_notification(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO notifications (notification_type, priority, title, message) VALUES (?, ?, ?, ?)",
                ("join_request", "high", "New request", "Client wants to join"),
            )
            conn.commit()

            c.execute("SELECT * FROM notifications")
            row = c.fetchone()
            assert row is not None
            assert row["title"] == "New request"
            assert row["read"] == 0

    def test_mark_read(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO notifications (notification_type, priority, title, message) VALUES (?, ?, ?, ?)",
                ("info", "low", "Test", "Body"),
            )
            nid = c.lastrowid
            conn.commit()

            c.execute(
                "UPDATE notifications SET read = 1 WHERE id = ?",
                (nid,),
            )
            conn.commit()

            c.execute("SELECT read FROM notifications WHERE id = ?", (nid,))
            assert c.fetchone()["read"] == 1


class TestTrainedModels:
    def test_insert_model(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO trained_models (group_id, model_type, file_path, accuracy, loss) VALUES (?, ?, ?, ?, ?)",
                ("grp1", "global", "/models/grp1_v1.pt", 0.92, 0.23),
            )
            conn.commit()

            c.execute(
                "SELECT * FROM trained_models WHERE group_id = ? AND model_type = ?",
                ("grp1", "global"),
            )
            row = c.fetchone()
            assert row is not None
            assert row["accuracy"] == 0.92
            assert row["loss"] == 0.23

    def test_client_model(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO trained_models (group_id, model_type, client_id, file_path) VALUES (?, ?, ?, ?)",
                ("grp1", "client", "c1", "/models/c1_v1.pt"),
            )
            conn.commit()

            c.execute("SELECT * FROM trained_models WHERE client_id = ?", ("c1",))
            row = c.fetchone()
            assert row is not None
            assert row["model_type"] == "client"


class TestEventLogs:
    def test_insert_log(self, db):
        with db.connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO event_logs (timestamp, event_type, message, group_id, details_json) VALUES (?, ?, ?, ?, ?)",
                (1712345678.0, "client_join", "Client c1 joined group", "grp1", json.dumps({"client_id": "c1"})),
            )
            conn.commit()

            c.execute(
                "SELECT * FROM event_logs WHERE group_id = ?", ("grp1",)
            )
            row = c.fetchone()
            assert row is not None
            assert row["event_type"] == "client_join"
