"""
Unit tests for auth module: UserDatabase, TokenManager, JoinRequestManager,
TrustScoreManager, AuthManager.
"""

from unittest.mock import MagicMock, patch
import tempfile
import os

import pytest

from astra.infra.security.auth import (
    User,
    UserDatabase,
    TokenManager,
    JoinRequestManager,
    TrustScoreManager,
    AuthManager,
    SECRET_KEY,
    init_auth_manager,
    get_auth_manager,
)


@pytest.fixture
def in_memory_db():
    from astra.app.database import AstraDB
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = AstraDB(db_path=path)
    yield db
    os.unlink(path)


@pytest.fixture
def user_db(in_memory_db):
    return UserDatabase(db=in_memory_db)


@pytest.fixture
def token_manager(user_db):
    return TokenManager(user_db)


@pytest.fixture
def join_manager(user_db):
    return JoinRequestManager(user_db)


@pytest.fixture
def trust_manager(user_db):
    return TrustScoreManager(user_db)


@pytest.fixture
def auth_manager(in_memory_db):
    return AuthManager(db=in_memory_db)


class TestUserDatabase:
    def test_create_user(self, user_db):
        user = user_db.create_user("alice", "password123", "client")
        assert user is not None
        assert user.username == "alice"
        assert user.role == "client"

    def test_duplicate_user_returns_none(self, user_db):
        user_db.create_user("bob", "pass", "client")
        result = user_db.create_user("bob", "pass2", "admin")
        assert result is None

    def test_invalid_role_raises(self, user_db):
        with pytest.raises(ValueError):
            user_db.create_user("eve", "pass", "invalid_role")

    def test_get_user(self, user_db):
        user_db.create_user("charlie", "secret", "client")
        user = user_db.get_user("charlie")
        assert user is not None
        assert user.username == "charlie"

    def test_get_nonexistent_user(self, user_db):
        assert user_db.get_user("nobody") is None

    def test_get_user_by_id(self, user_db):
        created = user_db.create_user("dave", "pass", "client")
        found = user_db.get_user_by_id(created.id)
        assert found is not None
        assert found.username == "dave"

    def test_verify_password_correct(self, user_db):
        user_db.create_user("eve", "correct_pass", "client")
        result = user_db.verify_password("eve", "correct_pass")
        assert result is not None
        assert result.username == "eve"

    def test_verify_password_wrong(self, user_db):
        user_db.create_user("frank", "real_pass", "client")
        result = user_db.verify_password("frank", "wrong_pass")
        assert result is None

    def test_get_all_users(self, user_db):
        user_db.create_user("u1", "p1", "client")
        user_db.create_user("u2", "p2", "admin")
        users = user_db.get_all_users()
        assert len(users) >= 2

    def test_filter_by_role(self, user_db):
        user_db.create_user("admin1", "p", "admin")
        user_db.create_user("client1", "p", "client")
        admins = user_db.get_all_users(role="admin")
        assert all(u.role == "admin" for u in admins)

    def test_update_user(self, user_db):
        created = user_db.create_user("update_me", "p", "client")
        success = user_db.update_user(created.id, email="new@email.com", full_name="New Name")
        assert success is True
        updated = user_db.get_user_by_id(created.id)
        assert updated.email == "new@email.com"
        assert updated.full_name == "New Name"

    def test_delete_user(self, user_db):
        created = user_db.create_user("delete_me", "p", "client")
        assert user_db.get_user("delete_me") is not None
        success = user_db.delete_user(created.id)
        assert success is True
        assert user_db.get_user("delete_me") is None


class TestTokenManager:
    def test_create_and_verify_token(self, token_manager, user_db):
        user = user_db.create_user("token_test", "pass", "client")
        token = token_manager.create_token(user)
        assert token is not None

        payload = token_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "token_test"
        assert payload["role"] == "client"
        assert payload["user_id"] == user.id

    def test_invalid_token_returns_none(self, token_manager):
        assert token_manager.verify_token("bad_token") is None

    def test_create_join_token(self, token_manager, user_db):
        user = user_db.create_user("join_test", "pass", "client")
        token, nonce = token_manager.create_join_token("grp1", user.id)
        assert len(token) > 10
        assert len(nonce) == 32

    def test_validate_join_token(self, token_manager, user_db):
        user = user_db.create_user("join2", "pass", "client")
        token, _nonce = token_manager.create_join_token("grp1", user.id)
        assert token_manager.validate_join_token(token) is True

    def test_replay_attack_blocked(self, token_manager, user_db):
        user = user_db.create_user("join3", "pass", "client")
        token, _nonce = token_manager.create_join_token("grp1", user.id)
        assert token_manager.validate_join_token(token) is True
        assert token_manager.validate_join_token(token) is False

    def test_invalid_join_token(self, token_manager):
        assert token_manager.validate_join_token("invalid_token_here") is False


class TestJoinRequestManager:
    def test_create_request(self, join_manager, user_db):
        user = user_db.create_user("req_user", "pass", "client")
        nonce = join_manager.create_request("grp1", user.id)
        assert nonce is not None
        assert len(nonce) == 32

    def test_duplicate_request_blocked(self, join_manager, user_db):
        user = user_db.create_user("req_dup", "pass", "client")
        first = join_manager.create_request("grp1", user.id)
        assert first is not None
        second = join_manager.create_request("grp1", user.id)
        assert second is None

    def test_get_pending_requests(self, join_manager, user_db):
        user = user_db.create_user("req_pend", "pass", "client")
        join_manager.create_request("grp1", user.id)
        pending = join_manager.get_pending_requests(group_id="grp1")
        assert len(pending) >= 1

    def test_get_all_pending(self, join_manager, user_db):
        user_a = user_db.create_user("req_a", "pass", "client")
        user_b = user_db.create_user("req_b", "pass", "client")
        join_manager.create_request("grp_a", user_a.id)
        join_manager.create_request("grp_b", user_b.id)
        all_pending = join_manager.get_pending_requests()
        assert len(all_pending) >= 2

    def test_approve_request(self, join_manager, user_db, token_manager):
        user = user_db.create_user("req_approve", "pass", "client")
        admin = user_db.create_user("req_admin", "pass", "admin")
        join_manager.create_request("grp1", user.id)
        pending = join_manager.get_pending_requests(group_id="grp1")
        assert len(pending) >= 1

        token, _nonce = token_manager.create_join_token("grp1", user.id)
        success = join_manager.approve_request(pending[0]["id"], admin.id, token)
        assert success is True

    def test_reject_request(self, join_manager, user_db):
        user = user_db.create_user("req_reject", "pass", "client")
        admin = user_db.create_user("req_admin2", "pass", "admin")
        join_manager.create_request("grp1", user.id)
        pending = join_manager.get_pending_requests(group_id="grp1")

        success = join_manager.reject_request(pending[0]["id"], admin.id)
        assert success is True

    def test_user_request_status(self, join_manager, user_db):
        user = user_db.create_user("req_status", "pass", "client")
        join_manager.create_request("grp1", user.id)
        status = join_manager.get_user_request_status(user.id, "grp1")
        assert status is not None
        assert status["status"] == "pending"

    def test_no_status_for_nonexistent(self, join_manager):
        assert join_manager.get_user_request_status(99999, "no_group") is None


class TestTrustScoreManager:
    def test_default_score_is_one(self, trust_manager):
        score = trust_manager.get_trust_score(99999, "grp1")
        assert score == 1.0

    def test_update_score(self, trust_manager, user_db):
        user = user_db.create_user("trust_user", "pass", "client")
        trust_manager.update_trust_score(user.id, "grp1", 0.5)
        assert trust_manager.get_trust_score(user.id, "grp1") == 0.5

    def test_quarantine(self, trust_manager, user_db):
        user = user_db.create_user("quar_user", "pass", "client")
        trust_manager.update_trust_score(user.id, "grp1", 0.2, quarantined=True)
        assert trust_manager.is_quarantined(user.id, "grp1") is True


class TestAuthManager:
    def test_signup(self, auth_manager):
        user, error = auth_manager.signup("new_user", "password123")
        assert user is not None
        assert error is None
        assert user.username == "new_user"
        assert user.role == "client"

    def test_signup_duplicate(self, auth_manager):
        auth_manager.signup("dup_user", "password123")
        user, error = auth_manager.signup("dup_user", "password456")
        assert user is None
        assert error == "Username already exists"

    def test_signup_weak_password(self, auth_manager):
        _user, error = auth_manager.signup("weak", "123")
        assert error == "Password must be at least 6 characters"

    def test_signup_short_username(self, auth_manager):
        _user, error = auth_manager.signup("ab", "password123")
        assert error == "Username must be at least 3 characters"

    def test_login(self, auth_manager):
        auth_manager.signup("login_user", "password123")
        data, error = auth_manager.login("login_user", "password123")
        assert data is not None
        assert error is None
        assert "token" in data
        assert data["user"]["username"] == "login_user"

    def test_login_wrong_password(self, auth_manager):
        auth_manager.signup("login2", "password123")
        data, error = auth_manager.login("login2", "wrong")
        assert data is None
        assert error == "Invalid credentials"

    def test_verify_token(self, auth_manager):
        auth_manager.signup("verify_me", "password123")
        data, _ = auth_manager.login("verify_me", "password123")
        payload = auth_manager.verify_token(data["token"])
        assert payload is not None
        assert payload["sub"] == "verify_me"

    def test_require_role_allowed(self, auth_manager):
        auth_manager.signup("client_role", "password123")
        data, _ = auth_manager.login("client_role", "password123")
        payload = auth_manager.require_role(data["token"], ["client"])
        assert payload is not None

    def test_require_role_denied(self, auth_manager):
        auth_manager.signup("client_denied", "password123")
        data, _ = auth_manager.login("client_denied", "password123")
        payload = auth_manager.require_role(data["token"], ["admin"])
        assert payload is None

    def test_role_checks(self, auth_manager):
        auth_manager.signup("admin_user", "password123", role="admin")
        data, _ = auth_manager.login("admin_user", "password123")
        token = data["token"]
        assert auth_manager.is_admin(token) is True
        assert auth_manager.is_client(token) is False
        assert auth_manager.is_admin_or_client(token) is True


class TestGlobalAuthManager:
    def test_init_and_get(self, in_memory_db):
        mgr = init_auth_manager(db=in_memory_db)
        assert get_auth_manager() is mgr

    def test_get_creates_default(self):
        from astra.infra.security.auth import _auth_manager
        old = _auth_manager
        import astra.infra.security.auth as auth_mod
        auth_mod._auth_manager = None
        try:
            mgr = get_auth_manager()
            assert mgr is not None
            assert mgr.user_db is not None
        finally:
            auth_mod._auth_manager = old
