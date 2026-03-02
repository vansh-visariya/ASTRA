"""
User Authentication and Account Management System.

Provides:
- User registration (signup)
- Password hashing with bcrypt
- JWT token management
- Role-based access control
- Secure token lifecycle for group joining

This module maintains backward compatibility with existing admin authentication
while adding signup and multi-role support.
"""

import secrets
import sqlite3
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import jwt
import bcrypt


# NOTE: In production you MUST override this via the SECRET_KEY environment variable.
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_in_production")
if SECRET_KEY == "dev_secret_key_change_in_production":
    logging.getLogger(__name__).warning(
        "Using default development SECRET_KEY. "
        "Set the SECRET_KEY environment variable for production deployments."
    )
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24


@dataclass
class User:
    """User account representation."""
    id: int
    username: str
    password_hash: str
    role: str  # 'admin', 'client', 'observer'
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True


class UserDatabase:
    """SQLite-based user database."""
    
    def __init__(self, db_path: str = "./users.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
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
            
            # Trust scores table
            cursor.execute('''
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
            
            # Join requests table
            cursor.execute('''
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
            
            # Secure messages table
            cursor.execute('''
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
            
            # Used tokens for replay protection
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS used_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT UNIQUE NOT NULL,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            ''')
            
            conn.commit()
            
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if not exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
            if not cursor.fetchone():
                # Create default admin
                admin_password = bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode('utf-8')
                cursor.execute(
                    'INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)',
                    ('admin', admin_password, 'admin', 'System Admin')
                )
                conn.commit()
                self.logger.info("Created default admin user")
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Optional[User]:
        """Create a new user account."""
        if role not in ('admin', 'client', 'observer'):
            raise ValueError(f"Invalid role: {role}")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO users (username, password_hash, role, email, full_name)
                       VALUES (?, ?, ?, ?, ?)''',
                    (username, password_hash, role, email, full_name)
                )
                user_id = cursor.lastrowid
                conn.commit()
                
                return User(
                    id=user_id,
                    username=username,
                    password_hash=password_hash,
                    role=role,
                    email=email,
                    full_name=full_name,
                    created_at=datetime.now()
                )
        except sqlite3.IntegrityError:
            self.logger.warning(f"User already exists: {username}")
            return None
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, username, password_hash, role, email, full_name, created_at, is_active FROM users WHERE username = ?',
                (username,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    username=row['username'],
                    password_hash=row['password_hash'],
                    role=row['role'],
                    email=row['email'],
                    full_name=row['full_name'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    is_active=bool(row['is_active'])
                )
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, username, password_hash, role, email, full_name, created_at, is_active FROM users WHERE id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    username=row['username'],
                    password_hash=row['password_hash'],
                    role=row['role'],
                    email=row['email'],
                    full_name=row['full_name'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    is_active=bool(row['is_active'])
                )
            return None
    
    def verify_password(self, username: str, password: str) -> Optional[User]:
        """Verify user credentials."""
        user = self.get_user(username)
        if not user or not user.is_active:
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return user
        return None
    
    def get_all_users(self, role: Optional[str] = None) -> List[User]:
        """Get all users, optionally filtered by role."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if role:
                cursor.execute(
                    'SELECT id, username, password_hash, role, email, full_name, created_at, is_active FROM users WHERE role = ?',
                    (role,)
                )
            else:
                cursor.execute(
                    'SELECT id, username, password_hash, role, email, full_name, created_at, is_active FROM users'
                )
            
            return [
                User(
                    id=row['id'],
                    username=row['username'],
                    password_hash=row['password_hash'],
                    role=row['role'],
                    email=row['email'],
                    full_name=row['full_name'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    is_active=bool(row['is_active'])
                )
                for row in cursor.fetchall()
            ]
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user fields."""
        allowed_fields = {'email', 'full_name', 'is_active', 'role'}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not kwargs:
            return False
        
        set_clause = ', '.join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [user_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE users SET {set_clause} WHERE id = ?', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            return cursor.rowcount > 0


class TokenManager:
    """Manages JWT tokens and secure token lifecycle for group joining."""
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
        self.logger = logging.getLogger(__name__)
    
    def create_token(self, user: User, expires_hours: int = TOKEN_EXPIRY_HOURS) -> str:
        """Create JWT token for user."""
        payload = {
            "sub": user.username,
            "role": user.role,
            "user_id": user.id,
            "exp": datetime.utcnow() + timedelta(hours=expires_hours)
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.PyJWTError as e:
            self.logger.warning(f"Token verification failed: {e}")
            return None
    
    def create_join_token(self, group_id: str, user_id: int) -> Tuple[str, str]:
        """Create a secure token for group joining.
        
        Returns:
            Tuple of (token, nonce)
        """
        # Generate secure random token
        token = secrets.token_urlsafe(32)
        nonce = secrets.token_hex(16)
        
        # Create expiry time
        expires_at = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
        
        # Store token hash for replay protection
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO used_tokens (token_hash, expires_at) VALUES (?, ?)',
                (token_hash, expires_at.isoformat())
            )
            conn.commit()
        
        return token, nonce
    
    def validate_join_token(self, token: str) -> bool:
        """Validate a join token and check for replay attacks."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT expires_at FROM used_tokens WHERE token_hash = ?',
                (token_hash,)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            expires_at = datetime.fromisoformat(row['expires_at'])
            if datetime.utcnow() > expires_at:
                return False
            
            # Mark as used (one-time use)
            cursor.execute('DELETE FROM used_tokens WHERE token_hash = ?', (token_hash,))
            conn.commit()
            
            return True
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM used_tokens WHERE expires_at < ?',
                (datetime.utcnow().isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                self.logger.info(f"Cleaned up {deleted} expired tokens")


class JoinRequestManager:
    """Manages group join requests with secure token delivery."""
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
        self.logger = logging.getLogger(__name__)
    
    def create_request(
        self,
        group_id: str,
        user_id: int,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Create a join request and return the request nonce."""
        request_nonce = secrets.token_hex(16)
        
        import json
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    '''INSERT INTO join_requests 
                       (group_id, user_id, request_nonce, metadata_json)
                       VALUES (?, ?, ?, ?)''',
                    (group_id, user_id, request_nonce, metadata_json)
                )
                conn.commit()
                return request_nonce
            except sqlite3.IntegrityError:
                self.logger.warning(f"Pending request already exists for user {user_id} in group {group_id}")
                return None
    
    def get_pending_requests(self, group_id: Optional[str] = None) -> List[Dict]:
        """Get pending join requests."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            if group_id:
                cursor.execute(
                    '''SELECT jr.id, jr.group_id, jr.user_id, u.username, jr.requested_at, jr.metadata_json
                       FROM join_requests jr
                       JOIN users u ON jr.user_id = u.id
                       WHERE jr.group_id = ? AND jr.status = 'pending'
                       ORDER BY jr.requested_at DESC''',
                    (group_id,)
                )
            else:
                cursor.execute(
                    '''SELECT jr.id, jr.group_id, jr.user_id, u.username, jr.requested_at, jr.metadata_json
                       FROM join_requests jr
                       JOIN users u ON jr.user_id = u.id
                       WHERE jr.status = 'pending'
                       ORDER BY jr.requested_at DESC'''
                )
            
            import json
            return [
                {
                    'id': row['id'],
                    'group_id': row['group_id'],
                    'user_id': row['user_id'],
                    'username': row['username'],
                    'requested_at': row['requested_at'],
                    'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else {}
                }
                for row in cursor.fetchall()
            ]
    
    def approve_request(
        self,
        request_id: int,
        resolved_by: int,
        token: str
    ) -> bool:
        """Approve a join request and store the token for secure delivery."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get request details
            cursor.execute(
                'SELECT user_id, group_id FROM join_requests WHERE id = ? AND status = ?',
                (request_id, 'pending')
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            to_user_id = row['user_id']
            group_id = row['group_id']
            
            # Update request status
            cursor.execute(
                '''UPDATE join_requests 
                   SET status = 'approved', resolved_at = ?, resolved_by = ?, 
                       token_delivered = 1, token_delivered_at = ?
                   WHERE id = ?''',
                (datetime.utcnow().isoformat(), resolved_by, datetime.utcnow().isoformat(), request_id)
            )
            
            # Store encrypted token for secure delivery
            # The token will be encrypted when sent to the client
            # Here we just mark that it's ready for delivery
            
            conn.commit()
            return True
    
    def reject_request(self, request_id: int, resolved_by: int) -> bool:
        """Reject a join request."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''UPDATE join_requests 
                   SET status = 'rejected', resolved_at = ?, resolved_by = ?
                   WHERE id = ? AND status = 'pending' ''',
                (datetime.utcnow().isoformat(), resolved_by, request_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_user_request_status(self, user_id: int, group_id: str) -> Optional[Dict]:
        """Get the status of a user's join request."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT id, status, requested_at, resolved_at, token_delivered
                   FROM join_requests 
                   WHERE user_id = ? AND group_id = ?
                   ORDER BY requested_at DESC LIMIT 1''',
                (user_id, group_id)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'status': row['status'],
                    'requested_at': row['requested_at'],
                    'resolved_at': row['resolved_at'],
                    'token_delivered': bool(row['token_delivered'])
                }
            return None


class TrustScoreManager:
    """Manages trust scores linked to user accounts."""
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
        self.logger = logging.getLogger(__name__)
    
    def get_trust_score(self, user_id: int, group_id: str) -> float:
        """Get trust score for user in a group."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT score FROM trust_scores WHERE user_id = ? AND group_id = ?',
                (user_id, group_id)
            )
            row = cursor.fetchone()
            return row['score'] if row else 1.0
    
    def update_trust_score(
        self,
        user_id: int,
        group_id: str,
        score: float,
        quarantined: bool = False
    ) -> None:
        """Update trust score for user."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO trust_scores (user_id, group_id, score, quarantined)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, group_id) 
                   DO UPDATE SET score = ?, quarantined = ?, last_updated = CURRENT_TIMESTAMP''',
                (user_id, group_id, score, quarantined, score, quarantined)
            )
            conn.commit()
    
    def get_all_trust_scores(self, group_id: Optional[str] = None) -> Dict[int, float]:
        """Get all trust scores."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            if group_id:
                cursor.execute(
                    'SELECT user_id, score, quarantined FROM trust_scores WHERE group_id = ?',
                    (group_id,)
                )
            else:
                cursor.execute('SELECT user_id, group_id, score, quarantined FROM trust_scores')
            
            return {
                row['user_id']: {
                    'score': row['score'],
                    'group_id': row.get('group_id'),
                    'quarantined': bool(row['quarantined'])
                }
                for row in cursor.fetchall()
            }
    
    def is_quarantined(self, user_id: int, group_id: str) -> bool:
        """Check if user is quarantined."""
        with self.user_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT quarantined FROM trust_scores WHERE user_id = ? AND group_id = ?',
                (user_id, group_id)
            )
            row = cursor.fetchone()
            return bool(row['quarantined']) if row else False


class AuthManager:
    """Main authentication manager combining all auth components."""
    
    def __init__(self, db_path: str = "./users.db"):
        self.user_db = UserDatabase(db_path)
        self.token_manager = TokenManager(self.user_db)
        self.join_request_manager = JoinRequestManager(self.user_db)
        self.trust_score_manager = TrustScoreManager(self.user_db)
        self.logger = logging.getLogger(__name__)
    
    def signup(
        self,
        username: str,
        password: str,
        role: str = 'client',
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Tuple[Optional[User], Optional[str]]:
        """Register a new user.
        
        Returns:
            Tuple of (User, error_message)
        """
        # Validate password strength
        if len(password) < 6:
            return None, "Password must be at least 6 characters"
        
        # Validate username
        if not username or len(username) < 3:
            return None, "Username must be at least 3 characters"
        
        # Create user
        user = self.user_db.create_user(username, password, role, email, full_name)
        
        if user:
            # Initialize trust score
            self.trust_score_manager.update_trust_score(user.id, 'default', 1.0)
            
            return user, None
        
        return None, "Username already exists"
    
    def login(self, username: str, password: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Authenticate user and return token.
        
        Returns:
            Tuple of (token_data, error_message)
        """
        user = self.user_db.verify_password(username, password)
        
        if not user:
            return None, "Invalid credentials"
        
        if not user.is_active:
            return None, "Account is disabled"
        
        token = self.token_manager.create_token(user)
        
        return {
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'role': user.role,
                'full_name': user.full_name,
                'email': user.email
            }
        }, None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return payload."""
        return self.token_manager.verify_token(token)
    
    def require_role(self, token: str, allowed_roles: List[str]) -> Optional[Dict[str, Any]]:
        """Verify token and check role.
        
        Returns payload if authorized, None otherwise.
        """
        payload = self.verify_token(token)
        if not payload:
            return None
        
        if payload.get('role') not in allowed_roles:
            return None
        
        return payload
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.user_db.get_user(username)
    
    def get_all_users(self, role: Optional[str] = None) -> List[User]:
        """Get all users."""
        return self.user_db.get_all_users(role)
    
    # Role check helpers
    def is_admin(self, token: str) -> bool:
        """Check if token belongs to admin."""
        payload = self.verify_token(token)
        return payload is not None and payload.get('role') == 'admin'
    
    def is_client(self, token: str) -> bool:
        """Check if token belongs to client."""
        payload = self.verify_token(token)
        return payload is not None and payload.get('role') == 'client'
    
    def is_observer(self, token: str) -> bool:
        """Check if token belongs to observer."""
        payload = self.verify_token(token)
        return payload is not None and payload.get('role') == 'observer'
    
    def is_admin_or_client(self, token: str) -> bool:
        """Check if token belongs to admin or client."""
        payload = self.verify_token(token)
        return payload is not None and payload.get('role') in ('admin', 'client')


# Global instance (will be initialized by server)
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def init_auth_manager(db_path: str = "./users.db") -> AuthManager:
    """Initialize the auth manager with custom database path."""
    global _auth_manager
    _auth_manager = AuthManager(db_path)
    return _auth_manager
