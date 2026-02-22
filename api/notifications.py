"""
Notification System for Federated Learning Platform.

Provides:
- Real-time notifications via WebSocket
- Persistent notification storage
- Event-driven notification triggers
- Multiple notification channels (in-memory, database)

This is a modular system designed for future extension.
"""

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


class NotificationType(str, Enum):
    """Notification event types."""
    # Group events
    GROUP_CREATED = "group_created"
    GROUP_STARTED = "group_started"
    GROUP_PAUSED = "group_paused"
    GROUP_STOPPED = "group_stopped"
    GROUP_COMPLETED = "group_completed"
    
    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_ROUND_COMPLETE = "training_round_complete"
    MODEL_UPDATED = "model_updated"
    AGGREGATION_COMPLETE = "aggregation_complete"
    
    # Join events
    JOIN_REQUEST = "join_request"
    JOIN_APPROVED = "join_approved"
    JOIN_REJECTED = "join_rejected"
    TOKEN_RECEIVED = "token_received"
    
    # Client events
    CLIENT_JOINED = "client_joined"
    CLIENT_LEFT = "client_left"
    CLIENT_UPDATE_RECEIVED = "client_update_received"
    
    # Error events
    TRAINING_FAILED = "training_failed"
    CONNECTION_LOST = "connection_lost"
    AGGREGATION_FAILED = "aggregation_failed"
    AUTH_FAILED = "auth_failed"
    
    # General
    GENERAL = "general"
    SYSTEM = "system"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


NOTIFICATION_MAPPING = {
    NotificationType.GROUP_CREATED: NotificationPriority.INFO,
    NotificationType.GROUP_STARTED: NotificationPriority.INFO,
    NotificationType.GROUP_PAUSED: NotificationPriority.WARNING,
    NotificationType.GROUP_STOPPED: NotificationPriority.INFO,
    NotificationType.GROUP_COMPLETED: NotificationPriority.SUCCESS,
    
    NotificationType.TRAINING_STARTED: NotificationPriority.INFO,
    NotificationType.TRAINING_ROUND_COMPLETE: NotificationPriority.INFO,
    NotificationType.MODEL_UPDATED: NotificationPriority.INFO,
    NotificationType.AGGREGATION_COMPLETE: NotificationPriority.INFO,
    
    NotificationType.JOIN_REQUEST: NotificationPriority.INFO,
    NotificationType.JOIN_APPROVED: NotificationPriority.SUCCESS,
    NotificationType.JOIN_REJECTED: NotificationPriority.WARNING,
    NotificationType.TOKEN_RECEIVED: NotificationPriority.SUCCESS,
    
    NotificationType.CLIENT_JOINED: NotificationPriority.INFO,
    NotificationType.CLIENT_LEFT: NotificationPriority.WARNING,
    NotificationType.CLIENT_UPDATE_RECEIVED: NotificationPriority.DEBUG,
    
    NotificationType.TRAINING_FAILED: NotificationPriority.ERROR,
    NotificationType.CONNECTION_LOST: NotificationPriority.WARNING,
    NotificationType.AGGREGATION_FAILED: NotificationPriority.ERROR,
    NotificationType.AUTH_FAILED: NotificationPriority.ERROR,
    
    NotificationType.GENERAL: NotificationPriority.INFO,
    NotificationType.SYSTEM: NotificationPriority.INFO,
}


@dataclass
class Notification:
    """Notification representation."""
    id: Optional[int] = None
    notification_type: NotificationType = NotificationType.GENERAL
    priority: NotificationPriority = NotificationPriority.INFO
    title: str = ""
    message: str = ""
    user_id: Optional[int] = None  # None = broadcast
    group_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    read: bool = False


class NotificationDatabase:
    """Persistent notification storage."""
    
    def __init__(self, db_path: str = "./notifications.db"):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize notification tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
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
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_notifications_user 
                ON notifications(user_id, read, created_at)
            ''')
            
            conn.commit()
    
    def create(self, notification: Notification) -> int:
        """Create a new notification."""
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO notifications 
                   (notification_type, priority, title, message, user_id, group_id, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    notification.notification_type.value,
                    notification.priority.value,
                    notification.title,
                    notification.message,
                    notification.user_id,
                    notification.group_id,
                    json.dumps(notification.data)
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_for_user(
        self,
        user_id: Optional[int],
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get notifications for a user."""
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id is None:
                # Broadcast notifications
                query = 'SELECT * FROM notifications WHERE user_id IS NULL'
                params = []
            else:
                query = 'SELECT * FROM notifications WHERE user_id = ? OR user_id IS NULL'
                params = [user_id]
            
            if unread_only:
                query += ' AND read = 0'
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            return [
                Notification(
                    id=row['id'],
                    notification_type=NotificationType(row['notification_type']),
                    priority=NotificationPriority(row['priority']),
                    title=row['title'],
                    message=row['message'] or '',
                    user_id=row['user_id'],
                    group_id=row['group_id'],
                    data=json.loads(row['data_json']) if row['data_json'] else {},
                    created_at=datetime.fromisoformat(row['created_at']),
                    read_at=datetime.fromisoformat(row['read_at']) if row['read_at'] else None,
                    read=bool(row['read'])
                )
                for row in cursor.fetchall()
            ]
    
    def mark_read(self, notification_id: int) -> bool:
        """Mark notification as read."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''UPDATE notifications 
                   SET read = 1, read_at = ? 
                   WHERE id = ?''',
                (datetime.now().isoformat(), notification_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def mark_all_read(self, user_id: int) -> int:
        """Mark all notifications as read for user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''UPDATE notifications 
                   SET read = 1, read_at = ? 
                   WHERE (user_id = ? OR user_id IS NULL) AND read = 0''',
                (datetime.now().isoformat(), user_id)
            )
            conn.commit()
            return cursor.rowcount
    
    def get_unread_count(self, user_id: int) -> int:
        """Get count of unread notifications."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT COUNT(*) FROM notifications 
                   WHERE (user_id = ? OR user_id IS NULL) AND read = 0''',
                (user_id,)
            )
            return cursor.fetchone()[0]
    
    def delete_old(self, days: int = 30) -> int:
        """Delete notifications older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''DELETE FROM notifications 
                   WHERE created_at < datetime('now', '-' || ? || ' days')''',
                (days,)
            )
            conn.commit()
            return cursor.rowcount


class WebSocketNotifier:
    """Real-time WebSocket notification delivery."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._connections: Dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_connection(self, user_id: int, queue: asyncio.Queue):
        """Register a WebSocket connection for a user."""
        self._connections[user_id] = queue
    
    def unregister_connection(self, user_id: int):
        """Unregister a WebSocket connection."""
        if user_id in self._connections:
            del self._connections[user_id]
    
    async def send_to_user(self, user_id: int, notification: Notification):
        """Send notification to a specific user."""
        if user_id in self._connections:
            await self._connections[user_id].put(notification)
    
    async def broadcast(self, notification: Notification):
        """Broadcast notification to all connected users."""
        for user_id in self._connections:
            await self._send_to_user(user_id, notification)
    
    async def _send_to_user(self, user_id: int, notification: Notification):
        """Internal method to send to user."""
        if user_id in self._connections:
            await self._connections[user_id].put({
                'type': 'notification',
                'notification_type': notification.notification_type.value,
                'priority': notification.priority.value,
                'title': notification.title,
                'message': notification.message,
                'group_id': notification.group_id,
                'data': notification.data,
                'created_at': notification.created_at.isoformat()
            })
    
    def start(self):
        """Start the broadcast task."""
        if not self._running:
            self._running = True
    
    def stop(self):
        """Stop the broadcast task."""
        self._running = False
        if self._task:
            self._task.cancel()


class NotificationService:
    """Main notification service combining all notification channels."""
    
    def __init__(self, db_path: str = "./notifications.db"):
        self.db = NotificationDatabase(db_path)
        self.ws_notifier = WebSocketNotifier()
        self.logger = logging.getLogger(__name__)
        
        # Event callbacks for external integrations
        self._event_handlers: Dict[NotificationType, List[Callable]] = defaultdict(list)
    
    def register_handler(self, notification_type: NotificationType, handler: Callable):
        """Register an event handler for a notification type."""
        self._event_handlers[notification_type].append(handler)
    
    def notify(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        user_id: Optional[int] = None,
        group_id: Optional[str] = None,
        data: Optional[Dict] = None,
        priority: Optional[NotificationPriority] = None,
        send_websocket: bool = True,
        persist: bool = True
    ) -> Notification:
        """Create and send a notification."""
        
        # Determine priority
        if priority is None:
            priority = NOTIFICATION_MAPPING.get(notification_type, NotificationPriority.INFO)
        
        notification = Notification(
            notification_type=notification_type,
            priority=priority,
            title=title,
            message=message,
            user_id=user_id,
            group_id=group_id,
            data=data or {}
        )
        
        # Persist to database
        if persist:
            notification.id = self.db.create(notification)
        
        # Send via WebSocket
        if send_websocket:
            if user_id:
                asyncio.create_task(self.ws_notifier.send_to_user(user_id, notification))
            else:
                asyncio.create_task(self.ws_notifier.broadcast(notification))
        
        # Trigger event handlers
        for handler in self._event_handlers[notification_type]:
            try:
                handler(notification)
            except Exception as e:
                self.logger.error(f"Event handler error: {e}")
        
        return notification
    
    # Convenience methods for common events
    
    def notify_group_created(self, group_id: str, admin_user_id: int):
        """Notify about group creation."""
        return self.notify(
            NotificationType.GROUP_CREATED,
            title="Group Created",
            message=f"Group '{group_id}' has been created",
            user_id=admin_user_id,
            group_id=group_id
        )
    
    def notify_group_started(self, group_id: str):
        """Notify about training start."""
        return self.notify(
            NotificationType.GROUP_STARTED,
            title="Training Started",
            message=f"Training has started for group '{group_id}'",
            group_id=group_id
        )
    
    def notify_group_completed(self, group_id: str, final_accuracy: float):
        """Notify about training completion."""
        return self.notify(
            NotificationType.GROUP_COMPLETED,
            title="Training Completed",
            message=f"Training completed for group '{group_id}' with accuracy {final_accuracy:.2%}",
            group_id=group_id
        )
    
    def notify_join_request(self, group_id: str, admin_user_id: int, request_data: Dict):
        """Notify admin about new join request."""
        return self.notify(
            NotificationType.JOIN_REQUEST,
            title="New Join Request",
            message=f"User '{request_data.get('username')}' wants to join group '{group_id}'",
            user_id=admin_user_id,
            group_id=group_id,
            data=request_data
        )
    
    def notify_join_approved(self, user_id: int, group_id: str, token: str):
        """Notify user about approved join request."""
        return self.notify(
            NotificationType.JOIN_APPROVED,
            title="Join Request Approved",
            message=f"Your request to join group '{group_id}' has been approved",
            user_id=user_id,
            group_id=group_id,
            data={'token': token}  # This will be encrypted in real implementation
        )
    
    def notify_join_rejected(self, user_id: int, group_id: str, reason: str = None):
        """Notify user about rejected join request."""
        message = f"Your request to join group '{group_id}' has been rejected"
        if reason:
            message += f": {reason}"
        
        return self.notify(
            NotificationType.JOIN_REJECTED,
            title="Join Request Rejected",
            message=message,
            user_id=user_id,
            group_id=group_id
        )
    
    def notify_client_joined(self, group_id: str, client_username: str):
        """Notify about client joining."""
        return self.notify(
            NotificationType.CLIENT_JOINED,
            title="Client Joined",
            message=f"Client '{client_username}' joined group '{group_id}'",
            group_id=group_id,
            data={'username': client_username}
        )
    
    def notify_training_round(self, group_id: str, round_num: int, accuracy: float):
        """Notify about training round completion."""
        return self.notify(
            NotificationType.TRAINING_ROUND_COMPLETE,
            title="Round Complete",
            message=f"Round {round_num} completed with accuracy {accuracy:.2%}",
            group_id=group_id,
            data={'round': round_num, 'accuracy': accuracy}
        )
    
    def notify_error(self, title: str, message: str, user_id: Optional[int] = None, group_id: Optional[str] = None):
        """Notify about an error."""
        return self.notify(
            NotificationType.TRAINING_FAILED,
            title=title,
            message=message,
            user_id=user_id,
            group_id=group_id,
            priority=NotificationPriority.ERROR
        )
    
    # Query methods
    
    def get_notifications(
        self,
        user_id: int,
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get notifications for user."""
        return self.db.get_for_user(user_id, limit, unread_only)
    
    def mark_read(self, notification_id: int) -> bool:
        """Mark notification as read."""
        return self.db.mark_read(notification_id)
    
    def mark_all_read(self, user_id: int) -> int:
        """Mark all notifications as read."""
        return self.db.mark_all_read(user_id)
    
    def get_unread_count(self, user_id: int) -> int:
        """Get unread notification count."""
        return self.db.get_unread_count(user_id)


# Global instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


def init_notification_service(db_path: str = "./notifications.db") -> NotificationService:
    """Initialize the notification service."""
    global _notification_service
    _notification_service = NotificationService(db_path)
    return _notification_service
