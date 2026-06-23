"""
Chat messages REST endpoints.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from astra.app.database import get_db
from astra.app.routes._auth import verify_request_jwt
from astra.app.state import get_fl_server

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_any_user(authorization: str = Header(None)):
    """Require valid JWT token (any role)."""
    return verify_request_jwt(authorization)


@router.get("/api/groups/{group_id}/messages")
async def get_messages(
    group_id: str,
    limit: int = 50,
    before: int = None,
    current_user=Depends(_get_any_user),
):
    """Get chat messages for a group."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    db = get_db()
    with db.connection() as conn:
        if before:
            rows = conn.execute(
                "SELECT m.*, u.username, u.full_name, u.role FROM secure_messages m "
                "LEFT JOIN users u ON m.sender_id = u.id "
                "WHERE m.group_id = ? AND m.id < ? "
                "ORDER BY m.created_at DESC LIMIT ?",
                (group_id, before, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT m.*, u.username, u.full_name, u.role FROM secure_messages m "
                "LEFT JOIN users u ON m.sender_id = u.id "
                "WHERE m.group_id = ? "
                "ORDER BY m.created_at DESC LIMIT ?",
                (group_id, limit),
            ).fetchall()
    messages = []
    for r in reversed(rows):
        messages.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "sender_id": r["sender_id"],
            "sender_name": r["full_name"] or r["username"] or f"User {r['sender_id']}",
            "sender_role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        })
    return {"messages": messages, "count": len(messages)}


@router.post("/api/groups/{group_id}/messages")
async def send_message(
    group_id: str,
    body: dict,
    current_user=Depends(_get_any_user),
):
    """Send a chat message in a group."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    db = get_db()
    with db.connection() as conn:
        cursor = conn.execute(
            "INSERT INTO secure_messages (group_id, sender_id, content) VALUES (?, ?, ?)",
            (group_id, current_user["user_id"], content),
        )
        message_id = cursor.lastrowid
        conn.commit()

    try:
        if fl_server and fl_server.group_manager:
            import asyncio
            asyncio.create_task(fl_server.group_manager.broadcast_to_group(group_id, {
                "type": "new_message",
                "group_id": group_id,
                "message_id": message_id,
                "sender_id": current_user["user_id"],
                "sender_name": current_user.get("full_name") or current_user.get("username", "Unknown"),
                "sender_role": current_user.get("role", "client"),
                "content": content,
            }))
    except Exception as e:
        logger.debug("Could not broadcast message: %s", e)

    return {
        "status": "sent",
        "message_id": message_id,
        "group_id": group_id,
    }


@router.get("/api/groups/{group_id}/unread-count")
async def get_unread_count(group_id: str, current_user=Depends(_get_any_user)):
    """Get count of messages for a group."""
    db = get_db()
    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM secure_messages WHERE group_id = ?",
            (group_id,),
        ).fetchone()
    count = row["cnt"] if row else 0
    return {"unread_count": count}
