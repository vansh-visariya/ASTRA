"""
Announcements REST endpoints.
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


def _require_admin(authorization: str = Header(None)):
    """Require admin role."""
    user = verify_request_jwt(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.get("/api/groups/{group_id}/announcements")
async def get_announcements(group_id: str, current_user=Depends(_get_any_user)):
    """Get announcements for a group."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    db = get_db()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT a.*, u.username, u.full_name FROM announcements a "
            "LEFT JOIN users u ON a.author_id = u.id "
            "WHERE a.group_id = ? ORDER BY a.created_at DESC LIMIT 50",
            (group_id,),
        ).fetchall()
    announcements = []
    for r in rows:
        announcements.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "author_id": r["author_id"],
            "author_name": r["full_name"] or r["username"] or f"User {r['author_id']}",
            "message": r["message"],
            "priority": r["priority"],
            "created_at": r["created_at"],
        })
    return {"announcements": announcements, "count": len(announcements)}


@router.post("/api/groups/{group_id}/announcements")
async def send_announcement(
    group_id: str,
    body: dict,
    current_user=Depends(_require_admin),
):
    """Send an announcement to all clients in a group (admin only)."""
    fl_server = get_fl_server()
    group = fl_server.group_manager.groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    priority = body.get("priority", "info")
    if priority not in ("info", "warning", "error"):
        priority = "info"

    db = get_db()
    with db.connection() as conn:
        cursor = conn.execute(
            "INSERT INTO announcements (group_id, author_id, message, priority) VALUES (?, ?, ?, ?)",
            (group_id, current_user["user_id"], message, priority),
        )
        announcement_id = cursor.lastrowid
        conn.commit()

    try:
        if fl_server and fl_server.group_manager:
            import asyncio
            asyncio.create_task(fl_server.group_manager.broadcast_to_group(group_id, {
                "type": "announcement",
                "group_id": group_id,
                "announcement_id": announcement_id,
                "author": current_user.get("full_name") or current_user.get("username", "Admin"),
                "message": message,
                "priority": priority,
            }))
    except Exception as e:
        logger.debug("Could not broadcast announcement: %s", e)

    return {
        "status": "sent",
        "announcement_id": announcement_id,
        "group_id": group_id,
    }
