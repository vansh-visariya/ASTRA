"""
WebSocket and Socket.IO handlers for the Federated Learning server.

Contains the main WebSocket endpoint for client communication,
and Socket.IO event handlers for real-time updates.
"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from astra.infra.models import ClientUpdate
from astra.infra.security.auth import get_auth_manager


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    from astra.app.state import get_fl_server

    fl_server = get_fl_server()

    # Require JWT token on the WebSocket query string for authentication.
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    auth_manager = get_auth_manager()
    payload = auth_manager.verify_token(token)
    if not payload:
        await websocket.close(code=1008)
        return

    await fl_server.connection_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "register":
                client_id = message.get("client_id")
                group_id = message.get("group_id", "default")
                join_token = message.get("join_token")
                data_metadata = message.get("data_metadata", {})
                capabilities = message.get("capabilities", {})

                # Validate group and token
                group = fl_server.group_manager.groups.get(group_id)
                if not group:
                    await websocket.send_json({"status": "rejected", "reason": "group_not_found"})
                else:
                    # Check if client is already registered in the group (activated via dashboard)
                    already_registered = client_id in group.clients

                    # Check if client has approved join request (activated via REST API)
                    has_approved_join = False
                    if not already_registered and not join_token and payload:
                        try:
                            from astra.app.integration import get_platform_integration

                            platform = get_platform_integration()
                            user_id = payload.get("user_id")
                            if isinstance(user_id, int):
                                join_status = platform.get_user_join_status(user_id, group_id)
                                status_val = join_status.get("status") if join_status else None
                                if status_val in ("approved", "joined"):
                                    has_approved_join = True
                        except Exception:
                            pass

                    token_valid = join_token and join_token == group.join_token

                    if not already_registered and not has_approved_join and not token_valid:
                        await websocket.send_json({"status": "rejected", "reason": "invalid_token"})
                    else:
                        # Register client - be more lenient
                        try:
                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"[REGISTER] Registering client {client_id} to group {group_id}"
                            )
                            success = fl_server.group_manager.register_client(
                                client_id=client_id,
                                group_id=group_id,
                                client_info={
                                    "has_gpu": capabilities.get("has_gpu", False),
                                    "device": capabilities.get("device", "cpu"),
                                    "data_metadata": data_metadata,
                                    "connection": "websocket",
                                },
                            )
                            if success:
                                group = fl_server.group_manager.groups[group_id]
                                logger.info(
                                    f"[REGISTER] Client {client_id}"
                                    f" registered. Group now has"
                                    f" {len(group.clients)} clients:"
                                    f" {list(group.clients.keys())}"
                                )
                                # Register websocket for sending messages to client
                                fl_server.connection_manager.register_client(client_id, websocket)
                                await websocket.send_json(
                                    {
                                        "status": "registered",
                                        "client_id": client_id,
                                        "group_id": group_id,
                                        "model_id": group.model_id,
                                    }
                                )
                            else:
                                await websocket.send_json(
                                    {"status": "rejected", "reason": "registration_failed"}
                                )
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.error(f"Registration error: {e}")
                            await websocket.send_json(
                                {"status": "rejected", "reason": f"registration_error: {str(e)}"}
                            )

            elif message.get("type") == "update":
                # Reject — clients now submit deltas via REST POST /api/clients/{id}/delta.
                await websocket.send_json(
                    {
                        "status": "rejected",
                        "reason": "updates_via_rest",
                        "detail": "POST /api/clients/{client_id}/delta",
                    }
                )

            elif message.get("type") == "metrics":
                # Reject — client-side training metrics are out of scope.
                await websocket.send_json(
                    {"status": "rejected", "reason": "metrics_no_longer_supported"}
                )

            elif message.get("type") in (
                "train_command",
                "training_started",
                "training_paused",
                "training_stopped",
            ):
                await websocket.send_json(
                    {"status": "rejected", "reason": "client_training_no_longer_supported"}
                )

            elif message.get("type") == "chat_message":
                group_id = message.get("group_id")
                content = message.get("content", "").strip()
                if not group_id or not content:
                    await websocket.send_json({"status": "rejected", "reason": "missing_group_id_or_content"})
                    continue

                group = fl_server.group_manager.groups.get(group_id)
                if not group:
                    await websocket.send_json({"status": "rejected", "reason": "group_not_found"})
                    continue

                user_id = payload.get("user_id")
                username = payload.get("username", "Unknown")

                from astra.app.database import get_db
                db = get_db()
                with db.connection() as conn:
                    cursor = conn.execute(
                        "INSERT INTO secure_messages (group_id, sender_id, content) VALUES (?, ?, ?)",
                        (group_id, user_id, content),
                    )
                    message_id = cursor.lastrowid
                    conn.commit()

                await fl_server.group_manager.broadcast_to_group(group_id, {
                    "type": "new_message",
                    "group_id": group_id,
                    "message_id": message_id,
                    "sender_id": user_id,
                    "sender_name": username,
                    "sender_role": payload.get("role", "client"),
                    "content": content,
                })

                await websocket.send_json({
                    "status": "sent",
                    "message_id": message_id,
                    "group_id": group_id,
                })

    except WebSocketDisconnect:
        logger = logging.getLogger(__name__)
        logger.info("WebSocket disconnected normally")
        fl_server.connection_manager.disconnect(websocket)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"WebSocket error: {e}", exc_info=True)
        fl_server.connection_manager.disconnect(websocket)
