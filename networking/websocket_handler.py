"""
WebSocket and Socket.IO handlers for the Federated Learning server.

Contains the main WebSocket endpoint for client communication,
and Socket.IO event handlers for real-time updates.
"""

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from api.auth_system import get_auth_manager
from networking.models import ClientUpdate


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    from networking.state import get_fl_server
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

            if message.get('type') == 'register':
                client_id = message.get('client_id')
                group_id = message.get('group_id', 'default')
                join_token = message.get('join_token')
                data_metadata = message.get('data_metadata', {})
                capabilities = message.get('capabilities', {})

                # Validate group and token
                group = fl_server.group_manager.groups.get(group_id)
                if not group:
                    await websocket.send_json({
                        'status': 'rejected',
                        'reason': 'group_not_found'
                    })
                else:
                    # Check if client is already registered in the group (activated via dashboard)
                    already_registered = client_id in group.clients

                    # Check if client has approved join request (activated via REST API)
                    has_approved_join = False
                    if not already_registered and not join_token and payload:
                        try:
                            from api.integration import get_platform_integration
                            platform = get_platform_integration()
                            user_id = payload.get("user_id")
                            join_status = platform.get_user_join_status(user_id, group_id)
                            if join_status and join_status.get("status") in ("approved", "joined"):
                                has_approved_join = True
                        except Exception:
                            pass

                    token_valid = (join_token and join_token == group.join_token)

                    if not already_registered and not has_approved_join and not token_valid:
                        await websocket.send_json({
                            'status': 'rejected',
                            'reason': 'invalid_token'
                        })
                    else:
                        # Register client - be more lenient
                        try:
                            logger = logging.getLogger(__name__)
                            logger.info(f"[REGISTER] Registering client {client_id} to group {group_id}")
                            success = fl_server.group_manager.register_client(
                                client_id=client_id,
                                group_id=group_id,
                                client_info={
                                    'has_gpu': capabilities.get('has_gpu', False),
                                    'device': capabilities.get('device', 'cpu'),
                                    'data_metadata': data_metadata,
                                    'connection': 'websocket'
                                }
                            )
                            if success:
                                group = fl_server.group_manager.groups[group_id]
                                logger.info(f"[REGISTER] Client {client_id} registered. Group now has {len(group.clients)} clients: {list(group.clients.keys())}")
                                # Register websocket for sending messages to client
                                fl_server.connection_manager.register_client(client_id, websocket)
                                await websocket.send_json({
                                    'status': 'registered',
                                    'client_id': client_id,
                                    'group_id': group_id,
                                    'model_id': group.model_id
                                })
                            else:
                                await websocket.send_json({
                                    'status': 'rejected',
                                    'reason': 'registration_failed'
                                })
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.error(f"Registration error: {e}")
                            await websocket.send_json({
                                'status': 'rejected',
                                'reason': f'registration_error: {str(e)}'
                            })

            elif message.get('type') == 'update':
                # Check if group is training
                try:
                    logger = logging.getLogger(__name__)
                    client_id = message.get('update', {}).get('client_id')
                    received_meta = message.get('update', {}).get('meta', {})
                    logger.info(f"[UPDATE-RECV] Client {client_id}: acc={received_meta.get('train_accuracy', 0):.4f}, loss={received_meta.get('train_loss', 0):.4f}")
                    group = fl_server.group_manager.get_client_group(client_id)

                    if not group:
                        logger.warning(f"[UPDATE] Client {client_id} not found in any group")
                        await websocket.send_json({'status': 'rejected', 'reason': 'group_not_found'})
                        continue

                    if not group.is_training:
                        logger.warning(f"[UPDATE] Group {group.group_id} not training")
                        await websocket.send_json({'status': 'rejected', 'reason': 'training_not_started'})
                        continue

                    logger.info(f"[UPDATE] Processing update for client {client_id} in group {group.group_id}")
                    update_payload = fl_server.group_manager.normalize_update(message.get('update', {}))
                    update_result = fl_server.group_manager.process_client_update(client_id, update_payload)

                    meta = message.get('update', {}).get('meta', {})
                    acc = meta.get('train_accuracy', 0)
                    loss = meta.get('train_loss', 0)
                    logger = logging.getLogger(__name__)
                    logger.info(f"[UPDATE] Client {client_id} in group {group.group_id}: acc={acc:.4f}, loss={loss:.4f}")
                    fl_server.group_manager.log_event('client_update', f'Client {client_id} sent update', group.group_id, {
                        'client_id': client_id,
                        'accuracy': acc,
                        'loss': loss
                    })

                    if update_result.get('triggered') and update_result.get('aggregate'):
                        agg_result = fl_server.group_manager.aggregate_group(group.group_id)
                        if agg_result:
                            await fl_server.group_manager.broadcast_to_group(group.group_id, {
                                'type': 'model_update',
                                'version': agg_result['version'],
                                'group_id': group.group_id,
                                'accuracy': agg_result.get('accuracy', 0),
                                'loss': agg_result.get('loss', 0)
                            })

                            # Always trigger next training round after aggregation
                            if group.is_training:
                                asyncio.create_task(
                                    fl_server.group_manager.trigger_clients_training(group.group_id)
                                )

                    await websocket.send_json({
                        'status': 'accepted',
                        'group_id': group.group_id,
                        'triggered': update_result.get('triggered', False),
                        'window_status': update_result.get('window_status')
                    })
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Update handling error: {e}")
                    await websocket.send_json({'status': 'error', 'reason': 'update_failed'})

            elif message.get('type') == 'metrics':
                try:
                    logger = logging.getLogger(__name__)
                    client_id = message.get('client_id')
                    logger.debug(f"Received metrics from {client_id}")

                    group = fl_server.group_manager.get_client_group(client_id)

                    if not group:
                        logger.warning(f"Group not found for client {client_id}")
                        await websocket.send_json({'status': 'rejected', 'reason': 'group_not_found'})
                        continue

                    metrics = message.get('meta', {})
                    if client_id in group.clients:
                        group.clients[client_id]['last_update'] = time.time()
                        group.clients[client_id]['local_accuracy'] = metrics.get('train_accuracy', 0)
                        group.clients[client_id]['local_loss'] = metrics.get('train_loss', 0)
                        logger.debug(f"Updated client {client_id} metrics: acc={metrics.get('train_accuracy', 0):.4f}, loss={metrics.get('train_loss', 0):.4f}")

                    fl_server.group_manager.log_event('client_metrics', f'Client {client_id} metrics', group.group_id, {
                        'client_id': client_id,
                        'accuracy': metrics.get('train_accuracy', 0),
                        'loss': metrics.get('train_loss', 0)
                    })

                    logger.debug(f"Sending metrics acknowledgment to {client_id}")
                    await websocket.send_json({'status': 'accepted', 'type': 'metrics'})
                    logger.info(f"Metrics from {client_id} processed successfully")
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Metrics handling error: {e}", exc_info=True)
                    try:
                        await websocket.send_json({'status': 'error', 'reason': 'metrics_failed'})
                    except Exception as send_err:
                        logger.error(f"Failed to send error response: {send_err}")

    except WebSocketDisconnect:
        logger = logging.getLogger(__name__)
        logger.info("WebSocket disconnected normally")
        fl_server.connection_manager.disconnect(websocket)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"WebSocket error: {e}", exc_info=True)
        fl_server.connection_manager.disconnect(websocket)


def register_socketio_handlers(socket_manager):
    """Register Socket.IO event handlers on the given SocketManager."""

    @socket_manager.on('connect')
    async def connect(sid, environ):
        print(f"Client connected: {sid}")

    @socket_manager.on('disconnect')
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")

    @socket_manager.on('register')
    async def register(sid, data):
        """Handle client registration via Socket.IO"""
        from networking.state import get_fl_server
        fl_server = get_fl_server()

        client_id = data.get('client_id')
        capabilities = data.get('capabilities', {})

        result = await fl_server.handle_client_register(client_id, capabilities)
        await socket_manager.emit('registered', result, room=sid)

    @socket_manager.on('update')
    async def handle_update(sid, data):
        """Handle client update via Socket.IO"""
        from networking.state import get_fl_server
        fl_server = get_fl_server()

        try:
            update = ClientUpdate(**data)
            result = await fl_server.handle_client_update(update)
            await socket_manager.emit('update_ack', result, room=sid)
        except Exception as e:
            await socket_manager.emit('error', {'message': str(e)}, room=sid)
