# SKILL: Distributed Networking for Federated Learning

## PURPOSE
Build production-grade, fault-tolerant networking between FL server and clients using gRPC + WebSocket. Covers connection management, message framing, retries, and TLS security.

---

## ARCHITECTURE

```
networking/
├── grpc/
│   ├── proto/
│   │   └── fl_service.proto      # Service definition (source of truth)
│   ├── server.py                 # gRPC server implementation
│   ├── client_stub.py            # Client-side stub with retry
│   └── interceptors.py           # Auth, logging, metrics interceptors
├── websocket/
│   ├── ws_server.py              # Dashboard real-time events
│   └── event_bus.py              # Internal pub/sub → WebSocket bridge
├── transport/
│   ├── chunked_transfer.py       # Large model transfer (chunking)
│   ├── compression.py            # zstd/lz4 model compression
│   └── rate_limiter.py           # Per-client bandwidth limits
└── security/
    ├── tls_config.py             # mTLS setup
    └── auth.py                   # JWT token validation interceptor
```

---

## PROTO DEFINITION (fl_service.proto)
```protobuf
syntax = "proto3";
package fl;

service FLService {
    rpc Register       (RegisterRequest)     returns (RegisterResponse);
    rpc GetGlobalModel (ModelRequest)        returns (stream ModelChunk);   // streaming for large models
    rpc SubmitUpdate   (stream UpdateChunk)  returns (SubmitResponse);      // client streaming
    rpc Heartbeat      (HeartbeatRequest)    returns (HeartbeatResponse);
    rpc GetRoundStatus (RoundStatusRequest)  returns (RoundStatusResponse);
}

message RegisterRequest {
    string client_id     = 1;
    string device_info   = 2;
    int64  dataset_size  = 3;
    map<string, string> capabilities = 4;   // "gpu": "true", "memory_gb": "16"
}

message ModelChunk {
    bytes  data          = 1;
    int32  chunk_index   = 2;
    int32  total_chunks  = 3;
    string checksum      = 4;   // sha256 of full payload, sent in last chunk
    string model_version = 5;
    int32  round_id      = 6;
}

message UpdateChunk {
    bytes  data        = 1;
    int32  chunk_index = 2;
    int32  total_chunks = 3;
    string client_id   = 4;
    int32  round_id    = 5;
    int64  sample_count = 6;
}

message HeartbeatRequest {
    string client_id   = 1;
    int32  round_id    = 2;   // Current round client is on
    string status      = 3;   // "idle" | "training" | "uploading"
    float  cpu_percent = 4;
    float  memory_percent = 5;
}
```

---

## CRITICAL RULES — ALWAYS FOLLOW

### 1. Always Use Streaming for Model Transfer
```python
# ✅ CORRECT — Stream large models in chunks (models can be GB)
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks

async def GetGlobalModel(self, request, context):
    model_bytes = serialize_model(self.global_model)
    total_chunks = math.ceil(len(model_bytes) / CHUNK_SIZE)
    checksum = hashlib.sha256(model_bytes).hexdigest()
    
    for i in range(total_chunks):
        chunk = model_bytes[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        is_last = (i == total_chunks - 1)
        yield fl_pb2.ModelChunk(
            data=chunk,
            chunk_index=i,
            total_chunks=total_chunks,
            checksum=checksum if is_last else "",
            round_id=self.current_round,
        )

# ❌ WRONG — Crashes on large models, hits gRPC 4MB limit
async def GetGlobalModel(self, request, context):
    model_bytes = serialize_model(self.global_model)
    return fl_pb2.ModelResponse(data=model_bytes)  # gRPC max message size = 4MB default
```

### 2. Client Stub with Exponential Backoff Retry
```python
import grpc
from grpc import aio
import asyncio
import random

class FLClientStub:
    def __init__(self, server_addr: str, credentials=None):
        if credentials:
            self.channel = aio.secure_channel(server_addr, credentials)
        else:
            self.channel = aio.insecure_channel(server_addr)
        self.stub = fl_pb2_grpc.FLServiceStub(self.channel)

    async def _retry(self, fn, max_attempts=5, base_delay=1.0):
        """Exponential backoff with jitter."""
        for attempt in range(max_attempts):
            try:
                return await fn()
            except grpc.RpcError as e:
                if e.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                ):
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    raise  # Don't retry auth errors, invalid args, etc.

    async def submit_update(self, update_bytes: bytes, round_id: int, sample_count: int):
        async def _send():
            async def chunk_gen():
                total = math.ceil(len(update_bytes) / CHUNK_SIZE)
                for i in range(total):
                    yield fl_pb2.UpdateChunk(
                        data=update_bytes[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE],
                        chunk_index=i,
                        total_chunks=total,
                        round_id=round_id,
                        sample_count=sample_count,
                    )
            return await self.stub.SubmitUpdate(chunk_gen())
        
        return await self._retry(_send)
```

### 3. gRPC Interceptors for Auth + Logging
```python
import time
import jwt
from grpc import aio, StatusCode

class AuthInterceptor(aio.ServerInterceptor):
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        # Public methods that don't need auth
        self.public_methods = {"/fl.FLService/Register"}

    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        if method in self.public_methods:
            return await continuation(handler_call_details)
        
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get("authorization", "").removeprefix("Bearer ")
        
        if not token:
            async def deny(req, ctx):
                await ctx.abort(StatusCode.UNAUTHENTICATED, "Missing token")
            return grpc.unary_unary_rpc_method_handler(deny)
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            # Attach client_id to context for downstream use
            handler_call_details.invocation_metadata += (("x-client-id", payload["sub"]),)
        except jwt.ExpiredSignatureError:
            async def deny(req, ctx):
                await ctx.abort(StatusCode.UNAUTHENTICATED, "Token expired")
            return grpc.unary_unary_rpc_method_handler(deny)
        
        return await continuation(handler_call_details)
```

### 4. TLS / mTLS Configuration
```python
import grpc

def create_server_credentials(
    cert_path: str,
    key_path: str,
    ca_cert_path: str = None,    # For mTLS (client cert verification)
) -> grpc.ServerCredentials:
    with open(key_path, "rb") as f:
        private_key = f.read()
    with open(cert_path, "rb") as f:
        certificate_chain = f.read()
    
    root_certificates = None
    require_client_auth = False
    if ca_cert_path:
        with open(ca_cert_path, "rb") as f:
            root_certificates = f.read()
        require_client_auth = True  # mTLS — verify client cert
    
    return grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=root_certificates,
        require_client_auth=require_client_auth,
    )
```

### 5. WebSocket Server for Real-Time Dashboard Events
```python
import asyncio
import json
from typing import Set
from websockets.server import serve, WebSocketServerProtocol

class EventBus:
    """Pub/sub bridge: FL events → connected WebSocket clients (dashboard)."""
    
    def __init__(self):
        self._clients: Set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocketServerProtocol):
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocketServerProtocol):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, event: dict):
        """Non-blocking broadcast; slow clients are dropped."""
        message = json.dumps(event)
        if not self._clients:
            return
        async with self._lock:
            targets = list(self._clients)
        
        results = await asyncio.gather(
            *[self._send_safe(ws, message) for ws in targets],
            return_exceptions=True,
        )
        # Remove disconnected clients
        dead = {ws for ws, r in zip(targets, results) if isinstance(r, Exception)}
        if dead:
            async with self._lock:
                self._clients -= dead

    async def _send_safe(self, ws, message: str):
        await asyncio.wait_for(ws.send(message), timeout=5.0)

# Event schema — always use these keys
def make_event(event_type: str, data: dict) -> dict:
    return {
        "type": event_type,       # "round_started" | "client_joined" | "round_complete" | "client_failed"
        "timestamp": time.time(),
        "data": data,
    }
```

---

## CONNECTION LIFECYCLE

```
Client                              Server
  |                                   |
  |--- Register(client_id, caps) ---> |  (auth not required)
  |<-- RegisterResponse(token) ------ |  (JWT issued)
  |                                   |
  |--- GetGlobalModel(round_id) ----> |  (token required)
  |<== ModelChunk stream ============ |  (chunked, with checksum)
  |                                   |
  |     [local training happens]      |
  |                                   |
  |=== SubmitUpdate stream =========> |  (token required)
  |<-- SubmitResponse(accepted) ----- |
  |                                   |
  |--- Heartbeat every 30s ---------> |  (server detects dead clients)
```

---

## COMMON NETWORKING BUGS & FIXES

| Bug | Symptom | Fix |
|-----|---------|-----|
| gRPC message size exceeded | `StatusCode.RESOURCE_EXHAUSTED` | Use streaming; set `max_receive_message_length=-1` |
| Channel not closed | File descriptor leak | Always `await channel.close()` in finally block |
| No deadline on RPCs | Client hangs forever | Add `timeout=` to every stub call |
| Heartbeat flood | Server CPU spike | Throttle: min 10s between heartbeats per client |
| WebSocket fan-out blocks event loop | Dashboard freezes under load | Use `asyncio.gather()` with `return_exceptions=True` |
| Token sent in URL query params | Security — logged in plaintext | Always use metadata/headers |
| No reconnect on channel drop | Client dies permanently | Implement `_retry()` with backoff |
| Chunk reassembly without ordering | Corrupted model | Sort by `chunk_index` before joining |

---

## CHUNK REASSEMBLY (CLIENT SIDE)
```python
async def download_global_model(stub, round_id: int) -> bytes:
    chunks = {}
    total = None
    expected_checksum = None
    
    async for chunk in stub.GetGlobalModel(fl_pb2.ModelRequest(round_id=round_id)):
        chunks[chunk.chunk_index] = chunk.data
        total = chunk.total_chunks
        if chunk.checksum:  # Last chunk carries checksum
            expected_checksum = chunk.checksum
    
    if len(chunks) != total:
        raise IncompleteTransferError(f"Got {len(chunks)}/{total} chunks")
    
    model_bytes = b"".join(chunks[i] for i in range(total))  # Sort by index
    
    actual_checksum = hashlib.sha256(model_bytes).hexdigest()
    if actual_checksum != expected_checksum:
        raise IntegrityError("Model download corrupted")
    
    return model_bytes
```

---

## SERVER STARTUP TEMPLATE
```python
async def serve():
    interceptors = [
        AuthInterceptor(secret_key=config.jwt_secret),
        LoggingInterceptor(),
        MetricsInterceptor(),
    ]
    server = aio.server(interceptors=interceptors)
    fl_pb2_grpc.add_FLServiceServicer_to_server(FLServicer(), server)
    
    credentials = create_server_credentials(
        config.tls_cert, config.tls_key, config.tls_ca
    )
    server.add_secure_port(f"[::]:{config.grpc_port}", credentials)
    
    # Graceful shutdown
    await server.start()
    
    async def shutdown():
        await server.stop(grace=30)  # 30s for in-flight RPCs to complete
    
    import signal
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown()))
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown()))
    
    await server.wait_for_termination()
```

---

## TESTING CHECKLIST
- [ ] Transfer 500MB model — verify all chunks arrive, checksum passes
- [ ] Kill server mid-transfer — client retries and recovers
- [ ] 100 concurrent client connections — no deadlocks
- [ ] Expired JWT rejected with UNAUTHENTICATED
- [ ] Heartbeat timeout triggers client eviction
- [ ] WebSocket broadcasts to 50 dashboard clients under load
- [ ] mTLS: client without cert is rejected
- [ ] SIGTERM triggers graceful shutdown (in-flight rounds complete)