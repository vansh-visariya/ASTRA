# SKILL: Web Dashboard (React + FastAPI)

## PURPOSE
Build a production-ready real-time monitoring dashboard for the FL platform. Covers React frontend, FastAPI backend, WebSocket live updates, charting, and state management.

---

## ARCHITECTURE

```
dashboard/
├── backend/                        # FastAPI
│   ├── main.py                     # App entry point, middleware
│   ├── routers/
│   │   ├── rounds.py               # /api/rounds endpoints
│   │   ├── clients.py              # /api/clients endpoints
│   │   ├── models.py               # /api/models endpoints
│   │   └── metrics.py              # /api/metrics endpoints
│   ├── websocket/
│   │   └── ws_handler.py           # WS endpoint — bridges EventBus
│   ├── db/
│   │   ├── database.py             # SQLAlchemy async setup
│   │   ├── models.py               # ORM models
│   │   └── crud.py                 # DB operations
│   └── schemas/
│       └── pydantic_schemas.py     # Request/Response schemas
│
└── frontend/                       # React + Vite
    ├── src/
    │   ├── components/
    │   │   ├── RoundMonitor.tsx     # Live round progress
    │   │   ├── ClientGrid.tsx       # Client status cards
    │   │   ├── MetricsChart.tsx     # Loss/accuracy curves (recharts)
    │   │   ├── ModelTable.tsx       # Model registry table
    │   │   └── ConnectionStatus.tsx # WS health indicator
    │   ├── hooks/
    │   │   ├── useWebSocket.ts      # Reconnecting WS hook
    │   │   ├── useFLData.ts         # SWR data fetching hooks
    │   │   └── useEventStream.ts    # Event stream reducer
    │   ├── store/
    │   │   └── flStore.ts           # Zustand global store
    │   └── api/
    │       └── client.ts           # Axios instance with interceptors
```

---

## CRITICAL RULES — ALWAYS FOLLOW

### 1. FastAPI — Always Use Async Database Sessions
```python
# ✅ CORRECT — Async session, proper dependency injection
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@router.get("/rounds/{round_id}", response_model=RoundResponse)
async def get_round(round_id: int, db: AsyncSession = Depends(get_db)):
    round_obj = await db.get(Round, round_id)
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    return round_obj

# ❌ WRONG — Sync session in async FastAPI blocks the event loop
def get_round(round_id: int, db = Depends(get_sync_db)):
    return db.query(Round).filter(Round.id == round_id).first()
```

### 2. WebSocket Endpoint — Handle Disconnect Gracefully
```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, token: str = Query(...)):
    # Authenticate before accepting
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    await websocket.accept()
    await event_bus.connect(websocket)
    
    try:
        # Send current state snapshot on connect
        snapshot = await get_system_snapshot()
        await websocket.send_json({"type": "snapshot", "data": snapshot})
        
        # Keep alive — ping every 30s
        while True:
            try:
                # Wait for client ping or disconnect
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("ping")  # Server-initiated keepalive
    except WebSocketDisconnect:
        pass  # Normal disconnect
    finally:
        await event_bus.disconnect(websocket)  # ALWAYS cleanup
```

### 3. Pydantic Schemas — Strict Validation
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class ClientStatus(str, Enum):
    IDLE      = "idle"
    TRAINING  = "training"
    UPLOADING = "uploading"
    OFFLINE   = "offline"
    FAILED    = "failed"

class ClientResponse(BaseModel):
    client_id: str
    status: ClientStatus
    last_heartbeat: datetime
    dataset_size: int = Field(gt=0)
    current_round: int | None
    train_loss: float | None
    is_online: bool
    
    @validator("client_id")
    def validate_client_id(cls, v):
        if not v or len(v) > 64:
            raise ValueError("client_id must be 1-64 characters")
        return v
    
    class Config:
        from_attributes = True  # SQLAlchemy ORM compat

class RoundResponse(BaseModel):
    round_id: int
    status: str
    num_clients_expected: int
    num_clients_completed: int
    progress_percent: float
    started_at: datetime
    completed_at: datetime | None
    global_loss: float | None
    global_accuracy: float | None
```

### 4. SQLAlchemy Async ORM Models
```python
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Round(Base):
    __tablename__ = "rounds"
    id              = Column(Integer, primary_key=True, index=True)
    status          = Column(SAEnum("waiting","collecting","aggregating","complete","failed"), default="waiting")
    num_clients     = Column(Integer, default=0)
    global_loss     = Column(Float, nullable=True)
    global_accuracy = Column(Float, nullable=True)
    started_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)
    model_version   = Column(String(64), nullable=True)
    
    client_updates  = relationship("ClientUpdate", back_populates="round")

class ClientUpdate(Base):
    __tablename__ = "client_updates"
    id           = Column(Integer, primary_key=True, index=True)
    round_id     = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    client_id    = Column(String(64), nullable=False, index=True)
    sample_count = Column(Integer, nullable=False)
    train_loss   = Column(Float, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    round = relationship("Round", back_populates="client_updates")

# Database setup
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

engine = create_async_engine(
    settings.database_url,           # "postgresql+asyncpg://..."
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,              # Detect stale connections
    echo=settings.debug,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

---

## FRONTEND: React Hooks

### useWebSocket Hook (Auto-Reconnecting)
```typescript
import { useEffect, useRef, useCallback, useState } from "react";

interface WSOptions {
  url: string;
  token: string;
  onMessage: (event: MessageEvent) => void;
  reconnectInterval?: number;  // ms, default 3000
  maxRetries?: number;
}

export function useWebSocket({ url, token, onMessage, reconnectInterval = 3000, maxRetries = 10 }: WSOptions) {
  const ws = useRef<WebSocket | null>(null);
  const retries = useRef(0);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(() => {
    if (retries.current >= maxRetries) return;
    
    const socket = new WebSocket(`${url}?token=${token}`);
    ws.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      retries.current = 0;
    };

    socket.onmessage = onMessage;

    socket.onclose = (e) => {
      setIsConnected(false);
      if (e.code !== 1000 && e.code !== 4001) {  // 4001 = auth failed, don't retry
        retries.current++;
        setTimeout(connect, reconnectInterval * Math.min(retries.current, 5)); // backoff
      }
    };

    socket.onerror = () => {
      socket.close();
    };
  }, [url, token, onMessage, reconnectInterval, maxRetries]);

  useEffect(() => {
    connect();
    return () => {
      ws.current?.close(1000, "Component unmounted");  // Clean close code
    };
  }, [connect]);

  const send = useCallback((data: string) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(data);
    }
  }, []);

  return { isConnected, send };
}
```

### Zustand Store
```typescript
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

interface ClientInfo {
  clientId: string;
  status: "idle" | "training" | "uploading" | "offline" | "failed";
  datasetSize: number;
  lastHeartbeat: string;
  currentRound: number | null;
  trainLoss: number | null;
}

interface RoundInfo {
  roundId: number;
  status: string;
  numClientsCompleted: number;
  numClientsExpected: number;
  globalLoss: number | null;
  globalAccuracy: number | null;
}

interface FLStore {
  clients: Record<string, ClientInfo>;
  rounds: RoundInfo[];
  currentRound: RoundInfo | null;
  metricsHistory: Array<{ round: number; loss: number; accuracy: number }>;
  
  // Actions
  updateClient: (client: ClientInfo) => void;
  addRoundUpdate: (round: RoundInfo) => void;
  handleEvent: (event: { type: string; data: unknown }) => void;
}

export const useFLStore = create<FLStore>()(
  immer((set) => ({
    clients: {},
    rounds: [],
    currentRound: null,
    metricsHistory: [],

    updateClient: (client) =>
      set((state) => { state.clients[client.clientId] = client; }),

    addRoundUpdate: (round) =>
      set((state) => {
        state.currentRound = round;
        const idx = state.rounds.findIndex((r) => r.roundId === round.roundId);
        if (idx >= 0) state.rounds[idx] = round;
        else state.rounds.unshift(round);
        
        if (round.status === "complete" && round.globalLoss !== null) {
          state.metricsHistory.push({
            round: round.roundId,
            loss: round.globalLoss!,
            accuracy: round.globalAccuracy!,
          });
        }
      }),

    handleEvent: (event) =>
      set((state) => {
        switch (event.type) {
          case "client_joined":
          case "client_update":
            state.clients[(event.data as ClientInfo).clientId] = event.data as ClientInfo;
            break;
          case "round_started":
          case "round_update":
          case "round_complete":
            const round = event.data as RoundInfo;
            state.currentRound = round;
            break;
          case "snapshot":
            // Full state from server on connect
            const snap = event.data as { clients: ClientInfo[]; rounds: RoundInfo[] };
            snap.clients.forEach(c => { state.clients[c.clientId] = c; });
            state.rounds = snap.rounds;
            break;
        }
      }),
  }))
);
```

### MetricsChart Component
```tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useFLStore } from "../store/flStore";

export function MetricsChart() {
  const history = useFLStore((s) => s.metricsHistory);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-3">Training Progress</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="round" label={{ value: "Round", position: "bottom" }} />
          <YAxis yAxisId="loss" label={{ value: "Loss", angle: -90, position: "insideLeft" }} />
          <YAxis yAxisId="acc" orientation="right" label={{ value: "Accuracy", angle: 90, position: "insideRight" }} />
          <Tooltip />
          <Legend />
          <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#ef4444" dot={false} name="Val Loss" />
          <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#22c55e" dot={false} name="Val Accuracy" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

## API ENDPOINTS REFERENCE

```
GET  /api/rounds?limit=20&offset=0    → Paginated round history
GET  /api/rounds/{id}                 → Round detail + client updates
GET  /api/clients                     → All registered clients + status
GET  /api/clients/{id}/history        → Client metrics history
GET  /api/models                      → Model registry
GET  /api/models/{version}/download   → Download model checkpoint
POST /api/rounds/{id}/stop            → Force stop a round (admin)
GET  /api/metrics/summary             → Latest global metrics
GET  /api/health                      → Server health check
WS   /ws/events?token=...            → Real-time event stream
```

---

## COMMON DASHBOARD BUGS & FIXES

| Bug | Symptom | Fix |
|-----|---------|-----|
| WebSocket reconnects in infinite loop | Browser DevTools shows 100s of connections | Check error code; don't retry on 4001 |
| Stale WS state in React closure | Events use old state values | Use `useRef` for WS, state in Zustand |
| Missing `finally` on WS disconnect | EventBus memory leak | Always call `event_bus.disconnect()` in finally |
| Pydantic v2 ORM mode | `ValidationError` on DB → schema conversion | Use `model_config = ConfigDict(from_attributes=True)` |
| React renders on every WS message | UI freezes under load | Use `useMemo`/`memo`; batch updates in Zustand |
| CORS errors in dev | Fetch blocked | Add FastAPI `CORSMiddleware` with `allow_origins` |
| Sync DB session in async route | `MissingGreenlet` error | Use `AsyncSession`; never sync inside async routes |
| N+1 queries in round list | Slow API response | Use `selectinload()` for relationships |

---

## FASTAPI STARTUP TEMPLATE
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(title="FL Dashboard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rounds_router, prefix="/api/rounds", tags=["rounds"])
app.include_router(clients_router, prefix="/api/clients", tags=["clients"])
app.include_router(models_router, prefix="/api/models", tags=["models"])

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

---

## TESTING CHECKLIST
- [ ] WS reconnects after server restart (no zombie connections)
- [ ] Dashboard shows correct round progress % in real-time
- [ ] 50 clients connect simultaneously — no UI lag
- [ ] Metrics chart updates live as rounds complete
- [ ] API returns 404, not 500, for missing resources
- [ ] Token expiry redirects to login (HTTP 401 → frontend)
- [ ] DB transactions rollback on exception (no partial writes)
- [ ] CORS allows frontend dev server, blocks random origins