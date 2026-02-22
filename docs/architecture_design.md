# Federated Learning Platform - Architecture Design

## 1. User Account System

### 1.1 Database Schema (users.db)

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'client', 'observer')),
    email TEXT,
    full_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Trust scores linked to user accounts
CREATE TABLE trust_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    group_id TEXT,
    score REAL DEFAULT 1.0,
    quarantined BOOLEAN DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, group_id)
);

-- Join requests (token request/approval)
CREATE TABLE join_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by INTEGER REFERENCES users(id),
    token_delivered BOOLEAN DEFAULT 0,
    token_delivered_at TIMESTAMP
);

-- Secure messages (token delivery)
CREATE TABLE secure_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_user_id INTEGER REFERENCES users(id),
    to_user_id INTEGER REFERENCES users(id),
    group_id TEXT,
    message_type TEXT NOT NULL,
    encrypted_content BLOB NOT NULL,
    nonce BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    UNIQUE(from_user_id, to_user_id, group_id, message_type)
);
```

### 1.2 Authentication Flow

```
Signup: POST /api/auth/signup
  ├── Validate input (username, password, role)
  ├── Hash password (bcrypt)
  ├── Store in users table
  └── Return JWT token

Login: POST /api/auth/login
  ├── Validate credentials
  ├── Check role
  └── Return role-specific JWT token

Token Payload:
{
  "sub": "username",
  "role": "admin|client|observer",
  "user_id": 123,
  "exp": ...
}
```

## 2. Secure Group Joining

### 2.1 Token Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  REQUEST    │────▶│  APPROVAL   │────▶│  DELIVERY   │────▶│  VALIDATION │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      │ Client requests   │ Admin approves    │ Secure message   │ Client uses
      │ to join group     │ or rejects        │ to deliver token │ token to join
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
  - group_id          - Notification      - AES-256-GCM       - WebSocket
  - user_id           - Update status     - Per-message         handshake
  - metadata          - Log event           nonce             - Token validated
                                        - Encrypted payload
```

### 2.2 Security Properties

- **Token Generation**: UUID4 + timestamp hash (256-bit entropy)
- **Expiry**: Default 24 hours, configurable
- **Replay Protection**: One-time use with database tracking
- **Secure Delivery**: AES-256-GCM encryption
- **Validation**: HMAC signature verification

## 3. Notification System

### 3.1 Architecture

```
Event Bus (In-Memory)
     │
     ├──▶ WebSocket Manager ──▶ Real-time Clients
     │
     ├──▶ Notification Queue ──▶ Persistent Storage
     │
     └──▶ Event Logger ──▶ Database
```

### 3.2 Event Types

```python
NOTIFICATION_TYPES = {
    # Group events
    'GROUP_CREATED': 'info',
    'GROUP_STARTED': 'info',
    'GROUP_PAUSED': 'warning',
    'GROUP_STOPPED': 'info',
    'GROUP_COMPLETED': 'success',
    
    # Training events
    'TRAINING_STARTED': 'info',
    'TRAINING_ROUND_COMPLETE': 'info',
    'MODEL_UPDATED': 'info',
    
    # Join events
    'JOIN_REQUEST': 'info',
    'JOIN_APPROVED': 'success',
    'JOIN_REJECTED': 'warning',
    
    # Error events
    'TRAINING_FAILED': 'error',
    'CONNECTION_LOST': 'warning',
    'AGGREGATION_FAILED': 'error',
}
```

## 4. Trust Score Integration

### 4.1 Account-Based Trust

```
┌─────────────────────────────────────────────────────────────┐
│                    Trust Score Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Account ──▶ Trust Score ──▶ Aggregation Weight       │
│       │                │                   │                 │
│       │                │                   │                 │
│       ▼                ▼                   ▼                 │
│  - user_id        - Account-based    - Weighted average    │
│  - username       - Persistent       - Trimmed mean        │
│  - role           - Per-group        - Hybrid aggregation  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Trust Update Triggers

- On each training round completion
- On model update submission
- On anomaly detection
- On admin manual adjustment

## 5. Gemini Model Recommendation

### 5.1 API Integration

```
Client Metadata ──▶ Gemini API ──▶ Model Recommendation
      │                            │
      │                            │
      ▼                            ▼
- Dataset size              - Model architecture
- Class distribution        - Parameter count
- Hardware capability       - Recommended config
- Network bandwidth          - Expected accuracy
```

### 5.2 Prompt Template

```
You are a federated learning model recommendation system.
Given the following client metadata, recommend an appropriate model:

Client Metadata:
- Dataset size: {size}
- Classes: {num_classes}
- Distribution: {distribution}
- Hardware: {capability}
- Network: {bandwidth}

Constraints:
- Must work in federated setting
- Prioritize privacy and efficiency
- Consider client heterogeneity

Respond with JSON:
{
  "model_type": "cnn|mlp|transformer",
  "recommended_params": "small|medium|large",
  "architecture": {...},
  "expected_accuracy": 0.XX,
  "reasoning": "..."
}
```

## 6. Heterogeneous Model Aggregation

### 6.1 Architecture

```
Baseline Model (Admin-selected)
    │
    ├── Client A (CNN-Large) ──▶ Parameter Mapping ──┐
    ├── Client B (CNN-Small) ──▶ Parameter Mapping ──┤
    └── Client C (MLP)         ──▶ Parameter Mapping ─┼──▶ Aggregated Update
                                                     │
                                                     ▼
                                            Layer Alignment Layer
```

### 6.2 Parameter Mapping Strategy

```python
class HeterogeneousAggregator:
    """
    Handles aggregation of models with different architectures.
    """
    
    # Strategy 1: Parameter Name Matching
    # Match parameters by name (e.g., "conv1.weight")
    
    # Strategy 2: Layer Type Matching
    # Match layers by type (Conv2d -> Conv2d)
    
    # Strategy 3: Partial Updates
    # Only update shared parameters, keep unique ones
    
    # Strategy 4: Projection
    # Learnable projection between different dimensions
```

### 6.3 Aggregation Algorithm

```
For each client update:
  1. Detect model architecture
  2. Extract shared parameters
  3. Map to baseline parameter space
  4. Apply aggregation (FedAvg/Robust)
  5. Map back to baseline model
```

## 7. Separate Inference Module

### 7.1 Privacy-Preserving Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option 1: Full Model Distribution                              │
│  ┌──────────┐    ┌────────────┐    ┌────────────────────┐    │
│  │  Server  │───▶│ Global Model│───▶│  Client Inference │    │
│  │          │    │  (full)     │    │     (download)    │    │
│  └──────────┘    └────────────┘    └────────────────────┘    │
│                                                                  │
│  Option 2: Parameter-Efficient Transfer                         │
│  ┌──────────┐    ┌────────────┐    ┌────────────────────┐    │
│  │  Server  │───▶│  Delta/    │───▶│  Client Model     │    │
│  │          │    │  Adapters  │    │  Apply + Delta    │    │
│  └──────────┘    └────────────┘    └────────────────────┘    │
│                                                                  │
│  Option 3: Server-Side Inference (Privacy-First)               │
│  ┌──────────┐    ┌────────────┐    ┌────────────────────┐    │
│  │  Client  │───▶│   Data     │───▶│  Server Inference  │    │
│  │ (local)  │    │ (encrypted)│    │   (result only)   │    │
│  └──────────┘    └────────────┘    └────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Inference API

```python
# Server-side inference endpoint
POST /api/inference/predict
{
    "model_id": "group_model_v3",
    "input_data": encrypted_bytes,
    "return_type": "class|probability|embedding"
}

# Client-side inference module
class InferenceModule:
    def __init__(self, model_path, config):
        self.model = load_model(model_path)
    
    def predict(self, input_data):
        # Local inference
        return result
```

## 8. UI Dashboard Architecture

### 8.1 Role-Based Views

```
┌─────────────────────────────────────────────────────────────┐
│                     Login Page                              │
│         (Admin / Client / Observer)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │  Admin  │       │ Client  │       │Observer │
   │  View   │       │  View   │       │  View   │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                │                 │
        ▼                ▼                 ▼
   - Create Groups  - Join Groups     - View Only
   - Manage Groups  - Train           - Metrics
   - View All       - Inference       - Logs
   - Approve Joins - Trust Score
   - Notifications - Notifications
```

### 8.2 Client Dashboard Features

- Group discovery and join requests
- Training status and progress
- Model download and local inference
- Trust score display
- Notification center

## 9. Security Considerations

### 9.1 Authentication
- JWT tokens with short expiry
- Role-based access control
- Password hashing with bcrypt

### 9.2 Token Security
- AES-256-GCM encryption for token delivery
- One-time use tokens with database tracking
- HMAC signature verification

### 9.3 Data Privacy
- No raw data leaves client
- Only metadata shared for recommendations
- Server-side inference option

### 9.4 Network Security
- HTTPS/WSS only
- Input validation
- SQL injection prevention

## 10. Implementation Priority

1. **Phase 1**: User Account System + Signup
2. **Phase 2**: Secure Group Joining
3. **Phase 3**: Notification System
4. **Phase 4**: Trust Score Integration
5. **Phase 5**: Gemini Model Recommendation
6. **Phase 6**: Heterogeneous Aggregation
7. **Phase 7**: Inference Module
8. **Phase 8**: Client UI Dashboard
