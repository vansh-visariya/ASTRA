export interface SystemMetrics {
  total_groups: number;
  active_groups: number;
  total_participants: number;
  active_participants: number;
  dp_enabled_groups: number;
  total_aggregations: number;
  latest_group_id?: string | null;
  latest_accuracy?: number;
  latest_loss?: number;
  latest_version?: number;
  latest_timestamp?: number;
}

export interface Group {
  group_id: string;
  model_id: string;
  status: string;
  is_training: boolean;
  is_locked?: boolean;
  clients: Record<string, ClientMeta>;
  window_size: number;
  time_limit: number;
  join_token: string;
  model_version?: number;
  completed_rounds?: number;
  latest_accuracy?: number;
  latest_loss?: number;
  metrics_source?: 'server' | 'unverified';
  client_count?: number;
  active_clients?: string[];
  config?: Record<string, unknown>;
  training_manifest?: Record<string, unknown> | null;
  window_status?: {
    pending_updates: number;
    window_size: number;
    time_elapsed: number;
    time_remaining: number;
    waiting: boolean;
  };
  created_at?: string;
}

export interface ClientMeta {
  client_id?: string;
  status?: string;
  last_update?: number;
  update_count?: number;
  trust_score?: number;
  joined_at?: string;
}

export interface Client {
  client_id: string;
  group_id: string;
  status: string;
  last_update: number;
  update_count: number;
  trust_score: number;
  joined_at: string;
}

export interface LogEntry {
  timestamp: number | string;
  type: string;
  message: string;
  group_id?: string;
  details?: Record<string, unknown>;
}

export interface Notification {
  id: number;
  type: string;
  priority: string;
  title: string;
  message: string;
  group_id?: string;
  created_at: string;
  read: boolean;
}

export interface JoinRequest {
  id: number;
  user_id: number;
  username: string;
  group_id: string;
  status: string;
  created_at: string;
}

export interface TrustData {
  user_id: number;
  score: number;
  group_id?: string;
  participation_count?: number;
  last_updated?: string;
}

export interface Model {
  model_id: string;
  model_type: string;
  architecture?: string;
  dataset?: string;
  source?: string;
  created_at?: string;
  is_peft?: boolean;
}

export interface Recommendation {
  model_id: string;
  model_type: string;
  source: string;
  reasoning: string;
  params?: Record<string, unknown>;
  accuracy?: number;
}

export interface AuthResponse {
  token: string;
  user: {
    id?: number;
    username: string;
    role: 'admin' | 'client' | 'observer';
    name: string;
  };
}

export interface ApiListResponse<T> {
  groups?: T[];
  notifications?: T[];
  logs?: T[];
  models?: T[];
  join_requests?: T[];
  requests?: T[];
  clients?: T[];
  count?: number;
  [key: string]: unknown;
}