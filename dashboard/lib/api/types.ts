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
  training_manifest?: TrainingManifest | null;
  metrics_history?: Array<{
    accuracy: number;
    loss: number;
    version: number;
    timestamp: number;
    clients: number;
  }>;
  window_status?: {
    pending_updates: number;
    window_size: number;
    time_elapsed: number;
    time_remaining: number;
    trigger_reason: string;
  };
  created_at?: string;
}

export interface TrainingManifest {
  contract_version?: number;
  model_id: string;
  is_peft?: boolean;
  target_modules?: string[];
  lora_rank?: number;
  lora_alpha?: number;
  expected_delta_bytes?: number;
  lr?: number;
  batch_size?: number;
  local_epochs?: number;
  optimizer?: string;
  loss_function?: string;
  max_grad_norm?: number;
  input_features?: string[];
  input_shape?: number[];
  num_classes?: number;
  label_type?: string;
  data_description?: string;
  preprocessing_steps?: string[];
  accepted_update_types?: string[];
  val_dataset?: string;
  val_metric?: string;
}

export interface ClientMeta {
  client_id?: string;
  status?: string;
  last_update?: number;
  update_count?: number;
  trust_score?: number;
  joined_at?: string;
  user_id?: number;
}

export interface Client {
  client_id: string;
  group_id: string;
  status: string;
  last_update: number;
  updates_count: number;
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
    full_name?: string;
    email?: string;
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
  announcements?: T[];
  messages?: T[];
  count?: number;
  [key: string]: unknown;
}

export interface Announcement {
  id: number;
  group_id: string;
  author_id: number;
  author_name: string;
  message: string;
  priority: 'info' | 'warning' | 'error';
  created_at: string;
}

export interface Message {
  id: number;
  group_id: string;
  sender_id: number;
  sender_name: string;
  sender_role: 'admin' | 'client' | 'observer';
  content: string;
  created_at: string;
}