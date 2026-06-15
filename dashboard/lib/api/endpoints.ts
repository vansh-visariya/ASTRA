import { api } from './client';
import type {
  SystemMetrics,
  Group,
  LogEntry,
  Notification,
  JoinRequest,
  TrainingStatus,
  TrustData,
  Model,
  Recommendation,
  AuthResponse,
  ApiListResponse,
} from './types';

export const getSystemMetrics = () => api.get<SystemMetrics>('/api/system/metrics');

export const getGroups = () => api.get<ApiListResponse<Group>>('/api/groups');

export const getGroup = (id: string) => api.get<{ group: Group }>(`/api/groups/${id}`);

export const createGroup = (body: Record<string, unknown>) =>
  api.post<{ group_id: string }>('/api/groups', body);

export const controlGroup = (id: string, action: 'start' | 'pause' | 'resume' | 'stop') =>
  api.post<{ status: string }>(`/api/groups/${id}/${action}`);

export const deleteGroup = (id: string) =>
  api.del<{ status: string }>(`/api/groups/${id}`);

export const getTrainingStatus = () => api.get<TrainingStatus>('/api/client/training-status');

export const getModels = () => api.get<ApiListResponse<Model>>('/api/models');

export const registerHfModel = (body: Record<string, unknown>) =>
  api.post<{ model_id: string }>('/api/models/register/hf', body);

export const registerCustomModel = (body: Record<string, unknown>) =>
  api.post<{ model_id: string }>('/api/models/register', body);

export const registerArchitecture = (body: {
  model_id: string;
  architecture_path: string;
  model_type?: string;
  config?: Record<string, unknown>;
}) => api.post<{ status: string }>('/api/models/register/architecture', body);

export const getJoinRequests = (groupId?: string) => {
  const path = groupId ? `/api/join/join-requests?group_id=${groupId}` : '/api/join/join-requests';
  return api.get<ApiListResponse<JoinRequest>>(path);
};

export const requestJoin = (body: { group_id: string; message?: string }) =>
  api.post<{ status: string }>('/api/join/join-request', body);

export const approveJoin = (body: { request_id: number }) =>
  api.post<{ status: string }>('/api/join/join-requests/approve', body);

export const rejectJoin = (body: { request_id: number }) =>
  api.post<{ status: string }>('/api/join/join-requests/reject', body);

export const getMyJoinStatus = (groupId: string) =>
  api.get<{ status: string }>(`/api/join/my-requests/${groupId}`);

export const activateJoin = (groupId: string) =>
  api.post<{ status: string }>(`/api/join/activate/${groupId}`);

export const getNotifications = (params?: { limit?: number; unread_only?: boolean }) => {
  const searchParams = new URLSearchParams();
  if (params?.limit) searchParams.set('limit', String(params.limit));
  if (params?.unread_only) searchParams.set('unread_only', 'true');
  const query = searchParams.toString();
  return api.get<ApiListResponse<Notification>>(`/api/notifications${query ? `?${query}` : ''}`);
};

export const getUnreadCount = () =>
  api.get<{ count: number }>('/api/notifications/unread-count');

export const markNotificationRead = (id: number) =>
  api.post<{ status: string }>(`/api/notifications/${id}/read`);

export const markAllNotificationsRead = () =>
  api.post<{ status: string }>('/api/notifications/read-all');

export const getAllTrustScores = (groupId?: string) => {
  const path = groupId ? `/api/trust/scores?group_id=${groupId}` : '/api/trust/scores';
  return api.get<ApiListResponse<TrustData>>(path);
};

export const getTrustScore = (userId: number | string) =>
  api.get<TrustData>(`/api/trust/scores/${userId}`);

export const getLogs = (params?: { limit?: number; event_type?: string; group_id?: string }) => {
  const searchParams = new URLSearchParams();
  if (params?.limit) searchParams.set('limit', String(params.limit));
  if (params?.event_type) searchParams.set('event_type', params.event_type);
  if (params?.group_id) searchParams.set('group_id', params.group_id);
  const query = searchParams.toString();
  return api.get<ApiListResponse<LogEntry>>(`/api/logs${query ? `?${query}` : ''}`);
};

export const getRecommendations = (body: Record<string, unknown>) =>
  api.post<{ recommendations: Recommendation[] }>('/api/recommendations/unified', body);

export const addHuggingFaceModel = (body: Record<string, unknown>) =>
  api.post<{ model_id: string }>('/api/recommendations/add-huggingface', body);

export const login = (username: string, password: string) =>
  api.post<AuthResponse>('/api/auth/login', { username, password });

export const signup = (body: {
  username: string;
  password: string;
  role: string;
  name: string;
  email?: string;
}) => api.post<AuthResponse>('/api/auth/signup', body);
