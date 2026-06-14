import { API_URL } from '@/lib/config';

export function buildClientCommand(
  serverUrl: string = API_URL,
  clientId?: string,
  groupId?: string,
): string {
  const base = 'python -m astra.client.cli';
  const parts: string[] = [base];
  parts.push(`--server ${serverUrl}`);
  if (clientId) parts.push(`--client-id ${clientId}`);
  if (groupId) parts.push(`--group-id ${groupId}`);
  return parts.join(' ');
}
