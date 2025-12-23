import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import type { PaginationMeta } from '@/lib/types';

// Webhook event status
export type WebhookEventStatus = 'PENDING' | 'SUCCESS' | 'FAILED';

// Webhook list item
export interface WebhookEventListItem {
  id: string;
  notificationType: string;
  subtype: string | null;
  originalTransactionId: string | null;
  status: WebhookEventStatus;
  userId: string | null;
  receivedAt: string;
  processedAt: string | null;
}

// Webhook event detail
export interface WebhookEventDetail extends WebhookEventListItem {
  transactionId: string | null;
  bundleId: string | null;
  payload: Record<string, unknown>;
  errorMessage: string | null;
  retryCount: number;
  lastRetryAt: string | null;
  createdAt: string;
  user?: {
    id: string;
    email: string;
    name: string;
  } | null;
}

// Webhooks list response
export interface WebhooksListResponse {
  events: WebhookEventListItem[];
  pagination: PaginationMeta;
}

// Webhook stats
export interface WebhookStats {
  total: number;
  pending: number;
  success: number;
  failed: number;
  byType: { type: string; count: number }[];
}

interface UseWebhooksParams {
  notificationType?: string;
  status?: WebhookEventStatus;
  startDate?: string;
  endDate?: string;
  originalTransactionId?: string;
  page?: number;
  limit?: number;
}

/**
 * Hook for fetching paginated webhook events
 */
export function useWebhooks(params: UseWebhooksParams = {}) {
  const { notificationType, status, startDate, endDate, originalTransactionId, page = 1, limit = 20 } = params;

  return useQuery<WebhooksListResponse>({
    queryKey: ['webhooks', { notificationType, status, startDate, endDate, originalTransactionId, page, limit }],
    queryFn: async () => {
      const response = await adminApi.getWebhooks({
        notificationType: notificationType || undefined,
        status: status || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        page,
        limit,
      });
      // Transform API response to match interface
      const apiResponse = response as { webhooks: WebhookEventListItem[]; pagination: PaginationMeta };
      return {
        events: apiResponse.webhooks,
        pagination: apiResponse.pagination,
      };
    },
  });
}

/**
 * Hook for fetching a single webhook event detail
 */
export function useWebhook(eventId: string | null) {
  return useQuery<WebhookEventDetail>({
    queryKey: ['webhook', eventId],
    queryFn: async () => {
      if (!eventId) throw new Error('Event ID is required');
      const response = await adminApi.getWebhook(eventId);
      return response as WebhookEventDetail;
    },
    enabled: !!eventId,
  });
}

/**
 * Hook for retrying a webhook event
 */
export function useRetryWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (eventId: string) => {
      return adminApi.retryWebhook(eventId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['webhooks'] });
      queryClient.invalidateQueries({ queryKey: ['webhook'] });
    },
  });
}
